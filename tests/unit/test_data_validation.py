"""
Data Cleaning, Outlier Filtering & Normalization Robustness Test Suite.

Coverage:
1. Zero-IQR behavior: uniform cohorts (e.g. 50 quotes at INR 4,500), minor deltas (+1%, -2%),
   true anomalies (+400%, INR 200, INR 90,000), bounds clamping, and storage filtering.
2. Flight number normalization & canonical deduplication: messy strings ('ai-101', 'AI 101',
   'AI101', 'AI - 101'), carrier priority over OTAs, OTA lowest fare fallback, and orthogonal isolation.
3. Boundary values and edge cases: zero/negative fares, exact domain boundaries (INR 998.99/999/25000/25000.01),
   extreme dates, advance days bounds, and missing optional fields.
"""

from collections import defaultdict
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import itertools
import re
from typing import List

import numpy as np
from pydantic import ValidationError
import pytest

from apix.pipeline.decomposition import decompose_fare, decompose_quote
from apix.pipeline.outlier import (
    OutlierDetector,
    evaluate_outlier_status,
    filter_iqr_bounds,
    is_domain_ceiling_violation,
    is_domain_floor_violation,
    is_temporal_surge_violation,
    is_cross_source_variance_violation,
)
from apix.pipeline.quality import (
    canonical_flight_key,
    clean_quote,
    clean_quotes_batch,
    compute_dqs_score,
    deduplicate_quotes,
    handle_sold_out,
)
from apix.pipeline.schemas import (
    CleanedFareQuote,
    DecomposedFare,
    OutlierReason,
    RawFareQuote,
)
from apix.pipeline.storage import StorageEngine


# ==============================================================================
# MANDATE 1: ZERO-IQR BEHAVIOR STRESS TESTS
# ==============================================================================

class TestZeroIQRStress:
    """Empirical verification of Zero-IQR bounds, minor deltas, and anomaly quarantine."""

    @pytest.fixture
    def base_quote_kwargs(self):
        return {
            "source": "indigo",
            "origin_iata": "DEL",
            "destination_iata": "BOM",
            "carrier_code": "6E",
            "departure_time": time(6, 0),
            "arrival_time": time(8, 15),
            "travel_date": date(2026, 10, 5),
            "advance_days": 7,
            "duration_minutes": 135,
        }

    def test_filter_iqr_bounds_uniform_cohort(self):
        """Uniform cohort of 50 quotes at INR 4,500 expands bounds to +/-15% around median."""
        cohort = [4500.0] * 50
        lb, ub = filter_iqr_bounds(cohort)
        assert lb == 3825.0  # 4500 * 0.85
        assert ub == 5175.0  # 4500 * 1.15

    def test_zero_iqr_minor_price_deltas_not_outliers(self, base_quote_kwargs):
        """Minor price deltas (+1%, -2%, +/-5%, +/-10%, +/-14.9%) are NOT falsely flagged as outliers."""
        cohort = [4500.0] * 50
        detector = OutlierDetector()

        test_deltas = [
            (+0.01, 4545.0),   # +1%
            (-0.02, 4410.0),   # -2%
            (+0.005, 4522.5),  # +0.5%
            (-0.01, 4455.0),   # -1%
            (+0.05, 4725.0),   # +5%
            (-0.05, 4275.0),   # -5%
            (+0.10, 4950.0),   # +10%
            (-0.10, 4050.0),   # -10%
            (+0.149, 5170.5),  # +14.9% (inside +15% bound)
            (-0.149, 3829.5),  # -14.9% (inside -15% bound)
        ]

        for pct, fare_val in test_deltas:
            # 1. Outlier detector check
            eval_res = detector.evaluate_fare(fare_val, cohort_fares=cohort, cohort_median=4500.0)
            assert not eval_res.is_outlier, f"Fare {fare_val} ({pct*100:+.1f}%) falsely flagged as outlier!"
            assert len(eval_res.flags) == 0

            # 2. Pipeline clean_quote check
            q = RawFareQuote(
                flight_number=f"6E-{int(fare_val)}",
                raw_total_fare=fare_val,
                **base_quote_kwargs,
            )
            cleaned = clean_quote(q, cohort_fares=cohort, cohort_median=4500.0)
            assert not cleaned.is_outlier, f"Cleaned quote {fare_val} has is_outlier=True!"
            assert not cleaned.is_quarantined, f"Cleaned quote {fare_val} falsely quarantined!"
            assert cleaned.data_quality_score >= 80.0, f"DQS {cleaned.data_quality_score} below threshold!"

    def test_zero_iqr_true_anomalies_quarantined_and_flagged(self, base_quote_kwargs):
        """
        True anomalies (+400%, INR 200, INR 90,000) are flagged as outliers and
        quarantined / excluded from the clean econometric dataset.
        """
        cohort = [4500.0] * 50
        detector = OutlierDetector()

        # 1. Sub-floor anomaly: INR 200
        res_200 = detector.evaluate_fare(200.0, cohort_fares=cohort, cohort_median=4500.0)
        assert res_200.is_outlier is True
        assert "DOMAIN_FLOOR" in res_200.flags
        assert "STATISTICAL_IQR" in res_200.flags

        q_200 = RawFareQuote(flight_number="6E-200", raw_total_fare=200.0, **base_quote_kwargs)
        c_200 = clean_quote(q_200, cohort_fares=cohort, cohort_median=4500.0)
        assert c_200.is_outlier is True
        assert c_200.is_quarantined is True
        assert c_200.outlier_reason == OutlierReason.DOMAIN_FLOOR.value

        # 2. Super-ceiling anomaly: INR 90,000
        res_90k = detector.evaluate_fare(90000.0, cohort_fares=cohort, cohort_median=4500.0)
        assert res_90k.is_outlier is True
        assert "DOMAIN_CEILING" in res_90k.flags
        assert "STATISTICAL_IQR" in res_90k.flags

        q_90k = RawFareQuote(flight_number="6E-90K", raw_total_fare=90000.0, **base_quote_kwargs)
        c_90k = clean_quote(q_90k, cohort_fares=cohort, cohort_median=4500.0)
        assert c_90k.is_outlier is True
        assert c_90k.is_quarantined is True
        assert c_90k.outlier_reason == OutlierReason.DOMAIN_CEILING.value

        # 3. Extreme statistical surge anomaly: +400% on INR 4,500 (INR 22,500)
        res_400 = detector.evaluate_fare(22500.0, cohort_fares=cohort, cohort_median=4500.0)
        assert res_400.is_outlier is True
        assert "STATISTICAL_IQR" in res_400.flags

        q_400 = RawFareQuote(flight_number="6E-400P", raw_total_fare=22500.0, **base_quote_kwargs)
        c_400 = clean_quote(q_400, cohort_fares=cohort, cohort_median=4500.0)
        assert c_400.is_outlier is True
        assert c_400.outlier_reason == OutlierReason.STATISTICAL_IQR.value
        # DQS drops below index inclusion threshold (80.0)
        assert c_400.data_quality_score < 80.0

        # 4. Ceiling violation surge: +400% on INR 5,500 (INR 27,500 > INR 25,000)
        cohort_5500 = [5500.0] * 50
        q_ceil_surge = RawFareQuote(flight_number="6E-CEIL", raw_total_fare=27500.0, **base_quote_kwargs)
        c_ceil_surge = clean_quote(q_ceil_surge, cohort_fares=cohort_5500, cohort_median=5500.0)
        assert c_ceil_surge.is_outlier is True
        assert c_ceil_surge.is_quarantined is True
        assert c_ceil_surge.outlier_reason == OutlierReason.DOMAIN_CEILING.value

    def test_zero_iqr_storage_clean_quotes_filtering(self, base_quote_kwargs):
        """End-to-end SQLite test: minor deltas are retrieved, while all true anomalies are excluded."""
        eng = StorageEngine("sqlite:///:memory:")

        quotes: List[RawFareQuote] = []
        # 50 uniform quotes at INR 4,500
        for i in range(50):
            quotes.append(RawFareQuote(
                flight_number=f"6E-{1000+i}",
                raw_total_fare=4500.0,
                **base_quote_kwargs,
            ))

        # Minor deltas
        quotes.append(RawFareQuote(flight_number="6E-DELTA-P1", raw_total_fare=4545.0, **base_quote_kwargs))
        quotes.append(RawFareQuote(flight_number="6E-DELTA-M2", raw_total_fare=4410.0, **base_quote_kwargs))

        # True anomalies
        quotes.append(RawFareQuote(flight_number="6E-ANOM-400P", raw_total_fare=22500.0, **base_quote_kwargs))
        quotes.append(RawFareQuote(flight_number="6E-ANOM-200", raw_total_fare=200.0, **base_quote_kwargs))
        quotes.append(RawFareQuote(flight_number="6E-ANOM-90K", raw_total_fare=90000.0, **base_quote_kwargs))

        # Batch clean
        cleaned_batch = clean_quotes_batch(quotes, deduplicate=False)
        eng.insert_quotes(cleaned_batch)

        # Retrieve clean quotes for econometric index
        clean_retrieved = eng.get_clean_quotes(sector="DEL-BOM", advance_window=7)
        retrieved_fns = {q.flight_number for q in clean_retrieved}

        # Verify minor deltas are included
        assert "6E-DELTA-P1" in retrieved_fns
        assert "6E-DELTA-M2" in retrieved_fns
        assert len(clean_retrieved) == 52  # 50 base + 2 minor deltas

        # Verify all anomalies are strictly excluded
        assert "6E-ANOM-400P" not in retrieved_fns
        assert "6E-ANOM-200" not in retrieved_fns
        assert "6E-ANOM-90K" not in retrieved_fns

    def test_zero_iqr_clamping_at_statutory_boundaries(self):
        """Zero-IQR expansion correctly respects the INR 999 floor and INR 25,000 ceiling."""
        # Low median: INR 1,000 -> 1000 * 0.85 = 850, clamped to 999.0
        low_cohort = [1000.0] * 30
        lb_low, ub_low = filter_iqr_bounds(low_cohort)
        assert lb_low == 999.0
        assert ub_low == 1150.0

        # High median: INR 24,000 -> 24000 * 1.15 = 27600, clamped to 25000.0
        high_cohort = [24000.0] * 30
        lb_high, ub_high = filter_iqr_bounds(high_cohort)
        assert lb_high == 20400.0
        assert ub_high == 25000.0

    def test_zero_iqr_small_sample_size_fallback(self):
        """Cohorts with sample size < min_cohort_sample_size (5) do not flag statistical outliers."""
        detector = OutlierDetector(min_cohort_sample_size=5)
        small_cohort = [4500.0] * 4  # Only 4 samples
        res = detector.evaluate_fare(4545.0, cohort_fares=small_cohort)
        assert not res.is_outlier
        assert "STATISTICAL_IQR" not in res.flags


# ==============================================================================
# MANDATE 2: FLIGHT NUMBER NORMALIZATION & DEDUPLICATION STRESS TESTS
# ==============================================================================

class TestFlightNumberNormalizationDeduplication:
    """Empirical verification of canonical flight key normalization and quote deduplication."""

    @pytest.fixture
    def base_flight_info(self):
        return {
            "origin_iata": "DEL",
            "destination_iata": "BOM",
            "carrier_code": "AI",
            "departure_time": time(7, 0),
            "arrival_time": time(9, 15),
            "travel_date": date(2026, 10, 1),
            "advance_days": 7,
            "duration_minutes": 135,
        }

    def test_messy_flight_number_normalization_canonical_key(self, base_flight_info):
        """Messy flight numbers ('ai-101', 'AI 101', 'AI101', 'AI - 101') all resolve to 'AI101'."""
        messy_variants = [
            "ai-101",
            "AI 101",
            "AI101",
            "AI - 101",
            "  ai - 101  ",
            "AI--101",
            "a_i_1_0_1",
            "Ai.101",
        ]

        expected_key = ("DEL", "BOM", "AI", "AI101", date(2026, 10, 1))
        for fn in messy_variants:
            q = RawFareQuote(source="airindia", flight_number=fn, raw_total_fare=5200.0, **base_flight_info)
            key = canonical_flight_key(q)
            assert key == expected_key, f"Flight number '{fn}' mapped to {key} instead of {expected_key}!"

    def test_deduplicate_carrier_priority_across_messy_numbers(self, base_flight_info):
        """Direct airline quote is prioritized over OTAs quoting lower fares across messy flight numbers."""
        q_airindia = RawFareQuote(
            source="airindia", flight_number="ai-101", raw_total_fare=5500.0, **base_flight_info
        )
        q_mmt = RawFareQuote(
            source="makemytrip", flight_number="AI 101", raw_total_fare=5100.0, **base_flight_info
        )
        q_ct = RawFareQuote(
            source="cleartrip", flight_number="AI101", raw_total_fare=5250.0, **base_flight_info
        )
        q_ixigo = RawFareQuote(
            source="ixigo", flight_number="AI - 101", raw_total_fare=4950.0, **base_flight_info
        )

        quotes = [q_mmt, q_ct, q_airindia, q_ixigo]
        deduped = deduplicate_quotes(quotes)

        assert len(deduped) == 1
        winner = deduped[0]
        assert winner.source == "airindia"
        assert float(winner.raw_total_fare) == 5500.0

    def test_deduplicate_ota_only_lowest_fare_priority(self, base_flight_info):
        """When only OTAs are present with messy flight numbers, the lowest fare is selected."""
        q1 = RawFareQuote(source="makemytrip", flight_number="ai-101", raw_total_fare=5300.0, **base_flight_info)
        q2 = RawFareQuote(source="cleartrip", flight_number="AI 101", raw_total_fare=5100.0, **base_flight_info)
        q3 = RawFareQuote(source="easemytrip", flight_number="AI101", raw_total_fare=4950.0, **base_flight_info)
        q4 = RawFareQuote(source="ixigo", flight_number="AI - 101", raw_total_fare=5200.0, **base_flight_info)

        deduped = deduplicate_quotes([q1, q2, q3, q4])
        assert len(deduped) == 1
        assert deduped[0].source == "easemytrip"
        assert float(deduped[0].raw_total_fare) == 4950.0

    def test_deduplication_order_invariance(self, base_flight_info):
        """Deduplication result is invariant to all permutations of input quotes."""
        q_carrier = RawFareQuote(source="airindia", flight_number="ai-101", raw_total_fare=5500.0, **base_flight_info)
        q_mmt = RawFareQuote(source="makemytrip", flight_number="AI 101", raw_total_fare=5100.0, **base_flight_info)
        q_ct = RawFareQuote(source="cleartrip", flight_number="AI101", raw_total_fare=5250.0, **base_flight_info)
        q_emt = RawFareQuote(source="easemytrip", flight_number="AI - 101", raw_total_fare=4950.0, **base_flight_info)

        pool = [q_carrier, q_mmt, q_ct, q_emt]
        for perm in itertools.permutations(pool):
            res = deduplicate_quotes(list(perm))
            assert len(res) == 1
            assert res[0].source == "airindia"

    def test_deduplication_orthogonal_isolation(self, base_flight_info):
        """Deduplication does not cross-contaminate different flight numbers, dates, or sectors."""
        quotes = [
            # Flight 1: DEL-BOM on 2026-10-01 (2 duplicates)
            RawFareQuote(source="airindia", flight_number="AI-101", raw_total_fare=5200.0, **base_flight_info),
            RawFareQuote(source="makemytrip", flight_number="ai 101", raw_total_fare=5000.0, **base_flight_info),

            # Flight 2: DEL-BOM on 2026-10-02 (different date)
            RawFareQuote(
                source="airindia", flight_number="AI-101", raw_total_fare=5300.0,
                **{**base_flight_info, "travel_date": date(2026, 10, 2)}
            ),

            # Flight 3: BOM-DEL on 2026-10-01 (different sector)
            RawFareQuote(
                source="airindia", flight_number="AI-101", raw_total_fare=5400.0,
                **{**base_flight_info, "origin_iata": "BOM", "destination_iata": "DEL"}
            ),

            # Flight 4: AI-102 (different flight number)
            RawFareQuote(source="airindia", flight_number="AI-102", raw_total_fare=5600.0, **base_flight_info),
        ]

        deduped = deduplicate_quotes(quotes)
        assert len(deduped) == 4, f"Expected 4 distinct flights, got {len(deduped)}!"


# ==============================================================================
# MANDATE 3: BOUNDARY VALUES & EDGE CASE STRESS TESTS
# ==============================================================================

class TestBoundaryValuesAndEdgeCases:
    """Empirical verification of zero/negative fares, exact domain limits, and extreme inputs."""

    @pytest.fixture
    def valid_params(self):
        return {
            "source": "indigo",
            "origin_iata": "DEL",
            "destination_iata": "BOM",
            "carrier_code": "6E",
            "flight_number": "6E-501",
            "departure_time": time(8, 0),
            "arrival_time": time(10, 15),
            "travel_date": date(2026, 10, 10),
            "advance_days": 12,
            "duration_minutes": 135,
        }

    def test_zero_fare_quote_rejection_and_quarantine(self, valid_params):
        """Zero fare (INR 0.00) preserves decomposition balance and is strictly quarantined."""
        q_zero = RawFareQuote(raw_total_fare=0.0, **valid_params)
        c_zero = clean_quote(q_zero)

        assert c_zero.is_outlier is True
        assert c_zero.outlier_reason == OutlierReason.DOMAIN_FLOOR.value
        assert c_zero.is_quarantined is True
        # Verify decomposition arithmetic balance holds
        assert c_zero.decomposed_fare.is_balanced(tolerance=1.0)

    def test_negative_fare_quote_rejection_and_quarantine(self, valid_params):
        """Negative fare (INR -500.00) preserves decomposition balance and is strictly quarantined."""
        q_neg = RawFareQuote(raw_total_fare=-500.0, **valid_params)
        c_neg = clean_quote(q_neg)

        assert c_neg.is_outlier is True
        assert c_neg.outlier_reason == OutlierReason.DOMAIN_FLOOR.value
        assert c_neg.is_quarantined is True
        assert c_neg.decomposed_fare.is_balanced(tolerance=1.0)

    def test_exact_domain_floor_boundary(self, valid_params):
        """Exact statutory floor boundary test: INR 998.99 (fails), INR 999.00 (passes), INR 999.01 (passes)."""
        # INR 998.99 -> sub-floor violation
        q_sub = RawFareQuote(raw_total_fare=998.99, **valid_params)
        c_sub = clean_quote(q_sub)
        assert c_sub.is_outlier is True
        assert c_sub.is_quarantined is True
        assert c_sub.outlier_reason == OutlierReason.DOMAIN_FLOOR.value

        # INR 999.00 -> exact floor boundary
        q_exact = RawFareQuote(raw_total_fare=999.00, **valid_params)
        c_exact = clean_quote(q_exact)
        assert c_exact.is_outlier is False
        assert c_exact.is_quarantined is False
        assert c_exact.data_quality_score >= 80.0

        # INR 999.01 -> strictly valid
        q_plus = RawFareQuote(raw_total_fare=999.01, **valid_params)
        c_plus = clean_quote(q_plus)
        assert c_plus.is_outlier is False
        assert c_plus.is_quarantined is False
        assert c_plus.data_quality_score >= 80.0

    def test_exact_domain_ceiling_boundary(self, valid_params):
        """Exact statutory ceiling boundary test: INR 24,999.99 (passes), INR 25,000.00 (passes), INR 25,000.01 (fails)."""
        # INR 24,999.99 -> strictly valid
        q_sub = RawFareQuote(raw_total_fare=24999.99, **valid_params)
        c_sub = clean_quote(q_sub)
        assert c_sub.is_outlier is False
        assert c_sub.is_quarantined is False
        assert c_sub.data_quality_score >= 80.0

        # INR 25,000.00 -> exact ceiling boundary
        q_exact = RawFareQuote(raw_total_fare=25000.00, **valid_params)
        c_exact = clean_quote(q_exact)
        assert c_exact.is_outlier is False
        assert c_exact.is_quarantined is False
        assert c_exact.data_quality_score >= 80.0

        # INR 25,000.01 -> ceiling violation
        q_over = RawFareQuote(raw_total_fare=25000.01, **valid_params)
        c_over = clean_quote(q_over)
        assert c_over.is_outlier is True
        assert c_over.is_quarantined is True
        assert c_over.outlier_reason == OutlierReason.DOMAIN_CEILING.value

    def test_extreme_dates_handling(self, valid_params):
        """System correctly handles far-future travel dates and leap years."""
        # Leap year: 2028-02-29
        q_leap = RawFareQuote(
            raw_total_fare=5000.0,
            **{**valid_params, "travel_date": date(2028, 2, 29), "advance_days": 519}
        )
        c_leap = clean_quote(q_leap)
        assert c_leap.travel_date == date(2028, 2, 29)
        assert c_leap.data_quality_score >= 80.0

        # Far future: 2099-12-31
        q_future = RawFareQuote(
            raw_total_fare=5000.0,
            **{**valid_params, "travel_date": date(2099, 12, 31), "advance_days": 26750}
        )
        c_future = clean_quote(q_future)
        assert c_future.travel_date == date(2099, 12, 31)
        assert c_future.data_quality_score >= 80.0

        # Negative advance days rejected by schema validator
        with pytest.raises(ValidationError):
            RawFareQuote(
                raw_total_fare=5000.0,
                **{**valid_params, "advance_days": -1}
            )

    def test_missing_optional_fields_integrity(self, valid_params):
        """Quotes with missing optional fields maintain high DQS and valid decomposition."""
        q_opt = RawFareQuote(
            raw_total_fare=4800.0,
            raw_base_fare=None,
            raw_fuel_surcharge=None,
            raw_taxes=None,
            raw_udf=None,
            raw_psf=None,
            raw_convenience_fee=None,
            seats_available=None,
            **valid_params,
        )
        c_opt = clean_quote(q_opt)
        assert c_opt.seats_available is None
        assert c_opt.data_quality_score >= 80.0
        assert c_opt.decomposed_fare.is_balanced(tolerance=1.0)

    def test_sold_out_flight_capacity_exhaustion(self, valid_params):
        """Sold out flights (seats_available=0, is_sold_out=True) preserve price signal."""
        q_sold = RawFareQuote(
            raw_total_fare=16500.0,
            seats_available=0,
            is_sold_out=True,
            **valid_params,
        )
        c_sold = clean_quote(q_sold)
        assert c_sold.is_sold_out is True
        assert c_sold.seats_available == 0
        assert float(c_sold.total_fare) == 16500.0
        assert c_sold.data_quality_score >= 80.0

    def test_empty_batch_robustness(self):
        """Pipeline and storage gracefully handle empty batches without errors."""
        eng = StorageEngine("sqlite:///:memory:")
        assert clean_quotes_batch([]) == []
        assert eng.insert_quotes([]) == 0
        assert filter_iqr_bounds([]) == (999.0, 25000.0)
        assert deduplicate_quotes([]) == []
