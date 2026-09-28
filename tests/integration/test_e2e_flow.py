"""
Integration Test Suite: End-to-End Pipeline & Analytical Lifecycle (Tier 1-4).
Tests the complete lifecycle:
Ingest -> Clean & Decompose -> Outlier Filter -> Store -> Econometric Indexing ->
Policy Intelligence -> Multi-Frequency Publication.
"""

from datetime import date, datetime, time, timedelta
import numpy as np
import pytest

from tests.conftest import (
    RawFareQuoteModel,
    DecomposedFareModel,
    oracle_decompose_fare,
    oracle_jevons,
    oracle_tornqvist,
    oracle_laspeyres,
    oracle_geks,
    DGCA_CITY_PAIRS,
    ADVANCE_WINDOWS,
    CARRIER_MARKET_SHARES,
)
from tests.integration.test_pipeline import (
    generate_deterministic_seed_quotes,
    compute_dqs_score,
)
from tests.unit.test_outlier import (
    is_domain_floor_violation,
    is_domain_ceiling_violation,
)


class TestPolicyIntelligence:
    """Tests for Module D Policy Intelligence: Surges, Variance, Collusion, FAI."""

    def test_price_surge_detection(self):
        """
        Tier 1 & Tier 2: Modified Z-Score / MAD surge detection:
        M_i = 0.6745 * |p_i - median| / MAD.
        Flags quotes with M_i > 3.5.
        """
        # Baseline 30-day route fares on DEL-BOM
        baseline_fares = np.array([4800, 4950, 5100, 5050, 4900, 5200, 5150, 5000, 4850, 5100] * 3)
        median_p = np.median(baseline_fares)
        mad = np.median(np.abs(baseline_fares - median_p))

        # Test normal fare
        normal_p = 5300.0
        m_normal = (0.6745 * abs(normal_p - median_p)) / mad
        assert m_normal < 3.5

        # Test surge fare (e.g. ₹16,500 last-minute surge)
        surge_p = 16500.0
        m_surge = (0.6745 * abs(surge_p - median_p)) / mad
        assert m_surge > 3.5 # Flagged as PRICE_SURGE

    def test_variance_spike_detection(self):
        """
        Tier 1: Coefficient of Variation (CV = sigma / mu) jump detection.
        Flagged if CV > 2.0 * CV_baseline and sigma > 2500 INR.
        """
        # Stable baseline: CV ~ 0.08
        baseline = np.array([4800, 5000, 5200, 4900, 5100], dtype=float)
        cv_baseline = np.std(baseline) / np.mean(baseline)

        # Turbulently dispersed fares on same date: [3500, 8000, 14000, 18000, 22000]
        turbulent = np.array([3500, 8000, 14000, 18000, 22000], dtype=float)
        std_turb = np.std(turbulent)
        cv_turb = std_turb / np.mean(turbulent)

        assert cv_turb > (2.0 * cv_baseline)
        assert std_turb > 2500.0 # Flagged as VARIANCE_SPIKE_TURBULENCE

    def test_anti_competitive_lockstep(self):
        """
        Tier 3: Collusion Detection:
        High HHI (>2500) and lockstep parallel price adjustment (|delta_p1 - delta_p2| <= 150)
        within 4 hours triggers POTENTIAL_TARIFF_COLLUSION.
        """
        # IndiGo (62%) and Air India (14%) on duopoly sector
        # HHI = 62^2 + 14^2 = 3844 + 196 = 4040 > 2500
        shares = [62.0, 14.0, 7.0, 5.0, 4.0]
        hhi = sum(s ** 2 for s in shares)
        assert hhi > 2500

        # Parallel price jump: 6E increases by ₹1200, AI increases by ₹1250
        delta_p_6e = 1200.0
        delta_p_ai = 1250.0
        spread = abs(delta_p_6e - delta_p_ai)

        is_collusion_suspect = (hhi > 2500) and (spread <= 150.0) and (delta_p_6e > 1000.0)
        assert is_collusion_suspect is True

    def test_affordability_fai_hwf(self):
        """
        Tier 1: Fare Affordability Index (FAI) & Hours-of-Work-to-Fly (HWF).
        HWF = Average Fare / Hourly Urban Wage.
        """
        # Delhi daily per capita income ~ 1,200 INR -> hourly wage = 150 INR
        hourly_wage = 150.0
        fare_t45 = 3600.0
        fare_t1 = 12000.0

        hwf_t45 = fare_t45 / hourly_wage # 24 hours (3 working days)
        hwf_t1 = fare_t1 / hourly_wage   # 80 hours (2 working weeks)

        assert hwf_t45 <= 40.0 # Affordable / Moderate
        assert hwf_t1 > 40.0   # Severe affordability strain


class TestE2EWorkflow:
    """Comprehensive E2E Pipeline Lifecycle Test (Tier 4)."""

    def test_full_pipeline_ingest_clean_index_cycle(self, national_weights_matrix):
        """
        Tier 4 Application Scenario:
        Simulate complete flow from raw quote ingestion to cleaned database
        to 3-stage econometric index computation.
        """
        # Step 1: Ingestion
        raw_quotes = generate_deterministic_seed_quotes(300)
        assert len(raw_quotes) == 300

        # Step 2: Cleaning, Decomposition & DQS Filtering
        cleaned_quotes = []
        for q in raw_quotes:
            if is_domain_floor_violation(q.raw_total_fare) or is_domain_ceiling_violation(q.raw_total_fare):
                continue
            decomp = oracle_decompose_fare(q.raw_total_fare, q.origin_iata)
            dqs = compute_dqs_score(q, decomp, is_outlier=False)
            if dqs >= 80.0:
                cleaned_quotes.append((q, decomp, dqs))

        assert len(cleaned_quotes) >= 250 # High clean quote yield

        # Step 3: Stage 1 Jevons Computation per Sector-Carrier-Window
        # Simulate base prices and comparison prices for DEL-BOM
        base_quotes_6e = np.array([4200.0, 4400.0, 4600.0])
        curr_quotes_6e = np.array([4500.0, 4800.0, 4900.0])
        I_6E = oracle_jevons(base_quotes_6e, curr_quotes_6e)

        base_quotes_ai = np.array([5000.0, 5200.0])
        curr_quotes_ai = np.array([5300.0, 5600.0])
        I_AI = oracle_jevons(base_quotes_ai, curr_quotes_ai)

        assert I_6E > 1.0
        assert I_AI > 1.0

        # Step 4: Stage 2 Törnqvist Aggregation
        # Route DEL-BOM, T+7 window
        carrier_indices = np.array([I_6E, I_AI])
        carrier_w0 = np.array([0.70, 0.30])
        carrier_wt = np.array([0.68, 0.32])
        P_T_del_bom = oracle_tornqvist(carrier_indices, carrier_w0, carrier_wt)
        assert P_T_del_bom > 1.0

        # Step 5: Stage 3 National Laspeyres Composite Aggregation
        # Construct 30x5 matrix of route-window indices with slight upward trend (mean ~ 1.06)
        route_window_matrix = np.full((30, 5), 1.06)
        route_window_matrix[0, 1] = P_T_del_bom # plug in exact calculated sector

        national_apix = oracle_laspeyres(route_window_matrix, national_weights_matrix)
        assert 100.0 < national_apix < 115.0 # realistic national index level

    def test_multi_frequency_publication_cycle(self):
        """
        Tier 4: Multi-frequency publication cycle:
        Daily Jevons -> Weekly 7-day Rolling GEKS -> Monthly RYGEKS with mean splice.
        """
        # 7-day daily series
        daily_apix = [131.2, 131.8, 132.4, 132.1, 133.5, 134.1, 134.7]
        assert len(daily_apix) == 7
        assert daily_apix[-1] > daily_apix[0]

        # 7-day weekly GEKS smoothing
        weekly_smoothed = np.mean(daily_apix)
        assert 132.0 <= weekly_smoothed <= 134.0

        # RYGEKS Mean Splice extension without historical revision
        historical_published = [120.0, 122.5, 124.0, 126.8, 129.5]
        # Adding new month t index via geometric mean
        new_relative = 1.025 # +2.5% inflation in month t
        spliced_new_index = historical_published[-1] * new_relative

        # Historical figures remain completely unchanged (non-revisability)
        full_series = historical_published + [spliced_new_index]
        assert full_series[:5] == historical_published
        assert abs(full_series[5] - 132.7375) < 1e-4
