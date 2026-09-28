"""
Comprehensive Robustness Test Suite for Ingestion & Playback Engine.

Tests:
1. Deterministic Seed Playback Engine (identic outputs on identical seeds, valid bounded variances)
2. Lead-Time Elasticity Monotonicity (T+1 > T+7 > T+15 > T+30 > T+45)
3. Carrier Yield Multiplier Rankings (AI > 6E > QP >= IX >= SG)
4. OTA Convenience Fee Policy Invariants (MMT=350, CT=300, IXI=250, EMT=0)
5. Edge Cases: Missing / Invalid Airport Codes, Identical Origin-Destination
6. Edge Cases: Extreme Fares, NaN, Infinity, Negative Values
7. Sold-Out Flights & Seat Availability Tracking
8. Concurrency & Race Condition in RateLimiter
9. Unhandled KeyError in SeedPlaybackEngine for Unrecognized Carriers
10. APScheduler Lifecycle & Exception Isolation
11. Reference Data Normalization Axioms & UDF Coverage
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal
import json
import os
import threading
import time as time_module
import pytest
from pydantic import ValidationError

from apix.ingestion.base import (
    RawFareQuote,
    RateLimiter,
    RobotsValidator,
    APPROVED_SECTORS,
    APPROVED_CARRIERS,
    APPROVED_SOURCES,
    DEFAULT_USER_AGENT
)
from apix.ingestion.airlines import (
    AIRLINE_SCRAPERS,
    get_airline_scraper,
    IndiGoScraper,
    AirIndiaScraper
)
from apix.ingestion.otas import (
    OTA_SCRAPERS,
    get_ota_scraper,
    MakeMyTripScraper,
    EaseMyTripScraper
)
from apix.ingestion.seed_engine import (
    SeedPlaybackEngine,
    AIRPORT_UDF_MAP,
    STATUTORY_PSF,
    GST_RATE_ECONOMY
)
from apix.ingestion.scheduler import (
    APIxIngestionScheduler,
    TOP_TRUNK_SECTORS
)


class TestSeedPlaybackDeterminism:
    """Stress tests for deterministic playback engine reproducibility and variance."""

    def test_identical_seed_exact_identity(self):
        """Identical seeds must produce 100% byte-for-byte identical RawFareQuote models."""
        engine1 = SeedPlaybackEngine(seed=42)
        engine2 = SeedPlaybackEngine(seed=42)
        travel_dt = date(2026, 10, 15)

        for sector in sorted(list(APPROVED_SECTORS))[:5]:
            orig, dest = sector.split("-")
            for adv in [1, 7, 30]:
                q1 = engine1.synthesize_quote(orig, dest, travel_dt, adv, "6E", "indigo")
                q2 = engine2.synthesize_quote(orig, dest, travel_dt, adv, "6E", "indigo")
                assert q1.quote_id == q2.quote_id
                assert q1.raw_total_fare == q2.raw_total_fare
                assert q1.raw_base_fare == q2.raw_base_fare
                assert q1.is_sold_out == q2.is_sold_out
                assert q1.model_dump() == q2.model_dump()

    def test_different_seeds_valid_bounded_variance(self):
        """Different seeds must produce realistic variances bounded within [-4%, +4%] noise."""
        travel_dt = date(2026, 10, 15)
        base_fares = []
        total_fares = []

        for seed in range(40):
            engine = SeedPlaybackEngine(seed=seed)
            q = engine.synthesize_quote("DEL", "BOM", travel_dt, 7, "6E", "indigo")
            base_fares.append(float(q.raw_base_fare))
            total_fares.append(float(q.raw_total_fare))

        # Check that variance is strictly positive
        assert len(set(total_fares)) > 1, "Expected variance across different seeds, got constant"

        # Check bounds: max deviation should be around ~8% total spread
        mean_base = sum(base_fares) / len(base_fares)
        min_dev = (min(base_fares) - mean_base) / mean_base
        max_dev = (max(base_fares) - mean_base) / mean_base
        assert min_dev >= -0.06, f"Min base fare deviation {min_dev:.4f} exceeds allowable noise"
        assert max_dev <= +0.06, f"Max base fare deviation {max_dev:.4f} exceeds allowable noise"

    def test_lead_time_elasticity_monotonicity(self):
        """Lead-time yield curve must be strictly monotonic: T+1 > T+7 > T+15 > T+30 > T+45."""
        engine = SeedPlaybackEngine(seed=42)
        travel_dt = date(2026, 10, 15)
        adv_windows = [1, 7, 15, 30, 45]
        quotes = {
            adv: engine.synthesize_quote("DEL", "BLR", travel_dt, adv, "6E", "indigo")
            for adv in adv_windows
        }

        for i in range(len(adv_windows) - 1):
            w_curr = adv_windows[i]
            w_next = adv_windows[i + 1]
            assert quotes[w_curr].raw_base_fare > quotes[w_next].raw_base_fare, (
                f"Monotonicity violated: T+{w_curr} ({quotes[w_curr].raw_base_fare}) "
                f"<= T+{w_next} ({quotes[w_next].raw_base_fare})"
            )

    def test_carrier_yield_hierarchy(self):
        """Carrier pricing multipliers must strictly preserve: AI (FSC 1.18) > 6E (1.00) > QP/IX/SG (0.94-0.96)."""
        engine = SeedPlaybackEngine(seed=42)
        travel_dt = date(2026, 10, 15)
        fares = {}
        for c in ["AI", "6E", "QP", "IX", "SG"]:
            src = {"6E": "indigo", "AI": "airindia", "IX": "airindiaexpress", "QP": "akasa", "SG": "spicejet"}[c]
            q = engine.synthesize_quote("DEL", "BOM", travel_dt, 15, c, src)
            fares[c] = q.raw_base_fare

        assert fares["AI"] > fares["6E"], "Full service carrier Air India should have higher base fare than IndiGo"
        assert fares["6E"] > fares["QP"], "IndiGo baseline should exceed Akasa"
        assert fares["6E"] > fares["IX"], "IndiGo baseline should exceed AI Express"
        assert fares["6E"] > fares["SG"], "IndiGo baseline should exceed SpiceJet"

    def test_ota_convenience_fees(self):
        """OTAs must apply correct platform fees; EaseMyTrip and direct airlines must have 0 fee."""
        engine = SeedPlaybackEngine(seed=42)
        travel_dt = date(2026, 10, 15)
        assert engine.synthesize_quote("DEL", "BOM", travel_dt, 7, "6E", "makemytrip").raw_convenience_fee == Decimal("350.00")
        assert engine.synthesize_quote("DEL", "BOM", travel_dt, 7, "6E", "cleartrip").raw_convenience_fee == Decimal("300.00")
        assert engine.synthesize_quote("DEL", "BOM", travel_dt, 7, "6E", "ixigo").raw_convenience_fee == Decimal("250.00")
        assert engine.synthesize_quote("DEL", "BOM", travel_dt, 7, "6E", "easemytrip").raw_convenience_fee == Decimal("0.00")
        assert engine.synthesize_quote("DEL", "BOM", travel_dt, 7, "6E", "indigo").raw_convenience_fee == Decimal("0.00")


class TestSchemaAndEdgeCases:
    """Stress tests on schema boundaries, missing data, and invalid inputs."""

    def test_missing_or_invalid_airport_codes_rejected(self):
        """Pydantic schema must reject empty, non-3-letter, numeric, or identical IATA codes."""
        invalid_pairs = [
            ("", "BOM"),
            ("DEL", ""),
            (None, "BOM"),
            ("DEL", None),
            ("DE", "BOM"),
            ("DEL", "BO"),
            ("DELHI", "BOM"),
            ("123", "BOM"),
            ("DEL", "456"),
            ("DEL", "DEL"),  # identical origin and destination
            ("BOM", "BOM")
        ]
        for orig, dest in invalid_pairs:
            with pytest.raises((ValidationError, ValueError)):
                RawFareQuote(
                    source="indigo",
                    origin_iata=orig,
                    destination_iata=dest,
                    carrier_code="6E",
                    flight_number="6E-101",
                    departure_time=time(8, 0),
                    arrival_time=time(10, 0),
                    travel_date=date(2026, 10, 1),
                    advance_days=7,
                    duration_minutes=120,
                    raw_total_fare=Decimal("5000.00")
                )

    def test_extreme_fare_values_validation(self):
        """Verify handling of NaN, Infinity, negative values, and domain boundaries."""
        # NaN and Infinity must be rejected
        with pytest.raises(ValidationError):
            RawFareQuote(
                source="indigo",
                origin_iata="DEL",
                destination_iata="BOM",
                carrier_code="6E",
                flight_number="6E-101",
                departure_time=time(8, 0),
                arrival_time=time(10, 0),
                travel_date=date(2026, 10, 1),
                advance_days=7,
                duration_minutes=120,
                raw_total_fare=Decimal("NaN")
            )

        with pytest.raises(ValidationError):
            RawFareQuote(
                source="indigo",
                origin_iata="DEL",
                destination_iata="BOM",
                carrier_code="6E",
                flight_number="6E-101",
                departure_time=time(8, 0),
                arrival_time=time(10, 0),
                travel_date=date(2026, 10, 1),
                advance_days=7,
                duration_minutes=120,
                raw_total_fare=Decimal("Infinity")
            )

        # Negative seats must be rejected
        with pytest.raises(ValidationError):
            RawFareQuote(
                source="indigo",
                origin_iata="DEL",
                destination_iata="BOM",
                carrier_code="6E",
                flight_number="6E-101",
                departure_time=time(8, 0),
                arrival_time=time(10, 0),
                travel_date=date(2026, 10, 1),
                advance_days=7,
                duration_minutes=120,
                raw_total_fare=Decimal("5000.00"),
                seats_available=-1
            )

    def test_sold_out_flights_quote_integrity(self):
        """Sold-out quotes must have is_sold_out=True and seats_available=0 with valid positive price."""
        engine = SeedPlaybackEngine(seed=42)
        quotes = engine.load_seed_dataset()
        sold_out_quotes = [q for q in quotes if q.is_sold_out]
        assert len(sold_out_quotes) > 0, "Seed dataset should include sold-out quotes"

        for q in sold_out_quotes:
            assert q.is_sold_out is True
            assert q.seats_available == 0
            assert q.raw_total_fare > Decimal("999.00")
            assert q.raw_total_fare <= Decimal("25000.00")


class TestAdversarialFindings:
    """Tests confirming specific failure modes and vulnerabilities found during adversarial review."""

    def test_rate_limiter_concurrency_race_condition(self):
        """
        Adversarial Test: Verify RateLimiter serializes concurrent requests with zero bursts
        using atomic future-slot reservation.
        """
        min_delay = 0.05
        rl = RateLimiter(min_delay_seconds=min_delay)
        dispatches = []
        lock = threading.Lock()

        def worker():
            rl.throttle("example.com")
            with lock:
                dispatches.append(time_module.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        sorted_d = sorted(dispatches)
        deltas = [sorted_d[i+1] - sorted_d[i] for i in range(len(sorted_d)-1)]
        near_simultaneous = sum(1 for d in deltas if d < min_delay * 0.5)
        assert near_simultaneous == 0, f"Expected 0 concurrent bursts, got {near_simultaneous} (deltas: {deltas})"
        assert all(d >= min_delay * 0.80 for d in deltas), f"Deltas breached min_delay: {deltas}"

    def test_unrecognized_carrier_keyerror_finding(self):
        """
        Adversarial Test: SeedPlaybackEngine.generate_quotes_for_sector() raises unhandled KeyError
        when passed an unapproved or unrecognized carrier code instead of returning an empty list or raising ValueError.
        """
        engine = SeedPlaybackEngine()
        with pytest.raises(KeyError) as exc_info:
            engine.generate_quotes_for_sector("DEL", "BOM", date(2026, 10, 1), 7, carrier_code="XX")
        assert "XX" in str(exc_info.value)


class TestSchedulerAndReferenceAxioms:
    """Stress tests on scheduler lifecycle and reference dataset mathematical properties."""

    def test_scheduler_lifecycle_and_callback_isolation(self):
        """Scheduler handles start/shutdown cleanly and isolates failing callbacks without crash."""
        def bad_callback(quotes, job_id):
            raise ValueError("Crashing callback intentionally")

        sched = APIxIngestionScheduler(on_quotes_collected=bad_callback, auto_start=False)
        assert not sched.is_running
        sched.start()
        assert sched.is_running

        # Trigger job with failing callback
        quotes = sched.run_trunk_volatility_monitor()
        assert len(quotes) == 100

        sched.shutdown(wait=False)
        assert not sched.is_running

    def test_dgca_joint_weight_normalization_axiom(self):
        """National index weights sum strictly: sum(omega_{r,w}) = 1.0000000000 +- 1e-12."""
        with open("c:/ApiX_main/data/dgca_weights.json") as f:
            dgca = json.load(f)
        with open("c:/ApiX_main/data/booking_distribution.json") as f:
            bdist = json.load(f)

        sectors = dgca["sectors"]
        assert len(sectors) == 30
        w_sum = sum(s["weight"] for s in sectors.values())
        assert abs(w_sum - 1.0) < 1e-12

        bw_sum = sum(bdist["distribution"].values())
        assert abs(bw_sum - 1.0) < 1e-12

        joint_sum = sum(s["weight"] * bw for s in sectors.values() for bw in bdist["distribution"].values())
        assert abs(joint_sum - 1.0) < 1e-12
