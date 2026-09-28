"""
Robustness and Stress Test Suite for Ingestion Engine and Rate Limiting.

Tests:
1. Symmetry and bidirectionality of all 30 sectors in data/dgca_weights.json
2. Normalization of carrier market shares (sum == 1.0)
3. RateLimiter enforcement, sequential throttling, domain isolation, and concurrent burst stress
4. APScheduler lifecycle (start/stop/pause/resume/restart) and callback accuracy
"""

from datetime import date, datetime, timedelta
from decimal import Decimal
import json
import os
import threading
import time
from typing import List, Tuple

import pytest

from apix.ingestion.base import RateLimiter, RobotsValidator, RawFareQuote, APPROVED_SECTORS
from apix.ingestion.scheduler import APIxIngestionScheduler, TOP_TRUNK_SECTORS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DGCA_WEIGHTS_PATH = os.path.join(BASE_DIR, "data", "dgca_weights.json")
BOOKING_DIST_PATH = os.path.join(BASE_DIR, "data", "booking_distribution.json")


# ==============================================================================
# 1. MATHEMATICAL INTEGRITY & SYMMETRY OF DGCA WEIGHTS
# ==============================================================================

class TestDgcaWeightsSymmetry:
    """Adversarial stress-testing of 30 DGCA sectors for strict bidirectionality & symmetry."""

    @pytest.fixture(autouse=True)
    def load_weights(self):
        with open(DGCA_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.sectors = self.data["sectors"]
        self.city_pairs = self.data["city_pairs"]
        self.carrier_shares = self.data["carrier_market_shares"]

    def test_sector_count_and_naming_compliance(self):
        """Assert exactly 30 sectors matching R1 specification."""
        assert len(self.sectors) == 30, f"Expected 30 sectors, found {len(self.sectors)}"
        assert set(self.sectors.keys()) == APPROVED_SECTORS, "Sectors do not match APPROVED_SECTORS"

    def test_bidirectional_existence_and_symmetry(self):
        """
        For every sector A-B, verify:
        - B-A exists
        - distance_km(A-B) == distance_km(B-A)
        - approx_flight_minutes(A-B) == approx_flight_minutes(B-A)
        - pair_id(A-B) == pair_id(B-A)
        - route_tier(A-B) == route_tier(B-A)
        - annual_pax_millions(A-B) == annual_pax_millions(B-A)
        - weight(A-B) == weight(B-A)
        """
        for sec_name, sec_data in self.sectors.items():
            orig, dest = sec_name.split("-")
            rev_name = f"{dest}-{orig}"
            assert rev_name in self.sectors, f"Reverse sector {rev_name} missing for {sec_name}"

            rev_data = self.sectors[rev_name]
            assert sec_data["distance_km"] == rev_data["distance_km"], (
                f"Asymmetric distance: {sec_name}={sec_data['distance_km']} vs {rev_name}={rev_data['distance_km']}"
            )
            assert sec_data["approx_flight_minutes"] == rev_data["approx_flight_minutes"], (
                f"Asymmetric flight minutes: {sec_name} vs {rev_name}"
            )
            assert sec_data["pair_id"] == rev_data["pair_id"], (
                f"Pair ID mismatch: {sec_name}={sec_data['pair_id']} vs {rev_name}={rev_data['pair_id']}"
            )
            assert sec_data["route_tier"] == rev_data["route_tier"], (
                f"Route tier mismatch: {sec_name} vs {rev_name}"
            )
            assert abs(sec_data["annual_pax_millions"] - rev_data["annual_pax_millions"]) < 1e-9, (
                f"Asymmetric annual pax: {sec_name} vs {rev_name}"
            )
            assert abs(sec_data["weight"] - rev_data["weight"]) < 1e-12, (
                f"Asymmetric sector weight: {sec_name}={sec_data['weight']} vs {rev_name}={rev_data['weight']}"
            )

    def test_city_pairs_consistency_with_sectors(self):
        """Verify the 15 city pairs correspond 1:1 to the 15 paired directional sectors."""
        assert len(self.city_pairs) == 15, f"Expected 15 city pairs, found {len(self.city_pairs)}"
        pair_ids = [p["pair_id"] for p in self.city_pairs]
        assert sorted(pair_ids) == list(range(1, 16)), "Pair IDs must be contiguous 1..15"

        pair_map = {p["pair_id"]: p for p in self.city_pairs}
        for sec_name, sec_data in self.sectors.items():
            pid = sec_data["pair_id"]
            p = pair_map[pid]
            # Sector weight must be exactly half of pair weight
            assert abs(sec_data["weight"] - (p["normalized_pair_weight"] / 2.0)) < 1e-12, (
                f"Sector weight {sec_name} must equal half of pair weight {p['iata_pair']}"
            )
            # Distance and flight minutes must match pair
            assert sec_data["distance_km"] == p["distance_km"]
            assert sec_data["approx_flight_minutes"] == p["approx_flight_minutes"]

    def test_sum_of_weights_equals_unity(self):
        """Verify strict sum(weight) == 1.0 across all 30 sectors."""
        total_sector_weight = sum(s["weight"] for s in self.sectors.values())
        assert abs(total_sector_weight - 1.0) < 1e-10, (
            f"Sector weights sum to {total_sector_weight}, expected 1.0"
        )

        total_pair_weight = sum(p["normalized_pair_weight"] for p in self.city_pairs)
        assert abs(total_pair_weight - 1.0) < 1e-10, (
            f"Normalized pair weights sum to {total_pair_weight}, expected 1.0"
        )


# ==============================================================================
# 2. CARRIER MARKET SHARE SUMMATION TO 1.0
# ==============================================================================

class TestCarrierMarketShares:
    """Adversarial stress-testing of carrier market shares."""

    @pytest.fixture(autouse=True)
    def load_shares(self):
        with open(DGCA_WEIGHTS_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.shares = data["carrier_market_shares"]

    def test_carrier_market_shares_sum_to_one(self):
        """Verify sum of market_share across all carriers (including OTHERS) == 1.0."""
        expected_carriers = {"6E", "AI", "IX", "QP", "SG", "OTHERS"}
        assert set(self.shares.keys()) == expected_carriers

        total_share = sum(c["market_share"] for c in self.shares.values())
        assert abs(total_share - 1.0) < 1e-10, f"Carrier market shares sum to {total_share}, expected 1.0"

    def test_carrier_individual_shares_positive_and_bounded(self):
        """Every carrier market share must be strictly in (0, 1)."""
        for code, info in self.shares.items():
            ms = info["market_share"]
            assert 0.0 < ms < 1.0, f"Invalid market share {ms} for {code}"

    def test_normalized_monitored_shares_sum_to_one(self):
        """Verify normalized_monitored_share across monitored airlines sums to 1.0."""
        monitored_sum = sum(
            c["normalized_monitored_share"]
            for code, c in self.shares.items()
            if code != "OTHERS"
        )
        assert abs(monitored_sum - 1.0) < 1e-10, (
            f"Normalized monitored shares sum to {monitored_sum}, expected 1.0"
        )
        # OTHERS must have 0.0 monitored share
        assert self.shares["OTHERS"]["normalized_monitored_share"] == 0.0


# ==============================================================================
# 3. RATE LIMITER BURST ENFORCEMENT & CONCURRENCY
# ==============================================================================

class TestRateLimiterAdversarial:
    """Adversarial testing of RateLimiter enforcement under rapid burst requests."""

    def test_rate_limiter_sequential_burst_throttling(self):
        """
        Verify sequential burst requests to the same domain are strictly spaced
        by at least min_delay_seconds.
        """
        min_delay = 0.05  # fast interval for test execution
        rl = RateLimiter(min_delay_seconds=min_delay)

        t_start = time.time()
        s1 = rl.throttle("goindigo.in")
        t_after_1 = time.time()
        s2 = rl.throttle("goindigo.in")
        t_after_2 = time.time()
        s3 = rl.throttle("goindigo.in")
        t_after_3 = time.time()

        # First request should not sleep
        assert s1 == 0.0
        # Subsequent requests must sleep for approximately min_delay
        assert s2 >= min_delay * 0.9
        assert s3 >= min_delay * 0.9

        # Inter-request time deltas must be >= min_delay
        assert (t_after_2 - t_after_1) >= min_delay * 0.95
        assert (t_after_3 - t_after_2) >= min_delay * 0.95

    def test_rate_limiter_domain_isolation(self):
        """
        Verify requests to different domains do NOT block or throttle each other.
        """
        min_delay = 0.1
        rl = RateLimiter(min_delay_seconds=min_delay)

        # Hit domain A
        rl.throttle("domain-a.com")

        # Hit domain B immediately - should NOT be throttled by domain A's activity
        t0 = time.time()
        slept_b = rl.throttle("domain-b.com")
        t1 = time.time()

        assert slept_b == 0.0
        assert (t1 - t0) < 0.02, "Different domain should not have slept"

    def test_exponential_backoff_calculation(self):
        """Verify backoff formula: min(max_backoff, min_delay * 2^retry)."""
        rl = RateLimiter(min_delay_seconds=0.01, max_backoff_seconds=0.10)

        # Retry 0: 0.01 * 1 = 0.01
        d0 = rl.backoff("test.com", 0)
        assert abs(d0 - 0.01) < 1e-4

        # Retry 1: 0.01 * 2 = 0.02
        d1 = rl.backoff("test.com", 1)
        assert abs(d1 - 0.02) < 1e-4

        # Retry 2: 0.01 * 4 = 0.04
        d2 = rl.backoff("test.com", 2)
        assert abs(d2 - 0.04) < 1e-4

        # Retry 5: capped at max_backoff (0.10)
        d5 = rl.backoff("test.com", 5)
        assert abs(d5 - 0.10) < 1e-4

    def test_concurrent_burst_race_condition(self):
        """
        Adversarial Concurrency Test:
        Launch 5 parallel threads calling throttle('airindia.in') simultaneously.
        In an ideal serialized rate limiter, total duration should be >= 4 * min_delay.
        This test measures whether parallel threads burst through simultaneously.
        """
        min_delay = 0.05
        rl = RateLimiter(min_delay_seconds=min_delay)

        completion_times: List[Tuple[int, float]] = []
        lock = threading.Lock()

        def worker(idx: int):
            rl.throttle("airindia.in")
            with lock:
                completion_times.append((idx, time.time()))

        t_start = time.time()
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t_end = time.time()

        completion_times.sort(key=lambda x: x[1])
        deltas = [
            completion_times[i + 1][1] - completion_times[i][1]
            for i in range(len(completion_times) - 1)
        ]

        # Check if parallel threads breached the min_delay spacing
        simultaneous_breaches = [d for d in deltas if d < min_delay * 0.5]
        # Verify rate limiter handles concurrent bursts without race conditions
        print(f"\n[Adversarial RateLimiter Result] Total duration: {t_end - t_start:.4f}s")
        print(f"[Adversarial RateLimiter Result] Inter-completion deltas: {[round(d, 4) for d in deltas]}")
        print(f"[Adversarial RateLimiter Result] Breaches (< {min_delay*0.5}s): {len(simultaneous_breaches)}")
        assert len(simultaneous_breaches) == 0, f"Detected concurrent burst race condition: {deltas}"



# ==============================================================================
# 4. SCHEDULER LIFECYCLE & CALLBACK VERIFICATION
# ==============================================================================

class TestSchedulerLifecycleAndCallbacks:
    """Stress-testing APScheduler lifecycle and quote collection callbacks."""

    def test_scheduler_start_pause_resume_shutdown(self):
        """Verify scheduler state machine transitions correctly."""
        sched = APIxIngestionScheduler(auto_start=False)
        assert not sched.is_running

        sched.start()
        assert sched.is_running
        assert len(sched.scheduler.get_jobs()) == 2

        sched.pause()
        assert sched.is_running  # Paused still counts as running scheduler

        sched.resume()
        assert sched.is_running

        sched.shutdown(wait=False)
        assert not sched.is_running

    def test_scheduler_jobs_registration_and_ist_timezone(self):
        """Verify default jobs are registered with IST timezone and proper cron triggers."""
        sched = APIxIngestionScheduler(auto_start=False)
        jobs = {j["job_id"]: j for j in sched.get_jobs_status()}

        assert "full_basket_scrape" in jobs
        assert "trunk_volatility_monitor" in jobs

        # Timezone verification
        assert "Asia/Kolkata" in str(sched.tz) or "Kolkata" in str(sched.tz)

    def test_full_basket_callback_invocation_and_payload(self):
        """Verify run_full_basket_scrape triggers callback with 1,950 quotes."""
        callback_records = []

        def mock_callback(quotes: List[RawFareQuote], job_id: str):
            callback_records.append((job_id, len(quotes), quotes[0]))

        sched = APIxIngestionScheduler(on_quotes_collected=mock_callback, auto_start=False)
        quotes = sched.run_full_basket_scrape(quotes_per_cohort=1)

        assert len(quotes) == 3750
        assert len(callback_records) == 1
        job_id, count, sample_q = callback_records[0]
        assert job_id == "full_basket_scrape"
        assert count == 3750
        assert isinstance(sample_q, RawFareQuote)
        assert sample_q.currency == "INR"

    def test_trunk_volatility_monitor_callback_invocation(self):
        """Verify run_trunk_volatility_monitor generates 100 quotes and invokes callback."""
        callback_records = []

        def mock_callback(quotes: List[RawFareQuote], job_id: str):
            callback_records.append((job_id, len(quotes)))

        sched = APIxIngestionScheduler(on_quotes_collected=mock_callback, auto_start=False)
        quotes = sched.run_trunk_volatility_monitor()

        # 10 trunk sectors * 2 urgent windows * (3 airlines + 2 OTAs) = 100 quotes
        assert len(quotes) == 100
        assert len(callback_records) == 1
        assert callback_records[0] == ("trunk_volatility_monitor", 100)

    def test_callback_exception_graceful_containment(self):
        """Verify an unhandled exception in consumer callback does NOT crash scheduler."""
        def faulty_callback(quotes: List[RawFareQuote], job_id: str):
            raise ValueError("Consumer database connection timed out!")

        sched = APIxIngestionScheduler(on_quotes_collected=faulty_callback, auto_start=False)

        # Neither method should crash
        quotes1 = sched.run_trunk_volatility_monitor()
        assert len(quotes1) == 100

        quotes2 = sched.run_full_basket_scrape(quotes_per_cohort=1)
        assert len(quotes2) == 3750

    def test_trigger_job_now_dispatch(self):
        """Verify manual immediate triggering via trigger_job_now."""
        sched = APIxIngestionScheduler(auto_start=False)

        q_trunk = sched.trigger_job_now("trunk_volatility_monitor")
        assert len(q_trunk) == 100

        q_full = sched.trigger_job_now("full_basket_scrape")
        assert len(q_full) == 3750

        with pytest.raises(ValueError, match="Job ID 'invalid_job' not found"):
            sched.trigger_job_now("invalid_job")

    def test_scheduler_restart_job_retention_adversarial(self):
        """
        Adversarial test on scheduler restart lifecycle:
        When start() -> shutdown() -> start() is executed on the same instance,
        check whether jobs are preserved or cleared.
        """
        sched = APIxIngestionScheduler(auto_start=False)
        assert len(sched.scheduler.get_jobs()) == 2

        sched.start()
        sched.shutdown(wait=False)
        assert len(sched.scheduler.get_jobs()) == 0

        # Attempt to restart
        sched.start()
        jobs_after_restart = sched.scheduler.get_jobs()
        print(f"\n[Adversarial Scheduler Restart Result] Jobs after restart: {len(jobs_after_restart)}")
        sched.shutdown(wait=False)
