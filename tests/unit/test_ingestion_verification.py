"""
APIx Comprehensive Verification Suite for Module A (Ingestion & Playback Engine):
1. RateLimiter Multi-Threading Concurrency, Future Slot Reservation & Domain Isolation
2. RobotsValidator RFC Compliance, Standard Directives & Graceful Network Fallbacks
3. BaseScraper HTTP Session Lifecycle & Connection Pooling
4. _scrape_live Execution, Network Error Containment & Resilient Seed Fallbacks
5. OTA Carrier Diversity across all 5 monitored airlines (6E, AI, IX, QP, SG)
"""

from datetime import date, datetime, time
from decimal import Decimal
import logging
import os
import threading
import time as time_module
from typing import List, Tuple
import urllib.error
import pytest

from apix.ingestion.base import (
    RateLimiter,
    RobotsValidator,
    RobotsDisallowedError,
    STANDARD_ROBOTS_DIRECTIVES,
    RawFareQuote,
    DEFAULT_USER_AGENT,
    APPROVED_SECTORS,
    APPROVED_CARRIERS
)
from apix.ingestion.airlines import (
    IndiGoScraper,
    AirIndiaScraper,
    AirIndiaExpressScraper,
    AkasaAirScraper,
    SpiceJetScraper,
    AIRLINE_SCRAPERS
)
from apix.ingestion.otas import (
    MakeMyTripScraper,
    CleartripScraper,
    IxigoScraper,
    EaseMyTripScraper,
    OTA_SCRAPERS,
    get_ota_scraper
)
from apix.ingestion.seed_engine import SeedPlaybackEngine


# ==============================================================================
# 1. RATE LIMITER CONCURRENCY & THREAD SAFETY TESTS
# ==============================================================================

class TestRateLimiterConcurrency:
    """Rigorous tests asserting zero race conditions and strict serialized throttling."""

    def test_rate_limiter_multithread_serialization(self):
        """5 concurrent threads to same domain must be spaced by >= min_delay * 0.85."""
        min_delay = 0.04
        rl = RateLimiter(min_delay_seconds=min_delay)
        dispatches = []
        lock = threading.Lock()

        def worker():
            rl.throttle("goindigo.in")
            with lock:
                dispatches.append(time_module.monotonic())

        threads = [threading.Thread(target=worker) for _ in range(5)]
        t0 = time_module.monotonic()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        t1 = time_module.monotonic()

        dispatches.sort()
        deltas = [dispatches[i+1] - dispatches[i] for i in range(len(dispatches)-1)]

        # Must have zero simultaneous dispatches
        simultaneous = [d for d in deltas if d < min_delay * 0.5]
        assert len(simultaneous) == 0, f"Detected concurrent burst race condition: {deltas}"
        # All intervals must respect min_delay (within 15% timer jitter)
        assert all(d >= min_delay * 0.85 for d in deltas), f"Deltas breached min_delay: {deltas}"
        # Total duration must cover serialized requests
        assert (t1 - t0) >= (4 * min_delay * 0.85)

    def test_rate_limiter_domain_concurrency_isolation(self):
        """Requests to distinct domains must execute in parallel without cross-blocking."""
        min_delay = 0.10
        rl = RateLimiter(min_delay_seconds=min_delay)
        results = {}
        lock = threading.Lock()

        def worker(domain: str, idx: int):
            rl.throttle(domain)
            with lock:
                results[f"{domain}_{idx}"] = time_module.monotonic()

        t_start = time_module.monotonic()
        threads = [
            threading.Thread(target=worker, args=("goindigo.in", 1)),
            threading.Thread(target=worker, args=("goindigo.in", 2)),
            threading.Thread(target=worker, args=("airindia.com", 1)),
            threading.Thread(target=worker, args=("airindia.com", 2)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        total_duration = time_module.monotonic() - t_start

        # Intra-domain separation
        indigo_delta = abs(results["goindigo.in_2"] - results["goindigo.in_1"])
        ai_delta = abs(results["airindia.com_2"] - results["airindia.com_1"])
        assert indigo_delta >= min_delay * 0.85
        assert ai_delta >= min_delay * 0.85

        # Cross-domain concurrency: total time should be ~1 delay, definitely < 2.5 * delay
        assert total_duration < min_delay * 2.5, f"Domains blocked each other: took {total_duration:.3f}s"

    def test_rate_limiter_institutional_default_delay(self):
        """Verify default delay complies with R1 (>= 5.0s)."""
        rl = RateLimiter()
        assert rl.min_delay_seconds >= 5.0

    def test_rate_limiter_backoff_advances_reservation(self):
        """Verify backoff atomically shifts future reservation."""
        rl = RateLimiter(min_delay_seconds=0.05, max_backoff_seconds=1.0)
        delay = rl.backoff("test.com", retry_count=1)
        assert delay >= 0.10
        remaining = rl.get_remaining_delay("test.com")
        # Remaining delay should be non-negative
        assert remaining >= 0.0

    def test_rate_limiter_reset(self):
        """Verify reset clears reservations."""
        rl = RateLimiter(min_delay_seconds=0.05)
        rl.throttle("example.com")
        assert rl.get_remaining_delay("example.com") >= 0.0
        rl.reset("example.com")
        assert rl.get_remaining_delay("example.com") == 0.0


# ==============================================================================
# 2. ROBOTS VALIDATOR COMPLIANCE TESTS
# ==============================================================================

class TestRobotsValidatorCompliance:
    """Tests RFC 9309 compliance, caching, and offline fallbacks."""

    def test_robots_validator_rule_parsing_allow_and_disallow(self, monkeypatch):
        """Verify parsing of allow and disallow directives for APIx bot user-agent."""
        sample_robots = (
            "User-agent: APIx-MoSPI-CPI-Bot/1.0\n"
            "Disallow: /admin/\n"
            "Disallow: /api/internal/\n"
            "Allow: /flight-booking\n"
            "Allow: /flights/\n"
            "\n"
            "User-agent: *\n"
            "Disallow: /secret/\n"
        )

        class MockResponse:
            def readlines(self):
                return [line.encode("utf-8") for line in sample_robots.splitlines(keepends=True)]
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: MockResponse())
        monkeypatch.setenv("APIX_INGESTION_MODE", "LIVE_NETWORK")

        validator = RobotsValidator(ttl_seconds=3600)
        assert validator.is_allowed("https://goindigo.in/flight-booking?from=DEL&to=BOM") is True
        assert validator.is_allowed("https://goindigo.in/admin/users") is False
        assert validator.is_allowed("https://goindigo.in/api/internal/rates") is False
        assert validator.is_allowed("https://goindigo.in/secret/fares") is False

    def test_robots_validator_cache_ttl(self, monkeypatch):
        """Verify multiple queries within TTL reuse cache and make only 1 fetch."""
        fetch_count = 0
        sample_robots = "User-agent: *\nAllow: /\n"

        class MockResponse:
            def readlines(self):
                nonlocal fetch_count
                fetch_count += 1
                return [sample_robots.encode("utf-8")]
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: MockResponse())
        monkeypatch.setenv("APIX_INGESTION_MODE", "LIVE_NETWORK")

        validator = RobotsValidator(ttl_seconds=3600)
        for _ in range(5):
            validator.is_allowed("https://airindia.com/in/en/book/flight-search.html")

        assert fetch_count == 1

    def test_robots_validator_network_error_graceful_containment(self, monkeypatch):
        """When robots.txt fetch fails, validator must not raise unhandled exception."""
        def mock_urlopen_fail(*args, **kwargs):
            raise urllib.error.URLError("Connection refused by target host")

        monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_fail)
        monkeypatch.setenv("APIX_INGESTION_MODE", "LIVE_NETWORK")

        validator = RobotsValidator()
        allowed = validator.is_allowed("https://makemytrip.com/flight/search")
        assert allowed is True

    def test_robots_validator_standard_embedded_directives(self):
        """Verify embedded directives for all 9 domains validate correctly offline."""
        validator = RobotsValidator()
        # Airline domains
        assert validator.is_allowed("https://www.goindigo.in/flight-booking.html") is True
        assert validator.is_allowed("https://goindigo.in/api/private/fares") is False
        assert validator.is_allowed("https://airindia.com/in/en/book/flight-search.html") is True
        assert validator.is_allowed("https://airindia.com/internal/debug") is False
        assert validator.is_allowed("https://airindiaexpress.com/flight-search") is True
        assert validator.is_allowed("https://airindiaexpress.com/admin/login") is False
        assert validator.is_allowed("https://akasaair.com/booking/search") is True
        assert validator.is_allowed("https://akasaair.com/cart/checkout") is False
        assert validator.is_allowed("https://spicejet.com/search") is True
        assert validator.is_allowed("https://spicejet.com/backend/fares") is False

        # OTA domains
        assert validator.is_allowed("https://makemytrip.com/flight/search") is True
        assert validator.is_allowed("https://makemytrip.com/payment/process") is False
        assert validator.is_allowed("https://cleartrip.com/flights/results") is True
        assert validator.is_allowed("https://cleartrip.com/secure/account") is False
        assert validator.is_allowed("https://ixigo.com/search/result/flight/DEL/BOM") is True
        assert validator.is_allowed("https://ixigo.com/api/internal/checkout") is False
        assert validator.is_allowed("https://flight.easemytrip.com/FlightList/Index") is True
        assert validator.is_allowed("https://flight.easemytrip.com/gateway/pay") is False

    def test_robots_validator_crawl_delay_extraction(self):
        """Verify crawl delay extraction from standard directives."""
        validator = RobotsValidator()
        delay = validator.get_crawl_delay("goindigo.in")
        assert delay == 5.0

    def test_robots_validator_set_domain_rules(self):
        """Verify custom dynamic rules injection."""
        validator = RobotsValidator()
        custom = "User-agent: *\nDisallow: /test-blocked/\nAllow: /test-allowed/\n"
        validator.set_domain_rules("custom-airline.in", custom)
        assert validator.is_allowed("https://custom-airline.in/test-allowed/fares") is True
        assert validator.is_allowed("https://custom-airline.in/test-blocked/fares") is False


# ==============================================================================
# 3. BASE SCRAPER & HTTP SESSION LIFECYCLE TESTS
# ==============================================================================

class TestBaseScraperSessionLifecycle:
    """Verifies connection pooling, polite headers, and socket cleanup."""

    def test_scraper_session_initialization_and_headers(self):
        """Verify session is lazily created with MoSPI User-Agent and connection pooling."""
        scraper = IndiGoScraper(min_delay_seconds=0.001)
        sess = scraper.session
        assert sess is not None
        assert DEFAULT_USER_AGENT in sess.headers["User-Agent"]
        assert "en-IN" in sess.headers["Accept-Language"]
        scraper.close()
        assert scraper._session is None

    def test_scraper_context_manager_lifecycle(self):
        """Verify BaseScraper cleanly manages session lifecycle via context manager."""
        with IndiGoScraper(min_delay_seconds=0.001) as scraper:
            sess = scraper.session
            assert sess is not None
        assert scraper._session is None

    def test_polite_request_strict_ethical_mode_blocks_disallowed(self):
        """In strict ethical mode, polite_request must raise RobotsDisallowedError if URL is forbidden."""
        scraper = IndiGoScraper(min_delay_seconds=0.001)
        scraper.strict_ethical_mode = True
        with pytest.raises(RobotsDisallowedError):
            scraper.polite_request("GET", "https://goindigo.in/api/private/rates")
        scraper.close()


# ==============================================================================
# 4. LIVE SCRAPING & SEED PLAYBACK FALLBACK TESTS
# ==============================================================================

class TestLiveScrapingAndFallback:
    """Tests live scraping execution and robust fallback under all error conditions."""

    def test_scrape_live_success_path(self, monkeypatch):
        """When live scraper returns quotes, scrape_route must return them without seed fallback."""
        scraper = IndiGoScraper(min_delay_seconds=0.001)
        monkeypatch.setattr(scraper, "ingestion_mode", "LIVE_NETWORK")
        monkeypatch.setattr(scraper, "check_robots_allowed", lambda url: True)

        live_quote = RawFareQuote(
            source="indigo",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            flight_number="6E-9999",
            departure_time=time(6, 0),
            arrival_time=time(8, 15),
            travel_date=date(2026, 10, 1),
            advance_days=7,
            duration_minutes=135,
            raw_base_fare=Decimal("4500.00"),
            raw_fuel_surcharge=Decimal("600.00"),
            raw_taxes=Decimal("255.00"),
            raw_udf=Decimal("450.00"),
            raw_psf=Decimal("91.00"),
            raw_convenience_fee=Decimal("0.00"),
            raw_total_fare=Decimal("5896.00")
        )
        monkeypatch.setattr(scraper, "_scrape_live", lambda *args, **kwargs: [live_quote])

        quotes = scraper.scrape_route("DEL", "BOM", date(2026, 10, 1), 7)
        assert len(quotes) == 1
        assert quotes[0].flight_number == "6E-9999"
        assert quotes[0].raw_total_fare == Decimal("5896.00")
        scraper.close()

    def test_scrape_live_network_exception_fallback(self, monkeypatch, caplog):
        """When live scraping raises network exception, clean fallback to seed engine occurs."""
        scraper = IndiGoScraper(min_delay_seconds=0.001)
        monkeypatch.setattr(scraper, "ingestion_mode", "LIVE_NETWORK")
        monkeypatch.setattr(scraper, "check_robots_allowed", lambda url: True)

        def mock_timeout(*args, **kwargs):
            import requests
            raise requests.exceptions.ConnectTimeout("Connection timed out to goindigo.in")

        monkeypatch.setattr(scraper, "_scrape_live", mock_timeout)

        with caplog.at_level(logging.WARNING):
            quotes = scraper.scrape_route("DEL", "BOM", date(2026, 10, 1), 7)

        assert len(quotes) >= 1
        assert quotes[0].carrier_code == "6E"
        assert quotes[0].origin_iata == "DEL"
        assert quotes[0].destination_iata == "BOM"
        assert any("falling back" in rec.message.lower() or "failed" in rec.message.lower() for rec in caplog.records)
        scraper.close()

    def test_scrape_live_empty_payload_fallback(self, monkeypatch, caplog):
        """When live scraping returns empty list, clean fallback to seed engine occurs."""
        scraper = MakeMyTripScraper(min_delay_seconds=0.001)
        monkeypatch.setattr(scraper, "ingestion_mode", "LIVE_NETWORK")
        monkeypatch.setattr(scraper, "check_robots_allowed", lambda url: True)
        monkeypatch.setattr(scraper, "_scrape_live", lambda *args, **kwargs: [])

        with caplog.at_level(logging.INFO):
            quotes = scraper.scrape_route("DEL", "BOM", date(2026, 10, 1), 7)

        assert len(quotes) >= 1
        assert quotes[0].source == "makemytrip"
        assert any("falling back" in rec.message.lower() for rec in caplog.records)
        scraper.close()

    def test_scrape_route_robots_disallow_block(self, monkeypatch, caplog):
        """When robots.txt disallows search URL, seed fallback serves quotes without live scraping."""
        scraper = IndiGoScraper(min_delay_seconds=0.001)
        monkeypatch.setattr(scraper, "ingestion_mode", "LIVE_NETWORK")
        monkeypatch.setattr(scraper, "check_robots_allowed", lambda url: False)

        live_called = False
        def mock_live(*args, **kwargs):
            nonlocal live_called
            live_called = True
            return []
        monkeypatch.setattr(scraper, "_scrape_live", mock_live)

        with caplog.at_level(logging.WARNING):
            quotes = scraper.scrape_route("DEL", "BOM", date(2026, 10, 1), 7)

        assert live_called is False, "Live scraping should not be called when robots disallows URL"
        assert len(quotes) >= 1
        assert quotes[0].carrier_code == "6E"
        scraper.close()


# ==============================================================================
# 5. OTA CARRIER DIVERSITY TESTS (FINDING 4)
# ==============================================================================

class TestOtaCarrierDiversity:
    """Verifies that all 5 monitored carriers are fully represented in OTA quotes."""

    def test_full_basket_contains_all_5_carriers_for_every_ota(self):
        """
        In generate_full_basket(), each OTA must generate quotes for all 5 carriers:
        IndiGo (6E), Air India (AI), Air India Express (IX), Akasa Air (QP), SpiceJet (SG).
        """
        engine = SeedPlaybackEngine(seed=42)
        quotes = engine.generate_full_basket()

        # Total quotes = 30 sectors * 5 windows * (5 carriers + 4 OTAs * 5 carriers) = 3750
        assert len(quotes) == 3750, f"Expected 3750 quotes, got {len(quotes)}"

        expected_carriers = {"6E", "AI", "IX", "QP", "SG"}
        otas = ["makemytrip", "cleartrip", "ixigo", "easemytrip"]

        for ota in otas:
            ota_quotes = [q for q in quotes if q.source == ota]
            # Each OTA has 30 sectors * 5 windows * 5 carriers = 750 quotes
            assert len(ota_quotes) == 750, f"OTA {ota} expected 750 quotes, got {len(ota_quotes)}"
            ota_carriers = set(q.carrier_code for q in ota_quotes)
            assert ota_carriers == expected_carriers, f"OTA {ota} missing carriers: {expected_carriers - ota_carriers}"

            from collections import Counter
            carrier_counts = Counter(q.carrier_code for q in ota_quotes)
            for c in expected_carriers:
                assert carrier_counts[c] == 150, f"OTA {ota} carrier {c} count {carrier_counts[c]} != 150"

    def test_persisted_seed_quotes_carrier_diversity(self):
        """Verify data/seed_quotes.json contains all 5 carriers for each OTA."""
        engine = SeedPlaybackEngine(seed=42)
        quotes = engine.load_seed_dataset()
        expected_carriers = {"6E", "AI", "IX", "QP", "SG"}
        otas = ["makemytrip", "cleartrip", "ixigo", "easemytrip"]

        for ota in otas:
            ota_quotes = [q for q in quotes if q.source == ota]
            ota_carriers = set(q.carrier_code for q in ota_quotes)
            assert ota_carriers == expected_carriers, f"Persisted {ota} missing carriers: {expected_carriers - ota_carriers}"
