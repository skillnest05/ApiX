"""
APIx Ingestion & Playback Engine (Module A)
Provides multi-source scraping for 5 Indian carriers and 4 OTAs,
ethical scraping safeguards, deterministic offline seed playback,
and automated APScheduler cron execution.
"""

from apix.ingestion.base import (
    BaseScraper,
    RawFareQuote,
    RateLimiter,
    RobotsValidator,
    DEFAULT_USER_AGENT,
    APPROVED_SECTORS,
    APPROVED_CARRIERS,
    APPROVED_SOURCES
)
from apix.ingestion.airlines import (
    CarrierScraper,
    IndiGoScraper,
    AirIndiaScraper,
    AirIndiaExpressScraper,
    AkasaAirScraper,
    SpiceJetScraper,
    AIRLINE_SCRAPERS,
    get_airline_scraper
)
from apix.ingestion.otas import (
    OTAScraper,
    MakeMyTripScraper,
    CleartripScraper,
    IxigoScraper,
    EaseMyTripScraper,
    OTA_SCRAPERS,
    get_ota_scraper
)
from apix.ingestion.seed_engine import SeedPlaybackEngine
from apix.ingestion.scheduler import APIxIngestionScheduler

__all__ = [
    # Base and Schema
    "BaseScraper",
    "RawFareQuote",
    "RateLimiter",
    "RobotsValidator",
    "DEFAULT_USER_AGENT",
    "APPROVED_SECTORS",
    "APPROVED_CARRIERS",
    "APPROVED_SOURCES",
    # Carriers
    "CarrierScraper",
    "IndiGoScraper",
    "AirIndiaScraper",
    "AirIndiaExpressScraper",
    "AkasaAirScraper",
    "SpiceJetScraper",
    "AIRLINE_SCRAPERS",
    "get_airline_scraper",
    # OTAs
    "OTAScraper",
    "MakeMyTripScraper",
    "CleartripScraper",
    "IxigoScraper",
    "EaseMyTripScraper",
    "OTA_SCRAPERS",
    "get_ota_scraper",
    # Seed Playback & Scheduler
    "SeedPlaybackEngine",
    "APIxIngestionScheduler",
]
