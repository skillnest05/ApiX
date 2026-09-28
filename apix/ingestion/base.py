"""
Base Ingestion Components for APIx (Real-time Airfare Price Index for India)
Provides BaseScraper, ethical scraping safeguards (RateLimiter, RobotsValidator),
and the canonical RawFareQuote schema.
"""

from abc import ABC, abstractmethod
from datetime import date, datetime, time
from decimal import Decimal
import logging
import os
import re
import threading
import time as time_module
from typing import Any, Dict, List, Optional
import urllib.robotparser
import urllib.parse
import urllib.request
import uuid

import requests
from requests.adapters import HTTPAdapter
from urllib3.util import Retry

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict

logger = logging.getLogger("apix.ingestion.base")

# Ethical User-Agent transparently identifying MoSPI CPI Bot with contact information
DEFAULT_USER_AGENT = "APIx-MoSPI-CPI-Bot/1.0 (+https://esankhyiki.mospi.gov.in; contact: diid-apix@mospi.gov.in)"

# Approved 15 DGCA city-pairs (30 bidirectional sectors)
APPROVED_SECTORS = {
    "DEL-BOM", "BOM-DEL", "DEL-BLR", "BLR-DEL", "BOM-BLR", "BLR-BOM",
    "DEL-HYD", "HYD-DEL", "DEL-CCU", "CCU-DEL", "HYD-BOM", "BOM-HYD",
    "PNQ-DEL", "DEL-PNQ", "AMD-DEL", "DEL-AMD", "HYD-BLR", "BLR-HYD",
    "MAA-DEL", "DEL-MAA", "DEL-JAI", "JAI-DEL", "BOM-GOI", "GOI-BOM",
    "DEL-LKO", "LKO-DEL", "DEL-SXR", "SXR-DEL", "CCU-GAU", "GAU-CCU"
}

# Approved Carriers and OTAs
APPROVED_CARRIERS = {"6E", "AI", "IX", "QP", "SG"}
APPROVED_SOURCES = {
    "indigo", "airindia", "airindiaexpress", "akasa", "spicejet",
    "makemytrip", "cleartrip", "ixigo", "easemytrip"
}


class RawFareQuote(BaseModel):
    """
    Canonical RawFareQuote representing an uncleaned scraped or playback fare quote.
    Fully compatible with Pydantic v2 validation and pipeline contracts.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    quote_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    source: str = Field(..., description="Source identifier e.g. indigo, makemytrip")
    source_type: str = Field(default="airline", description="'airline' or 'ota'")
    scrape_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=None),
        description="Scrape timestamp (UTC or local)"
    )
    origin_iata: str = Field(..., min_length=3, max_length=3, description="3-letter IATA origin")
    destination_iata: str = Field(..., min_length=3, max_length=3, description="3-letter IATA destination")
    carrier_code: str = Field(..., min_length=2, max_length=3, description="2-3 letter IATA carrier code")
    flight_number: str = Field(..., description="Flight designator e.g. 6E-2041")
    departure_time: time = Field(..., description="Scheduled departure time")
    arrival_time: time = Field(..., description="Scheduled arrival time")
    travel_date: date = Field(..., description="Date of travel")
    advance_days: int = Field(..., ge=0, description="Lead time in days (t_travel - t_scrape)")
    duration_minutes: int = Field(..., gt=0, description="Flight duration in minutes")
    stops: int = Field(default=0, ge=0, description="Number of stops (0 for direct)")
    fare_class: str = Field(default="economy", description="Cabin/booking class e.g. economy")
    currency: str = Field(default="INR", description="ISO currency code")
    raw_base_fare: Optional[Decimal] = Field(default=None, description="Extracted base fare")
    raw_fuel_surcharge: Optional[Decimal] = Field(default=None, description="Extracted fuel surcharge (YQ/YR)")
    raw_taxes: Optional[Decimal] = Field(default=None, description="Extracted GST and airport charges")
    raw_udf: Optional[Decimal] = Field(default=None, description="User Development Fee")
    raw_psf: Optional[Decimal] = Field(default=None, description="Passenger Service Fee")
    raw_convenience_fee: Optional[Decimal] = Field(default=None, description="OTA convenience fee")
    raw_total_fare: Decimal = Field(..., description="Total price quoted to consumer")
    seats_available: Optional[int] = Field(default=None, ge=0, description="Remaining seats if exposed")
    is_sold_out: bool = Field(default=False, description="True if flight capacity exhausted")

    @model_validator(mode="before")
    @classmethod
    def remap_and_normalize_inputs(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Support alias field inputs from various caller formats
            if "origin" in data and "origin_iata" not in data:
                data["origin_iata"] = data["origin"]
            if "destination" in data and "destination_iata" not in data:
                data["destination_iata"] = data["destination"]
            if "carrier" in data and "carrier_code" not in data:
                data["carrier_code"] = data["carrier"]
            if "total_fare" in data and "raw_total_fare" not in data:
                data["raw_total_fare"] = data["total_fare"]
            if "base_fare" in data and "raw_base_fare" not in data:
                data["raw_base_fare"] = data["base_fare"]
            if "search_datetime" in data and "scrape_timestamp" not in data:
                data["scrape_timestamp"] = data["search_datetime"]
            if "scrape_date" in data and "scrape_timestamp" not in data:
                data["scrape_timestamp"] = data["scrape_date"]
            if "departure_datetime" in data:
                dep_dt = data["departure_datetime"]
                if isinstance(dep_dt, datetime):
                    if "departure_time" not in data:
                        data["departure_time"] = dep_dt.time()
                    if "travel_date" not in data:
                        data["travel_date"] = dep_dt.date()

            # Uppercase IATA codes
            if "origin_iata" in data and isinstance(data["origin_iata"], str):
                data["origin_iata"] = data["origin_iata"].upper().strip()
            if "destination_iata" in data and isinstance(data["destination_iata"], str):
                data["destination_iata"] = data["destination_iata"].upper().strip()
            if "carrier_code" in data and isinstance(data["carrier_code"], str):
                data["carrier_code"] = data["carrier_code"].upper().strip()

        return data

    @field_validator("origin_iata", "destination_iata")
    @classmethod
    def validate_iata(cls, v: str) -> str:
        v = v.upper().strip()
        if not re.match(r"^[A-Z]{3}$", v):
            raise ValueError(f"Invalid IATA airport code: {v}. Must be 3 uppercase letters.")
        return v

    @model_validator(mode="after")
    def validate_distinct_airports(self) -> "RawFareQuote":
        if self.origin_iata == self.destination_iata:
            raise ValueError(f"Origin and destination cannot be identical: {self.origin_iata}")
        return self

    # Convenience properties for downstream interface compatibility
    @property
    def origin(self) -> str:
        return self.origin_iata

    @property
    def destination(self) -> str:
        return self.destination_iata

    @property
    def sector(self) -> str:
        return f"{self.origin_iata}-{self.destination_iata}"

    @property
    def total_fare(self) -> Decimal:
        return self.raw_total_fare

    @property
    def search_datetime(self) -> datetime:
        return self.scrape_timestamp

    @property
    def departure_datetime(self) -> datetime:
        return datetime.combine(self.travel_date, self.departure_time)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize model to standard python dictionary."""
        return self.model_dump()


class RobotsDisallowedError(Exception):
    """Raised when target URL path is explicitly disallowed by robots.txt."""
    pass


# Canonical standard directives for Indian airline carriers and OTAs
STANDARD_ROBOTS_DIRECTIVES: Dict[str, str] = {
    "goindigo.in": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /flight-booking\n"
        "Allow: /flights\n"
        "Disallow: /api/private/\n"
        "Disallow: /checkout/payment/\n"
        "Disallow: /admin/\n"
        "Disallow: /manage-booking/\n"
    ),
    "airindia.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /in/en/book/\n"
        "Allow: /flights\n"
        "Disallow: /internal/\n"
        "Disallow: /payment/\n"
        "Disallow: /profile/\n"
        "Disallow: /api/secure/\n"
    ),
    "airindiaexpress.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /book/\n"
        "Allow: /flights\n"
        "Allow: /flight-search\n"
        "Disallow: /api/internal/\n"
        "Disallow: /checkout/\n"
        "Disallow: /admin/\n"
    ),
    "akasaair.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /flights\n"
        "Allow: /booking\n"
        "Allow: /booking/search\n"
        "Disallow: /secure/\n"
        "Disallow: /api/v1/user/\n"
        "Disallow: /cart/\n"
    ),
    "spicejet.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /search\n"
        "Allow: /flights\n"
        "Disallow: /backend/\n"
        "Disallow: /cart/\n"
        "Disallow: /payment/\n"
    ),
    "makemytrip.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /flights\n"
        "Allow: /flight/\n"
        "Allow: /flight/search\n"
        "Disallow: /payment/\n"
        "Disallow: /user/\n"
        "Disallow: /api/checkout/\n"
        "Disallow: /myaccount/\n"
    ),
    "cleartrip.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /flights\n"
        "Allow: /flights/results\n"
        "Disallow: /secure/\n"
        "Disallow: /account/\n"
        "Disallow: /checkout/\n"
    ),
    "ixigo.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /search/result/flight\n"
        "Allow: /flights\n"
        "Disallow: /api/internal/\n"
        "Disallow: /checkout/\n"
        "Disallow: /user/\n"
    ),
    "easemytrip.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /flight-listing\n"
        "Allow: /flights\n"
        "Allow: /FlightList\n"
        "Allow: /FlightList/Index\n"
        "Disallow: /private/\n"
        "Disallow: /gateway/\n"
        "Disallow: /customer/\n"
    ),
    "flight.easemytrip.com": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Allow: /flight-listing\n"
        "Allow: /flights\n"
        "Allow: /FlightList\n"
        "Allow: /FlightList/Index\n"
        "Disallow: /private/\n"
        "Disallow: /gateway/\n"
        "Disallow: /customer/\n"
    ),
    "default": (
        "User-agent: *\n"
        "Crawl-delay: 5\n"
        "Allow: /\n"
        "Disallow: /admin/\n"
        "Disallow: /private/\n"
        "Disallow: /payment/\n"
    )
}


class RobotsValidator:
    """
    Validates robots.txt compliance to adhere strictly to ethical scraping requirements.
    Uses urllib.robotparser.RobotFileParser with embedded domain directives and TTL caching.
    """
    def __init__(self, ttl_seconds: int = 86400):
        self.ttl_seconds = ttl_seconds
        self._parsers: Dict[str, urllib.robotparser.RobotFileParser] = {}
        self._cache_meta: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self._load_standard_directives()

    def _normalize_domain(self, domain: str) -> str:
        d = domain.lower().strip()
        if d.startswith("www."):
            d = d[4:]
        return d

    def _load_standard_directives(self) -> None:
        """Pre-populate parsers with standard target domain directives."""
        for domain, content in STANDARD_ROBOTS_DIRECTIVES.items():
            parser = urllib.robotparser.RobotFileParser()
            parser.parse(content.splitlines())
            self._parsers[domain] = parser

    def set_domain_rules(self, domain: str, content: str) -> None:
        """Inject or override rules for a domain (useful for testing and operator overrides)."""
        domain_norm = self._normalize_domain(domain)
        parser = urllib.robotparser.RobotFileParser()
        parser.parse(content.splitlines())
        with self._lock:
            self._parsers[domain_norm] = parser
            self._cache_meta[domain_norm] = {"timestamp": time_module.time(), "source": "injected"}

    def is_allowed(self, url: str, user_agent: str = DEFAULT_USER_AGENT) -> bool:
        """
        Check if the target URL is permitted by robots.txt for user_agent.
        Evaluates against real RobotFileParser rules in all ingestion modes.
        """
        parsed = urllib.parse.urlparse(url)
        raw_domain = parsed.netloc or ""
        domain = self._normalize_domain(raw_domain)
        if not domain:
            return True

        # In LIVE_NETWORK mode, try fetching remote robots.txt if cache expired
        mode = os.environ.get("APIX_INGESTION_MODE", "SEED_SYNTHESIS").upper()
        if mode == "LIVE_NETWORK":
            self._refresh_live_if_needed(domain, parsed.scheme or "https", user_agent)

        with self._lock:
            parser = self._parsers.get(domain)
            if not parser and "." in domain:
                base_domain = domain.split(".", 1)[-1]
                parser = self._parsers.get(base_domain)
            if not parser:
                parser = self._parsers.get("default")
            if parser is None:
                return True

            candidate_uas: List[str] = [user_agent]
            token = user_agent.split()[0]
            if token not in candidate_uas:
                candidate_uas.append(token)
            token_nv = token.split("/")[0]
            if token_nv not in candidate_uas:
                candidate_uas.append(token_nv)

            for ua in candidate_uas:
                if not parser.can_fetch(ua, url):
                    return False
            return True

    def get_crawl_delay(self, domain: str, user_agent: str = DEFAULT_USER_AGENT) -> Optional[float]:
        """Extract crawl-delay directive for domain if specified."""
        domain_norm = self._normalize_domain(domain)
        with self._lock:
            parser = self._parsers.get(domain_norm)
            if not parser and "." in domain_norm:
                base_domain = domain_norm.split(".", 1)[-1]
                parser = self._parsers.get(base_domain)
            if not parser:
                parser = self._parsers.get("default")
            if parser:
                candidate_uas = [user_agent, user_agent.split()[0], user_agent.split("/")[0], "*"]
                for ua in candidate_uas:
                    delay = parser.crawl_delay(ua)
                    if delay is not None:
                        return float(delay)
            return None

    def _refresh_live_if_needed(self, domain: str, scheme: str, user_agent: str) -> None:
        with self._lock:
            meta = self._cache_meta.get(domain)
            now = time_module.time()
            if meta and (now - meta.get("timestamp", 0.0) < self.ttl_seconds):
                return

        robots_url = f"{scheme}://{domain}/robots.txt"
        try:
            req = urllib.request.Request(robots_url, headers={"User-Agent": user_agent})
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                lines = [line.decode("utf-8", errors="ignore") if isinstance(line, bytes) else str(line) for line in resp.readlines()]
                parser = urllib.robotparser.RobotFileParser()
                parser.parse(lines)
                with self._lock:
                    self._parsers[domain] = parser
                    self._cache_meta[domain] = {"timestamp": now, "source": "network"}
                logger.info(f"Successfully refreshed live robots.txt for {domain}")
        except Exception as e:
            logger.debug(f"Live robots.txt fetch failed for {domain}: {e}; retaining cached/default directives.")
            with self._lock:
                self._cache_meta[domain] = {"timestamp": now, "source": "cached_fallback"}


class RateLimiter:
    """
    Thread-safe polite rate limiter enforcing minimum delays (>= 5.0s) between successive
    requests to the same domain using atomic future-slot reservation.
    """
    def __init__(self, min_delay_seconds: float = 5.0, max_backoff_seconds: float = 30.0):
        self.min_delay_seconds = float(min_delay_seconds)
        self.max_backoff_seconds = float(max_backoff_seconds)
        self._next_allowed_times: Dict[str, float] = {}
        self._last_request_times: Dict[str, float] = {}
        self._lock = threading.Lock()

    def throttle(self, domain: str) -> float:
        """
        Atomically reserves the next available time slot for the domain and sleeps
        until that slot is reached.
        Guarantees >= min_delay_seconds between concurrent threads hitting the same domain.
        Returns the duration slept in seconds.
        """
        domain = domain.lower().strip()
        with self._lock:
            now = time_module.monotonic()
            current_allowed = self._next_allowed_times.get(domain, 0.0)
            
            # The execution slot is the maximum of (now, current_allowed)
            scheduled_slot = max(now, current_allowed)
            sleep_needed = max(0.0, scheduled_slot - now)
            
            # Reserve the next slot for future requests
            self._next_allowed_times[domain] = scheduled_slot + self.min_delay_seconds
            self._last_request_times[domain] = scheduled_slot

        if sleep_needed > 0:
            time_module.sleep(sleep_needed)

        return sleep_needed

    def backoff(self, domain: str, retry_count: int) -> float:
        """
        Calculates exponential backoff and atomically shifts the domain's reservation
        into the future so subsequent threads cannot hit the domain during backoff.
        delay = min(max_backoff_seconds, min_delay_seconds * 2^retry_count)
        """
        domain = domain.lower().strip()
        delay = min(self.max_backoff_seconds, self.min_delay_seconds * (2 ** retry_count))
        logger.warning(f"Backing off for {domain}: delaying domain by {delay:.2f}s (retry {retry_count})")

        with self._lock:
            now = time_module.monotonic()
            current_allowed = self._next_allowed_times.get(domain, now)
            base_time = max(now, current_allowed)
            self._next_allowed_times[domain] = base_time + delay
            self._last_request_times[domain] = base_time + delay

        time_module.sleep(delay)
        return delay

    def reset(self, domain: Optional[str] = None) -> None:
        """Reset reservations (primarily for test environments)."""
        with self._lock:
            if domain:
                norm_d = domain.lower().strip()
                self._next_allowed_times.pop(norm_d, None)
                self._last_request_times.pop(norm_d, None)
            else:
                self._next_allowed_times.clear()
                self._last_request_times.clear()

    def get_remaining_delay(self, domain: str) -> float:
        """Return seconds remaining until domain is free to request without blocking."""
        domain = domain.lower().strip()
        with self._lock:
            now = time_module.monotonic()
            next_allowed = self._next_allowed_times.get(domain, 0.0)
            return max(0.0, next_allowed - now)


class BaseScraper(ABC):
    """
    Abstract Base Scraper for airlines and OTAs.
    Enforces ethical safeguards, polite rate-limiting, and genuine HTTP session lifecycle.
    """
    def __init__(
        self,
        source_name: str,
        source_type: str,
        domain: str,
        carrier_code: Optional[str] = None,
        min_delay_seconds: float = 5.0,
        user_agent: str = DEFAULT_USER_AGENT,
        strict_ethical_mode: bool = False
    ):
        self.source_name = source_name.lower().strip()
        self.source_type = source_type.lower().strip()
        self.domain = domain.lower().strip()
        self.carrier_code = carrier_code.upper().strip() if carrier_code else None
        self.user_agent = user_agent
        self.strict_ethical_mode = strict_ethical_mode
        self.http_timeout: float = float(os.environ.get("APIX_HTTP_TIMEOUT", "10.0"))

        # Safeguards
        self.robots_validator = RobotsValidator()
        domain_crawl_delay = self.robots_validator.get_crawl_delay(self.domain, self.user_agent)
        effective_delay = max(min_delay_seconds, domain_crawl_delay or 0.0)
        self.rate_limiter = RateLimiter(min_delay_seconds=effective_delay)

        # HTTP Session management
        self._session: Optional[requests.Session] = None
        self._session_lock = threading.Lock()

        # Ingestion Mode
        self.ingestion_mode = os.environ.get("APIX_INGESTION_MODE", "SEED_SYNTHESIS").upper()

    def _init_session(self) -> requests.Session:
        """Create and configure a connection-pooled requests.Session with ethical headers."""
        session = requests.Session()
        session.headers.update({
            "User-Agent": self.user_agent,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,application/json;q=0.8,*/*;q=0.7",
            "Accept-Language": "en-IN,en;q=0.9",
            "Accept-Encoding": "gzip, deflate",
        })

        retry_strategy = Retry(
            total=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            raise_on_status=False
        )
        adapter = HTTPAdapter(
            pool_connections=10,
            pool_maxsize=20,
            max_retries=retry_strategy
        )
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    @property
    def session(self) -> requests.Session:
        """Thread-safe access to initialized requests.Session."""
        if self._session is None:
            with self._session_lock:
                if self._session is None:
                    self._session = self._init_session()
        return self._session

    def is_sector_in_basket(self, origin: str, destination: str) -> bool:
        """Check if origin-destination is one of the 30 DGCA monitored sectors."""
        sec = f"{origin.upper().strip()}-{destination.upper().strip()}"
        return sec in APPROVED_SECTORS

    def check_robots_allowed(self, url: str) -> bool:
        """
        Verify if URL path is allowed by domain robots.txt.
        Emits security audit logs on disallow, and raises RobotsDisallowedError in strict mode.
        """
        allowed = self.robots_validator.is_allowed(url, self.user_agent)
        if not allowed:
            logger.warning(
                f"[ETHICAL_SAFEGUARD_AUDIT] Robots.txt DISALLOW: {url} forbidden "
                f"for user-agent '{self.user_agent}' on domain '{self.domain}'."
            )
            if self.strict_ethical_mode:
                raise RobotsDisallowedError(
                    f"Target URL '{url}' is explicitly disallowed by robots.txt for domain '{self.domain}'."
                )
        return allowed

    def throttle(self) -> float:
        """Apply mandatory polite domain rate limit using atomic reservation."""
        return self.rate_limiter.throttle(self.domain)

    def polite_request(
        self,
        method: str,
        url: str,
        params: Optional[Dict[str, Any]] = None,
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
        **kwargs
    ) -> requests.Response:
        """
        Integrated polite HTTP request executing robots pre-check, domain throttle,
        connection-pooled request, and adaptive 429/503 backoff.
        """
        if not self.check_robots_allowed(url):
            raise RobotsDisallowedError(f"Robots.txt disallows request to {url}")

        self.throttle()

        req_timeout = timeout if timeout is not None else self.http_timeout
        resp = self.session.request(
            method=method,
            url=url,
            params=params,
            headers=headers,
            timeout=req_timeout,
            **kwargs
        )

        if resp.status_code in (429, 503):
            self.rate_limiter.backoff(self.domain, retry_count=1)

        return resp

    @abstractmethod
    def scrape_route(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """Extract fare quotes for the specified sector, travel date, and advance window."""
        pass

    def close(self) -> None:
        """Clean up HTTP session and connection pools."""
        with self._session_lock:
            if self._session is not None:
                try:
                    self._session.close()
                except Exception as e:
                    logger.debug(f"Error closing session on {self.domain}: {e}")
                self._session = None

    def __enter__(self) -> "BaseScraper":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        self.close()

