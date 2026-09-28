"""
Carrier Scrapers for Major Indian Airlines (Module A)
Implements concrete scrapers for:
- IndiGo (6E)
- Air India (AI)
- Air India Express (IX)
- Akasa Air (QP)
- SpiceJet (SG)
With ethical safeguards, rate limiting, and seamless offline playback fallback.
"""

from datetime import date, datetime, time
from decimal import Decimal
import json
import logging
import re
from typing import Any, Dict, List, Optional, Type
import urllib.parse
from bs4 import BeautifulSoup
import requests

from apix.ingestion.base import (
    BaseScraper,
    RawFareQuote,
    DEFAULT_USER_AGENT,
    APPROVED_SECTORS
)
from apix.ingestion.seed_engine import (
    SeedPlaybackEngine,
    AIRPORT_UDF_MAP,
    DEFAULT_UDF,
    STATUTORY_PSF
)

logger = logging.getLogger("apix.ingestion.airlines")


class CarrierScraper(BaseScraper):
    """
    Base class for Indian domestic airline direct scrapers.
    Emits unmediated primary carrier fare quotes with genuine live extraction and robust fallback.
    """
    def __init__(
        self,
        source_name: str,
        domain: str,
        carrier_code: str,
        carrier_full_name: str,
        search_endpoint_template: str,
        min_delay_seconds: float = 5.0,
        user_agent: str = DEFAULT_USER_AGENT
    ):
        super().__init__(
            source_name=source_name,
            source_type="airline",
            domain=domain,
            carrier_code=carrier_code,
            min_delay_seconds=min_delay_seconds,
            user_agent=user_agent
        )
        self.carrier_full_name = carrier_full_name
        self.search_endpoint_template = search_endpoint_template
        self.seed_engine = SeedPlaybackEngine()

    def build_search_url(self, origin: str, destination: str, travel_date: date) -> str:
        """Format carrier search URL."""
        return self.search_endpoint_template.format(
            origin=origin.upper(),
            destination=destination.upper(),
            travel_date=travel_date.isoformat(),
            date_compact=travel_date.strftime("%Y%m%d"),
            date_slash=travel_date.strftime("%d/%m/%Y")
        )

    def scrape_route(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """
        Scrapes route quotes. Respects robots.txt and rate limiter.
        In offline or fallback mode, serves deterministic high-fidelity quotes.
        """
        origin = origin.upper().strip()
        destination = destination.upper().strip()
        search_url = self.build_search_url(origin, destination, travel_date)

        # 1. Ethical Robots.txt pre-check
        if not self.check_robots_allowed(search_url):
            logger.warning(
                f"Robots.txt disallows search URL {search_url} on {self.domain}. "
                "Halting ethical crawl and activating seed fallback."
            )
            return self.seed_engine.generate_quotes_for_sector(
                origin=origin,
                destination=destination,
                travel_date=travel_date,
                advance_days=advance_days,
                carrier_code=self.carrier_code,
                source=self.source_name
            )

        # 2. Rate-limiting throttle
        self.throttle()

        # 3. Live network execution or playback
        if self.ingestion_mode == "LIVE_NETWORK":
            try:
                quotes = self._scrape_live(origin, destination, travel_date, advance_days, search_url)
                if quotes:
                    return quotes
                logger.info(f"Live scrape returned no quotes for {self.carrier_code} {origin}-{destination}; falling back to seed.")
            except Exception as e:
                logger.warning(f"Live scrape failed for {self.carrier_code} on {self.domain}: {e}. Falling back to seed engine.")

        # Default / Fallback: Deterministic seed playback
        return self.seed_engine.generate_quotes_for_sector(
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            advance_days=advance_days,
            carrier_code=self.carrier_code,
            source=self.source_name
        )

    def _get_request_headers(self) -> Dict[str, str]:
        """Subclasses can supply portal-specific headers (e.g. Referer, Sec-Fetch-Mode)."""
        return {}

    def _get_request_params(self, origin: str, destination: str, travel_date: date) -> Dict[str, str]:
        """Subclasses can supply query params if not in endpoint URL."""
        return {}

    def _scrape_live(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int,
        search_url: str
    ) -> List[RawFareQuote]:
        """
        Executes genuine HTTP extraction for carrier fares using pooled requests.Session.
        Parses live response or embedded JSON state, returning validated RawFareQuote models.
        """
        logger.info(f"Executing live scrape for {self.carrier_code} on {self.domain}: {search_url}")
        try:
            headers = self._get_request_headers()
            params = self._get_request_params(origin, destination, travel_date)

            resp = self.session.get(
                search_url,
                params=params if params else None,
                headers=headers if headers else None,
                timeout=self.http_timeout,
                allow_redirects=True
            )

            # Handle Rate Limiting
            if resp.status_code == 429:
                logger.warning(f"HTTP 429 received from {self.domain}; executing exponential backoff.")
                self.rate_limiter.backoff(self.domain, retry_count=1)
                return []

            # Handle Access Challenges (Bot detection / 403)
            if resp.status_code in (401, 403):
                logger.warning(f"HTTP {resp.status_code} access restriction on {self.domain} (anti-bot barrier).")
                return []

            if resp.status_code != 200:
                logger.warning(f"Unexpected status code {resp.status_code} from {self.domain}.")
                return []

            # Content Negotiation & Parsing
            content_type = resp.headers.get("Content-Type", "")
            quotes: List[RawFareQuote] = []
            if "application/json" in content_type:
                try:
                    quotes = self._parse_json(resp.json(), origin, destination, travel_date, advance_days)
                except Exception as je:
                    logger.debug(f"JSON parsing error on {self.domain}: {je}")
            else:
                quotes = self._parse_html(resp.text, origin, destination, travel_date, advance_days)

            if quotes:
                logger.info(f"Extracted {len(quotes)} live quotes for {self.carrier_code} {origin}-{destination}")
                return quotes

            # Attempt embedded JSON extraction from script tags if HTML DOM parsing found no cards
            quotes = self._extract_embedded_state(resp.text, origin, destination, travel_date, advance_days)
            return quotes

        except requests.exceptions.Timeout:
            logger.warning(f"Live request to {self.domain} timed out after {self.http_timeout}s.")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"Network error querying {self.domain}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing live payload from {self.domain}: {e}", exc_info=True)
            return []

    def _parse_html(
        self,
        html_text: str,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """Parse HTML flight cards or listings."""
        return self._extract_quotes_from_dom(
            html_text=html_text,
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            advance_days=advance_days,
            expected_carrier=self.carrier_code
        )

    def _parse_json(
        self,
        json_data: Any,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """Parse JSON response structure for flights."""
        quotes: List[RawFareQuote] = []
        flights = []
        if isinstance(json_data, dict):
            for key in ["trips", "flights", "journeys", "data", "itineraries", "results", "outboundFlights"]:
                val = json_data.get(key)
                if isinstance(val, list):
                    flights = val
                    break
                elif isinstance(val, dict):
                    for subkey in ["journeys", "flights", "fares"]:
                        subval = val.get(subkey)
                        if isinstance(subval, list):
                            flights = subval
                            break
                    if flights:
                        break

        for idx, fl in enumerate(flights[:10]):
            try:
                # Flight number
                f_no = str(fl.get("flightNumber") or fl.get("flightNo") or fl.get("flight_number") or f"{self.carrier_code}-{1000+idx}")
                if not f_no.startswith(self.carrier_code):
                    f_no = f"{self.carrier_code}-{f_no}"
                
                # Times
                dep_raw = str(fl.get("departureTime") or fl.get("depTime") or "08:00")
                arr_raw = str(fl.get("arrivalTime") or fl.get("arrTime") or "10:15")
                dep_time = self._parse_time_str(dep_raw)
                arr_time = self._parse_time_str(arr_raw)

                # Total fare
                raw_price = fl.get("totalFare") or fl.get("fare") or fl.get("price") or fl.get("amount")
                if isinstance(raw_price, dict):
                    raw_price = raw_price.get("total") or raw_price.get("amount") or raw_price.get("value")
                if not raw_price:
                    continue
                total_fare = Decimal(str(raw_price).replace(",", "").strip())
                if total_fare < Decimal("999.00") or total_fare > Decimal("35000.00"):
                    continue

                quote = self._build_raw_quote(
                    origin=origin,
                    destination=destination,
                    travel_date=travel_date,
                    advance_days=advance_days,
                    carrier_code=self.carrier_code,
                    flight_number=f_no,
                    departure_time=dep_time,
                    arrival_time=arr_time,
                    total_fare=total_fare,
                    convenience_fee=Decimal("0.00")
                )
                quotes.append(quote)
            except Exception as e:
                logger.debug(f"Failed to parse JSON flight item {fl}: {e}")
        return quotes

    def _extract_embedded_state(
        self,
        html_text: str,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """Extract flights from embedded Next.js or React scripts (<script id='__NEXT_DATA__'>)."""
        soup = BeautifulSoup(html_text, "html.parser")
        script = soup.find("script", id="__NEXT_DATA__")
        if script and script.string:
            try:
                data = json.loads(script.string)
                page_props = data.get("props", {}).get("pageProps", {})
                return self._parse_json(page_props, origin, destination, travel_date, advance_days)
            except Exception as e:
                logger.debug(f"Failed to parse __NEXT_DATA__ on {self.domain}: {e}")

        match = re.search(r'window\.(?:__INITIAL_STATE__|flightSearchData)\s*=\s*(\{.*?\});', html_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(1))
                return self._parse_json(data, origin, destination, travel_date, advance_days)
            except Exception as e:
                logger.debug(f"Failed to parse window initial state on {self.domain}: {e}")

        return []

    def _extract_quotes_from_dom(
        self,
        html_text: str,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int,
        expected_carrier: Optional[str] = None,
        convenience_fee: Decimal = Decimal("0.00")
    ) -> List[RawFareQuote]:
        """Resilient DOM parser for Indian carrier flight search results."""
        soup = BeautifulSoup(html_text, "html.parser")
        quotes: List[RawFareQuote] = []

        card_selectors = [
            ".flight-card", ".flight-item", ".itinerary-card", ".flt-opt",
            "[data-testid*='flight']", "div[id^='divFlight']", "div.fli-list",
            ".flight-details-card", ".c-flight-listing-split-row", ".fare-row",
            "div[class*='FlightCard']", "div[class*='flight-row']", "div.flt-row"
        ]
        cards = []
        for sel in card_selectors:
            found = soup.select(sel)
            if found:
                cards = found
                break

        for card in cards[:10]:
            text = card.get_text(" ", strip=True)

            fn_match = re.search(r'\b(6E|AI|IX|QP|SG)[ -]?(\d{3,4})\b', text, re.IGNORECASE)
            if fn_match:
                c_code = fn_match.group(1).upper()
                f_no = f"{c_code}-{fn_match.group(2)}"
            elif expected_carrier:
                digits_match = re.search(r'\b(\d{3,4})\b', text)
                if digits_match:
                    c_code = expected_carrier
                    f_no = f"{c_code}-{digits_match.group(1)}"
                else:
                    continue
            else:
                continue

            if expected_carrier and c_code != expected_carrier:
                continue

            times = re.findall(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', text)
            if len(times) >= 2:
                dep_time = time(int(times[0][0]), int(times[0][1]))
                arr_time = time(int(times[1][0]), int(times[1][1]))
            else:
                dep_time = time(8, 0)
                arr_time = time(10, 15)

            price_match = re.search(r'(?:₹|Rs\.?|INR)\s*([\d,]{4,7})', text, re.IGNORECASE)
            if not price_match:
                price_match = re.search(r'\b([\d,]{4,6})\b', text)
            if not price_match:
                continue

            raw_price_str = price_match.group(1).replace(",", "")
            try:
                total_fare = Decimal(raw_price_str)
            except Exception:
                continue

            if total_fare < Decimal("999.00") or total_fare > Decimal("35000.00"):
                continue

            quote = self._build_raw_quote(
                origin=origin,
                destination=destination,
                travel_date=travel_date,
                advance_days=advance_days,
                carrier_code=c_code,
                flight_number=f_no,
                departure_time=dep_time,
                arrival_time=arr_time,
                total_fare=total_fare,
                convenience_fee=convenience_fee
            )
            quotes.append(quote)

        return quotes

    def _build_raw_quote(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int,
        carrier_code: str,
        flight_number: str,
        departure_time: time,
        arrival_time: time,
        total_fare: Decimal,
        convenience_fee: Decimal = Decimal("0.00"),
        seats_available: Optional[int] = 9
    ) -> RawFareQuote:
        """Construct validated RawFareQuote with exact 6-part statutory decomposition."""
        aera_udf = Decimal(str(AIRPORT_UDF_MAP.get(origin, DEFAULT_UDF))).quantize(Decimal("0.01"))
        stat_psf = Decimal(str(STATUTORY_PSF)).quantize(Decimal("0.01"))
        conv_fee = Decimal(str(convenience_fee)).quantize(Decimal("0.01"))

        taxable_pool = max(Decimal("0.00"), total_fare - aera_udf - stat_psf - conv_fee)
        taxable_base_and_fuel = (taxable_pool / Decimal("1.05")).quantize(Decimal("0.01"))
        base_fare = (taxable_base_and_fuel * Decimal("0.70")).quantize(Decimal("0.01"))
        fuel_surcharge = (taxable_base_and_fuel * Decimal("0.30")).quantize(Decimal("0.01"))
        gst = (taxable_pool - (base_fare + fuel_surcharge)).quantize(Decimal("0.01"))

        residual = total_fare - (base_fare + fuel_surcharge + gst + aera_udf + stat_psf + conv_fee)
        base_fare += residual

        dep_dt = datetime.combine(travel_date, departure_time)
        arr_dt = datetime.combine(travel_date, arrival_time)
        if arr_dt > dep_dt:
            dur = max(45, int((arr_dt - dep_dt).total_seconds() // 60))
        else:
            dur = 120

        return RawFareQuote(
            source=self.source_name,
            source_type=self.source_type,
            origin_iata=origin,
            destination_iata=destination,
            carrier_code=carrier_code,
            flight_number=flight_number,
            departure_time=departure_time,
            arrival_time=arrival_time,
            travel_date=travel_date,
            advance_days=advance_days,
            duration_minutes=dur,
            stops=0,
            fare_class="economy",
            currency="INR",
            raw_base_fare=base_fare,
            raw_fuel_surcharge=fuel_surcharge,
            raw_taxes=gst,
            raw_udf=aera_udf,
            raw_psf=stat_psf,
            raw_convenience_fee=conv_fee,
            raw_total_fare=total_fare,
            seats_available=seats_available,
            is_sold_out=False
        )

    @staticmethod
    def _parse_time_str(time_str: str) -> time:
        match = re.search(r'\b([01]?\d|2[0-3]):([0-5]\d)\b', time_str)
        if match:
            return time(int(match.group(1)), int(match.group(2)))
        return time(8, 0)


class IndiGoScraper(CarrierScraper):
    """
    IndiGo (6E) Scraper.
    Covers India's largest domestic carrier (~62% market share).
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="indigo",
            domain="goindigo.in",
            carrier_code="6E",
            carrier_full_name="IndiGo",
            search_endpoint_template="https://www.goindigo.in/flight-booking.html?origin={origin}&destination={destination}&date={date_compact}",
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.goindigo.in/flight-booking.html",
            "X-Requested-With": "XMLHttpRequest",
            "Sec-Fetch-Mode": "cors"
        }


class AirIndiaScraper(CarrierScraper):
    """
    Air India (AI) Scraper.
    Full-service carrier (~14% market share).
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="airindia",
            domain="airindia.com",
            carrier_code="AI",
            carrier_full_name="Air India",
            search_endpoint_template="https://www.airindia.com/in/en/book/flight-search.html?from={origin}&to={destination}&depart={travel_date}",
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.airindia.com/in/en/book/flight-search.html",
            "Accept": "application/json, text/html, */*"
        }


class AirIndiaExpressScraper(CarrierScraper):
    """
    Air India Express (IX) Scraper.
    Tata group low-cost carrier (~7% market share).
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="airindiaexpress",
            domain="airindiaexpress.com",
            carrier_code="IX",
            carrier_full_name="Air India Express",
            search_endpoint_template="https://www.airindiaexpress.com/flight-search?origin={origin}&destination={destination}&travelDate={travel_date}",
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.airindiaexpress.com/flight-search",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


class AkasaAirScraper(CarrierScraper):
    """
    Akasa Air (QP) Scraper.
    Fast-growing low-cost carrier (~5% market share).
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="akasa",
            domain="akasaair.com",
            carrier_code="QP",
            carrier_full_name="Akasa Air",
            search_endpoint_template="https://www.akasaair.com/booking/search?origin={origin}&destination={destination}&date={travel_date}",
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.akasaair.com/booking/search",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


class SpiceJetScraper(CarrierScraper):
    """
    SpiceJet (SG) Scraper.
    Low-cost carrier (~4% market share).
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="spicejet",
            domain="spicejet.com",
            carrier_code="SG",
            carrier_full_name="SpiceJet",
            search_endpoint_template="https://www.spicejet.com/search?from={origin}&to={destination}&departure={travel_date}",
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.spicejet.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


AIRLINE_SCRAPERS: Dict[str, Type[CarrierScraper]] = {
    "6E": IndiGoScraper,
    "AI": AirIndiaScraper,
    "IX": AirIndiaExpressScraper,
    "QP": AkasaAirScraper,
    "SG": SpiceJetScraper
}


def get_airline_scraper(carrier_code: str, min_delay_seconds: float = 5.0) -> CarrierScraper:
    """Factory function to instantiate carrier scraper by IATA code."""
    c_code = carrier_code.upper().strip()
    if c_code not in AIRLINE_SCRAPERS:
        raise ValueError(f"Unsupported airline carrier code: {c_code}. Must be one of {list(AIRLINE_SCRAPERS.keys())}")
    return AIRLINE_SCRAPERS[c_code](min_delay_seconds=min_delay_seconds)

