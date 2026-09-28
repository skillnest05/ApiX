"""
OTA Scrapers for Major Indian Online Travel Agencies (Module A)
Implements concrete scrapers for:
- MakeMyTrip (MMT)
- Cleartrip (CT)
- Ixigo (IXI)
- EaseMyTrip (EMT)
With ethical safeguards, convenience fee tracking, and offline playback fallback.
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
    DEFAULT_USER_AGENT
)
from apix.ingestion.seed_engine import (
    SeedPlaybackEngine,
    AIRPORT_UDF_MAP,
    DEFAULT_UDF,
    STATUTORY_PSF
)

logger = logging.getLogger("apix.ingestion.otas")


class OTAScraper(BaseScraper):
    """
    Base class for Online Travel Agency (OTA) aggregators.
    Aggregates fares across multiple airlines and captures platform convenience fees.
    """
    def __init__(
        self,
        source_name: str,
        platform_code: str,
        domain: str,
        platform_full_name: str,
        search_endpoint_template: str,
        convenience_fee_standard: float = 300.0,
        min_delay_seconds: float = 5.0,
        user_agent: str = DEFAULT_USER_AGENT
    ):
        super().__init__(
            source_name=source_name,
            source_type="ota",
            domain=domain,
            carrier_code=None,  # OTAs aggregate multiple carriers
            min_delay_seconds=min_delay_seconds,
            user_agent=user_agent
        )
        self.platform_code = platform_code.upper().strip()
        self.platform_full_name = platform_full_name
        self.search_endpoint_template = search_endpoint_template
        self.convenience_fee_standard = convenience_fee_standard
        self.seed_engine = SeedPlaybackEngine()

    def build_search_url(self, origin: str, destination: str, travel_date: date) -> str:
        """Format OTA flight search URL."""
        return self.search_endpoint_template.format(
            origin=origin.upper(),
            destination=destination.upper(),
            travel_date=travel_date.isoformat(),
            date_compact=travel_date.strftime("%d%m%Y"),
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
        Extract aggregated multi-carrier fare quotes for the sector and date.
        Respects robots.txt and rate limits. Falls back cleanly to seed playback.
        """
        origin = origin.upper().strip()
        destination = destination.upper().strip()
        search_url = self.build_search_url(origin, destination, travel_date)

        # 1. Robots.txt ethical validation
        if not self.check_robots_allowed(search_url):
            logger.warning(
                f"Robots.txt disallows search on {self.domain} for {search_url}. "
                "Halting ethical scrape and using seed playback."
            )
            return self.seed_engine.generate_quotes_for_sector(
                origin=origin,
                destination=destination,
                travel_date=travel_date,
                advance_days=advance_days,
                carrier_code=None,
                source=self.source_name
            )

        # 2. Rate-limiting polite throttle
        self.throttle()

        # 3. Live network execution or playback
        if self.ingestion_mode == "LIVE_NETWORK":
            try:
                quotes = self._scrape_live(origin, destination, travel_date, advance_days, search_url)
                if quotes:
                    return quotes
                logger.info(f"Live scrape returned no quotes on {self.platform_code}; falling back to seed playback.")
            except Exception as e:
                logger.warning(f"Live scrape failed on {self.domain}: {e}. Activating seed fallback.")

        # Default / Fallback: Deterministic seed playback across available carriers
        return self.seed_engine.generate_quotes_for_sector(
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            advance_days=advance_days,
            carrier_code=None,
            source=self.source_name
        )

    def _get_request_headers(self) -> Dict[str, str]:
        """Subclasses can supply portal-specific headers."""
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
        Executes genuine HTTP extraction for multi-carrier quotes on OTA aggregators.
        Attaches statutory platform convenience fee.
        """
        logger.info(f"Executing live OTA extraction for {self.platform_code} on {self.domain}: {search_url}")
        try:
            headers = self._get_request_headers()
            resp = self.session.get(
                search_url,
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
                logger.warning(f"OTA access restricted (HTTP {resp.status_code}) on {self.domain}.")
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
                logger.info(f"Extracted {len(quotes)} live multi-carrier quotes from {self.platform_code} {origin}-{destination}")
                return quotes

            quotes = self._extract_embedded_state(resp.text, origin, destination, travel_date, advance_days)
            return quotes

        except requests.exceptions.Timeout:
            logger.warning(f"Live request to OTA {self.domain} timed out after {self.http_timeout}s.")
            return []
        except requests.exceptions.RequestException as e:
            logger.warning(f"Live OTA scrape network error on {self.domain}: {e}")
            return []
        except Exception as e:
            logger.error(f"Error parsing OTA response from {self.domain}: {e}", exc_info=True)
            return []

    def _parse_html(
        self,
        html_text: str,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """Parse HTML flight cards or listings for multi-carrier quotes."""
        return self._extract_quotes_from_dom(
            html_text=html_text,
            origin=origin,
            destination=destination,
            travel_date=travel_date,
            advance_days=advance_days,
            convenience_fee=Decimal(str(self.convenience_fee_standard))
        )

    def _parse_json(
        self,
        json_data: Any,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int
    ) -> List[RawFareQuote]:
        """Parse JSON response structure for multi-carrier flights."""
        quotes: List[RawFareQuote] = []
        flights = []
        if isinstance(json_data, dict):
            for key in ["flights", "trips", "itineraries", "journeys", "results", "data"]:
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

        for idx, fl in enumerate(flights[:15]):
            try:
                # Detect carrier code and flight number
                c_code = fl.get("carrierCode") or fl.get("carrier") or fl.get("airlineCode")
                f_no = str(fl.get("flightNumber") or fl.get("flightNo") or fl.get("flight_number") or "")
                
                if not c_code:
                    fn_match = re.search(r'\b(6E|AI|IX|QP|SG)[ -]?(\d{3,4})\b', f_no, re.IGNORECASE)
                    if fn_match:
                        c_code = fn_match.group(1).upper()
                        f_no = f"{c_code}-{fn_match.group(2)}"
                    else:
                        c_code = "6E"
                        f_no = f"6E-{1000+idx}"
                else:
                    c_code = str(c_code).upper()
                    if not f_no.startswith(c_code):
                        f_no = f"{c_code}-{f_no}"

                dep_raw = str(fl.get("departureTime") or fl.get("depTime") or "08:00")
                arr_raw = str(fl.get("arrivalTime") or fl.get("arrTime") or "10:15")
                dep_time = self._parse_time_str(dep_raw)
                arr_time = self._parse_time_str(arr_raw)

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
                    carrier_code=c_code,
                    flight_number=f_no,
                    departure_time=dep_time,
                    arrival_time=arr_time,
                    total_fare=total_fare,
                    convenience_fee=Decimal(str(self.convenience_fee_standard))
                )
                quotes.append(quote)
            except Exception as e:
                logger.debug(f"Failed to parse OTA JSON flight item {fl}: {e}")
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
        convenience_fee: Decimal = Decimal("0.00")
    ) -> List[RawFareQuote]:
        """Resilient DOM parser for OTA aggregator flight search results."""
        soup = BeautifulSoup(html_text, "html.parser")
        quotes: List[RawFareQuote] = []

        card_selectors = [
            ".listingCard", ".fli-list", ".flight-card", ".itinerary-card",
            ".flight-item", ".flt-opt", "div[id^='divFlight']", ".flt-row",
            ".c-flight-listing-split-row", "div[data-component='flightCard']",
            "div[class*='FlightCard']", "div[class*='flight-row']"
        ]
        cards = []
        for sel in card_selectors:
            found = soup.select(sel)
            if found:
                cards = found
                break

        for card in cards[:15]:
            text = card.get_text(" ", strip=True)

            fn_match = re.search(r'\b(6E|AI|IX|QP|SG)[ -]?(\d{3,4})\b', text, re.IGNORECASE)
            if fn_match:
                c_code = fn_match.group(1).upper()
                f_no = f"{c_code}-{fn_match.group(2)}"
            else:
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


class MakeMyTripScraper(OTAScraper):
    """
    MakeMyTrip (MMT) Scraper.
    India's benchmark travel aggregator (~50% OTA market share).
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="makemytrip",
            platform_code="MMT",
            domain="makemytrip.com",
            platform_full_name="MakeMyTrip",
            search_endpoint_template="https://www.makemytrip.com/flight/search?itinerary={origin}-{destination}-{date_compact}&tripType=O&paxType=A-1_C-0_I-0",
            convenience_fee_standard=350.0,
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.makemytrip.com/flights/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


class CleartripScraper(OTAScraper):
    """
    Cleartrip (CT) Scraper.
    Flipkart-backed OTA known for clean fare breakdowns.
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="cleartrip",
            platform_code="CT",
            domain="cleartrip.com",
            platform_full_name="Cleartrip",
            search_endpoint_template="https://www.cleartrip.com/flights/results?from={origin}&to={destination}&depart_date={date_slash}&adults=1",
            convenience_fee_standard=300.0,
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.cleartrip.com/flights",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


class IxigoScraper(OTAScraper):
    """
    Ixigo (IXI) Scraper.
    Leading meta-search OTA aggregator.
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="ixigo",
            platform_code="IXI",
            domain="ixigo.com",
            platform_full_name="Ixigo",
            search_endpoint_template="https://www.ixigo.com/search/result/flight/{origin}/{destination}/{travel_date}/1/0/0/e/0",
            convenience_fee_standard=250.0,
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.ixigo.com/flights",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


class EaseMyTripScraper(OTAScraper):
    """
    EaseMyTrip (EMT) Scraper.
    Budget OTA featuring statutory zero convenience fee policy.
    """
    def __init__(self, min_delay_seconds: float = 5.0):
        super().__init__(
            source_name="easemytrip",
            platform_code="EMT",
            domain="easemytrip.com",
            platform_full_name="EaseMyTrip",
            search_endpoint_template="https://flight.easemytrip.com/FlightList/Index?srch={origin}-{destination}-{date_slash}-1-0-0-E-D",
            convenience_fee_standard=0.0,  # Zero convenience fee model
            min_delay_seconds=min_delay_seconds
        )

    def _get_request_headers(self) -> Dict[str, str]:
        return {
            "Referer": "https://www.easemytrip.com/flights.html",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
        }


OTA_SCRAPERS: Dict[str, Type[OTAScraper]] = {
    "MMT": MakeMyTripScraper,
    "CT": CleartripScraper,
    "IXI": IxigoScraper,
    "EMT": EaseMyTripScraper,
    "MAKEMYTRIP": MakeMyTripScraper,
    "CLEARTRIP": CleartripScraper,
    "IXIGO": IxigoScraper,
    "EASEMYTRIP": EaseMyTripScraper
}


def get_ota_scraper(platform_code: str, min_delay_seconds: float = 5.0) -> OTAScraper:
    """Factory function to instantiate OTA scraper by platform code or name."""
    code = platform_code.upper().strip()
    if code not in OTA_SCRAPERS:
        raise ValueError(f"Unsupported OTA platform code: {code}. Must be one of {list(OTA_SCRAPERS.keys())}")
    return OTA_SCRAPERS[code](min_delay_seconds=min_delay_seconds)

