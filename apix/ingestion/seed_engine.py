"""
Deterministic Seed Playback and Synthesis Engine for APIx
Provides 100% offline, reproducible, statistically valid fare quotes across
all 30 DGCA sectors, 5 airlines, 4 OTAs, and 5 advance-purchase booking windows.
"""

from datetime import date, datetime, time, timedelta
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple
import uuid

from apix.ingestion.base import (
    RawFareQuote,
    APPROVED_SECTORS,
    APPROVED_CARRIERS,
    APPROVED_SOURCES
)

logger = logging.getLogger("apix.ingestion.seed_engine")

# Default paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DEFAULT_WEIGHTS_PATH = os.path.join(DATA_DIR, "dgca_weights.json")
DEFAULT_BOOKING_PATH = os.path.join(DATA_DIR, "booking_distribution.json")
DEFAULT_SEED_QUOTES_PATH = os.path.join(DATA_DIR, "seed_quotes.json")

# Statutory Airport Charges (AERA approved UDF in INR)
AIRPORT_UDF_MAP: Dict[str, float] = {
    "DEL": 450.0,
    "BOM": 380.0,
    "BLR": 450.0,
    "HYD": 410.0,
    "CCU": 390.0,
    "PNQ": 360.0,
    "AMD": 350.0,
    "MAA": 370.0,
    "JAI": 390.0,
    "GOI": 420.0,
    "LKO": 350.0,
    "SXR": 340.0,
    "GAU": 350.0
}
DEFAULT_UDF = 400.0

# Passenger Service Fee (statutory BCAS/MoCA security fee in INR)
STATUTORY_PSF = 180.0

# GST statutory rate on domestic economy class
GST_RATE_ECONOMY = 0.05

# Carrier yield multipliers and brand profiles
CARRIER_PROFILES: Dict[str, Dict[str, Any]] = {
    "6E": {
        "name": "IndiGo",
        "multiplier": 1.00,
        "type": "LCC",
        "flight_prefix": "6E",
        "flight_numbers": [2041, 5012, 6105, 108, 334, 782, 915, 442]
    },
    "AI": {
        "name": "Air India",
        "multiplier": 1.18,
        "type": "FSC",
        "flight_prefix": "AI",
        "flight_numbers": [887, 665, 504, 102, 440, 701, 312, 998]
    },
    "IX": {
        "name": "Air India Express",
        "multiplier": 0.95,
        "type": "LCC",
        "flight_prefix": "IX",
        "flight_numbers": [1142, 245, 983, 712, 550, 318, 804]
    },
    "QP": {
        "name": "Akasa Air",
        "multiplier": 0.96,
        "type": "LCC",
        "flight_prefix": "QP",
        "flight_numbers": [1321, 1102, 1405, 1204, 1510, 1632]
    },
    "SG": {
        "name": "SpiceJet",
        "multiplier": 0.94,
        "type": "LCC",
        "flight_prefix": "SG",
        "flight_numbers": [8169, 136, 291, 403, 524, 710]
    }
}

# Source convenience fees (INR)
SOURCE_CONVENIENCE_FEES: Dict[str, float] = {
    "indigo": 0.0,
    "airindia": 0.0,
    "airindiaexpress": 0.0,
    "akasa": 0.0,
    "spicejet": 0.0,
    "makemytrip": 350.0,
    "cleartrip": 300.0,
    "ixigo": 250.0,
    "easemytrip": 0.0  # EMT baseline zero convenience fee policy
}

# Lead-time yield curve multipliers
LEAD_TIME_MULTIPLIERS: Dict[int, float] = {
    1: 2.85,
    7: 1.85,
    15: 1.35,
    30: 1.08,
    45: 1.00
}

# Sold-out flight probability per lead window
SOLD_OUT_PROBABILITIES: Dict[int, float] = {
    1: 0.12,
    7: 0.05,
    15: 0.02,
    30: 0.01,
    45: 0.005
}

# Sector distances and approximate flight durations fallback
DEFAULT_SECTOR_SPECS: Dict[str, Tuple[int, int]] = {
    "DEL-BOM": (1148, 130), "BOM-DEL": (1148, 130),
    "DEL-BLR": (1740, 165), "BLR-DEL": (1740, 165),
    "BOM-BLR": (842, 105),  "BLR-BOM": (842, 105),
    "DEL-HYD": (1253, 135), "HYD-DEL": (1253, 135),
    "DEL-CCU": (1305, 135), "CCU-DEL": (1305, 135),
    "HYD-BOM": (617, 90),   "BOM-HYD": (617, 90),
    "PNQ-DEL": (1173, 130), "DEL-PNQ": (1173, 130),
    "AMD-DEL": (775, 95),   "DEL-AMD": (775, 95),
    "HYD-BLR": (500, 75),   "BLR-HYD": (500, 75),
    "MAA-DEL": (1760, 170), "DEL-MAA": (1760, 170),
    "DEL-JAI": (241, 55),   "JAI-DEL": (241, 55),
    "BOM-GOI": (435, 75),   "GOI-BOM": (435, 75),
    "DEL-LKO": (418, 70),   "LKO-DEL": (418, 70),
    "DEL-SXR": (650, 90),   "SXR-DEL": (650, 90),
    "CCU-GAU": (510, 75),   "GAU-CCU": (510, 75)
}


class SeedPlaybackEngine:
    """
    Deterministic offline playback and quote generation engine.
    Ensures zero external network dependencies for CI/CD, mathematical unit tests,
    and end-to-end integration validation.
    """
    def __init__(
        self,
        seed: int = 42,
        weights_path: str = DEFAULT_WEIGHTS_PATH,
        seed_quotes_path: str = DEFAULT_SEED_QUOTES_PATH
    ):
        self.seed = seed
        self.weights_path = weights_path
        self.seed_quotes_path = seed_quotes_path
        self._rng = random.Random(seed)
        self._sector_specs: Dict[str, Tuple[int, int]] = {}
        self._cached_quotes: List[RawFareQuote] = []
        self._indexed_cache: Dict[str, List[RawFareQuote]] = {}
        
        self._load_sector_specifications()

    def set_seed(self, seed: int) -> None:
        """Reset pseudo-random generator state with given seed."""
        self.seed = seed
        self._rng = random.Random(seed)

    def _load_sector_specifications(self) -> None:
        """Load sector distances and durations from dgca_weights.json if available."""
        if os.path.exists(self.weights_path):
            try:
                with open(self.weights_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                sectors = data.get("sectors", {})
                for sec, info in sectors.items():
                    dist = int(info.get("distance_km", 1000))
                    duration = int(info.get("approx_flight_minutes", 120))
                    self._sector_specs[sec] = (dist, duration)
                logger.debug(f"Loaded {len(self._sector_specs)} sector specs from {self.weights_path}")
                return
            except Exception as e:
                logger.warning(f"Could not load weights from {self.weights_path}: {e}")

        # Fallback to defaults
        self._sector_specs = dict(DEFAULT_SECTOR_SPECS)

    def get_sector_distance_and_duration(self, origin: str, destination: str) -> Tuple[int, int]:
        """Lookup distance (km) and approx duration (min) for a sector."""
        sector_key = f"{origin.upper().strip()}-{destination.upper().strip()}"
        if sector_key in self._sector_specs:
            return self._sector_specs[sector_key]
        reverse_key = f"{destination.upper().strip()}-{origin.upper().strip()}"
        if reverse_key in self._sector_specs:
            return self._sector_specs[reverse_key]
        return (1000, 120)

    def _calculate_fuel_surcharge(self, distance_km: int) -> Decimal:
        """Calculate standard Indian domestic fuel surcharge slab."""
        if distance_km < 500:
            return Decimal("600.00")
        elif distance_km < 1000:
            return Decimal("800.00")
        elif distance_km < 1500:
            return Decimal("1000.00")
        else:
            return Decimal("1200.00")

    def _get_day_of_week_factor(self, travel_date: date) -> float:
        """
        Return day of week demand factor:
        Friday (4) & Sunday (6): 1.15
        Tuesday (1) & Wednesday (2): 0.92
        Mon (0), Thu (3), Sat (5): 1.00
        """
        dow = travel_date.weekday()
        if dow in (4, 6):
            return 1.15
        elif dow in (1, 2):
            return 0.92
        else:
            return 1.00

    def _get_lead_time_factor(self, advance_days: int) -> float:
        """Return lead-time multiplier for advance window."""
        if advance_days in LEAD_TIME_MULTIPLIERS:
            return LEAD_TIME_MULTIPLIERS[advance_days]
        # Interpolate or snap
        if advance_days <= 1:
            return 2.85
        elif advance_days <= 7:
            return 1.85
        elif advance_days <= 15:
            return 1.35
        elif advance_days <= 30:
            return 1.08
        else:
            return 1.00

    def synthesize_quote(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int,
        carrier_code: str,
        source: str,
        flight_idx: int = 0,
        departure_hour: Optional[int] = None
    ) -> RawFareQuote:
        """
        Synthesize a mathematically consistent, realistic RawFareQuote.
        Decomposition strictly satisfies:
        total_fare == base_fare + fuel_surcharge + gst + udf + psf + convenience_fee
        """
        origin = origin.upper().strip()
        destination = destination.upper().strip()
        carrier_code = carrier_code.upper().strip()
        source = source.lower().strip()
        sector_key = f"{origin}-{destination}"

        dist_km, base_duration = self.get_sector_distance_and_duration(origin, destination)
        
        # Flight schedule
        carrier_info = CARRIER_PROFILES.get(carrier_code, CARRIER_PROFILES["6E"])
        fn_list = carrier_info["flight_numbers"]
        fn_num = fn_list[flight_idx % len(fn_list)]
        flight_number = f"{carrier_code}-{fn_num}"

        if departure_hour is None:
            dep_hours = [6, 8, 11, 14, 17, 20]
            dep_hour = dep_hours[flight_idx % len(dep_hours)]
        else:
            dep_hour = departure_hour

        dep_min = (flight_idx * 15 + 10) % 60
        dep_time = time(dep_hour, dep_min)
        duration_mins = base_duration + (flight_idx % 3) * 5
        dep_dt = datetime.combine(travel_date, dep_time)
        arr_dt = dep_dt + timedelta(minutes=duration_mins)
        arr_time = arr_dt.time()

        # Deterministic pseudo-random variation based on flight & travel date
        hash_seed = int(hashlib.md5(f"{sector_key}_{flight_number}_{travel_date}_{source}_{self.seed}".encode()).hexdigest()[:8], 16)
        local_rng = random.Random(hash_seed)

        # Capacity exhaustion / sold out
        sold_out_p = SOLD_OUT_PROBABILITIES.get(advance_days, 0.02)
        is_sold_out = local_rng.random() < sold_out_p
        seats_available = 0 if is_sold_out else (local_rng.randint(1, 8) if local_rng.random() < 0.35 else None)

        # Yield formula components
        base_tariff = 2200.0 + 1.85 * dist_km
        carrier_mult = carrier_info["multiplier"]
        lead_mult = self._get_lead_time_factor(advance_days)
        dow_mult = self._get_day_of_week_factor(travel_date)
        stochastic_noise = 1.0 + (local_rng.uniform(-0.04, 0.04))

        raw_base_fare_float = (base_tariff * carrier_mult * lead_mult * dow_mult * stochastic_noise)
        raw_base_fare = Decimal(str(round(raw_base_fare_float, 2))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Fuel surcharge
        fuel_surcharge = self._calculate_fuel_surcharge(dist_km)

        # Statutory UDF and PSF
        udf = Decimal(str(AIRPORT_UDF_MAP.get(origin, DEFAULT_UDF))).quantize(Decimal("0.01"))
        psf = Decimal(str(STATUTORY_PSF)).quantize(Decimal("0.01"))

        # GST 5% on (Base + Fuel)
        gst = ((raw_base_fare + fuel_surcharge) * Decimal(str(GST_RATE_ECONOMY))).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)

        # Convenience fee
        source_type = "airline" if source in ("indigo", "airindia", "airindiaexpress", "akasa", "spicejet") else "ota"
        convenience_fee_float = SOURCE_CONVENIENCE_FEES.get(source, 0.0)
        convenience_fee = Decimal(str(convenience_fee_float)).quantize(Decimal("0.01"))

        # Exact total fare arithmetic reconciliation
        total_fare = raw_base_fare + fuel_surcharge + gst + udf + psf + convenience_fee

        # Scrape timestamp
        scrape_dt = datetime.combine(travel_date - timedelta(days=advance_days), time(6, 0))

        # Unique deterministic quote UUID
        quote_uuid = str(uuid.UUID(int=hash_seed))

        return RawFareQuote(
            quote_id=quote_uuid,
            source=source,
            source_type=source_type,
            scrape_timestamp=scrape_dt,
            origin_iata=origin,
            destination_iata=destination,
            carrier_code=carrier_code,
            flight_number=flight_number,
            departure_time=dep_time,
            arrival_time=arr_time,
            travel_date=travel_date,
            advance_days=advance_days,
            duration_minutes=duration_mins,
            stops=0,
            fare_class="economy",
            currency="INR",
            raw_base_fare=raw_base_fare,
            raw_fuel_surcharge=fuel_surcharge,
            raw_taxes=gst,
            raw_udf=udf,
            raw_psf=psf,
            raw_convenience_fee=convenience_fee,
            raw_total_fare=total_fare,
            seats_available=seats_available,
            is_sold_out=is_sold_out
        )

    def generate_quotes_for_sector(
        self,
        origin: str,
        destination: str,
        travel_date: date,
        advance_days: int,
        carrier_code: Optional[str] = None,
        source: Optional[str] = None,
        quotes_per_carrier: int = 1
    ) -> List[RawFareQuote]:
        """
        Generate quotes for a specific sector, date, and advance window.
        If carrier_code or source is omitted, samples across all approved entities.
        """
        origin = origin.upper().strip()
        destination = destination.upper().strip()
        quotes: List[RawFareQuote] = []

        target_carriers = [carrier_code.upper().strip()] if carrier_code else list(CARRIER_PROFILES.keys())

        for c_code in target_carriers:
            carrier_src = {
                "6E": "indigo", "AI": "airindia", "IX": "airindiaexpress",
                "QP": "akasa", "SG": "spicejet"
            }[c_code]
            # Direct carrier quote
            if source is None or source == carrier_src or source == CARRIER_PROFILES[c_code]["name"].lower().replace(" ", ""):
                for f_idx in range(quotes_per_carrier):
                    q = self.synthesize_quote(
                        origin=origin,
                        destination=destination,
                        travel_date=travel_date,
                        advance_days=advance_days,
                        carrier_code=c_code,
                        source=carrier_src,
                        flight_idx=f_idx
                    )
                    quotes.append(q)

            # OTA aggregator quotes
            ota_sources = ["makemytrip", "cleartrip", "ixigo", "easemytrip"]
            target_otas = [source.lower().strip()] if (source and source in ota_sources) else (ota_sources if source is None else [])
            for ota in target_otas:
                for f_idx in range(quotes_per_carrier):
                    q = self.synthesize_quote(
                        origin=origin,
                        destination=destination,
                        travel_date=travel_date,
                        advance_days=advance_days,
                        carrier_code=c_code,
                        source=ota,
                        flight_idx=f_idx
                    )
                    quotes.append(q)

        return quotes

    def generate_full_basket(
        self,
        base_scrape_date: Optional[date] = None,
        quotes_per_cohort: int = 1
    ) -> List[RawFareQuote]:
        """
        Synthesize comprehensive quote dataset covering:
        - All 30 DGCA sectors
        - All 5 advance windows: T+1, T+7, T+15, T+30, T+45
        - All 5 airlines + 4 OTAs
        """
        if base_scrape_date is None:
            base_scrape_date = date.today()

        advance_windows = [1, 7, 15, 30, 45]
        all_quotes: List[RawFareQuote] = []

        sectors = sorted(list(APPROVED_SECTORS))
        carriers = ["6E", "AI", "IX", "QP", "SG"]
        otas = ["makemytrip", "cleartrip", "ixigo", "easemytrip"]

        for sector in sectors:
            orig, dest = sector.split("-")
            for adv in advance_windows:
                travel_date = base_scrape_date + timedelta(days=adv)

                # Generate direct airline quotes
                for c_code in carriers:
                    carrier_src = {
                        "6E": "indigo", "AI": "airindia", "IX": "airindiaexpress",
                        "QP": "akasa", "SG": "spicejet"
                    }[c_code]
                    for idx in range(quotes_per_cohort):
                        q = self.synthesize_quote(
                            origin=orig,
                            destination=dest,
                            travel_date=travel_date,
                            advance_days=adv,
                            carrier_code=c_code,
                            source=carrier_src,
                            flight_idx=idx
                        )
                        all_quotes.append(q)

                # Generate OTA quotes
                for ota in otas:
                    # Cover all 5 monitored domestic carriers (6E, AI, IX, QP, SG)
                    # to enable cross-source consensus and OTA aggregator parity
                    for c_code in carriers:
                        for idx in range(quotes_per_cohort):
                            q = self.synthesize_quote(
                                origin=orig,
                                destination=dest,
                                travel_date=travel_date,
                                advance_days=adv,
                                carrier_code=c_code,
                                source=ota,
                                flight_idx=idx
                            )
                            all_quotes.append(q)

        logger.info(f"Synthesized full basket of {len(all_quotes)} quotes across 30 sectors and 5 windows.")
        return all_quotes

    def save_seed_dataset(
        self,
        output_path: Optional[str] = None,
        quotes: Optional[List[RawFareQuote]] = None
    ) -> str:
        """Save quotes to JSON seed dataset file."""
        if output_path is None:
            output_path = self.seed_quotes_path
        if quotes is None:
            quotes = self.generate_full_basket()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        serialized = [q.to_dict() for q in quotes]
        
        # Serialize Decimals, Dates, and Times to strings
        def json_encoder(obj: Any) -> Any:
            if isinstance(obj, (date, datetime, time)):
                return obj.isoformat()
            if isinstance(obj, Decimal):
                return str(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serialized, f, default=json_encoder, indent=2)

        logger.info(f"Successfully saved {len(quotes)} seed quotes to {output_path}")
        return output_path

    def load_seed_dataset(self, filepath: Optional[str] = None) -> List[RawFareQuote]:
        """Load and index cached seed quotes from JSON file."""
        target_path = filepath or self.seed_quotes_path
        if not os.path.exists(target_path):
            logger.warning(f"Seed file {target_path} not found. Generating fresh dataset...")
            quotes = self.generate_full_basket()
            self.save_seed_dataset(target_path, quotes)
            self._cached_quotes = quotes
        else:
            with open(target_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._cached_quotes = [RawFareQuote(**item) for item in data]
            logger.info(f"Loaded {len(self._cached_quotes)} cached quotes from {target_path}")

        # Index by sector and window for fast lookup
        self._indexed_cache.clear()
        for q in self._cached_quotes:
            key = f"{q.origin_iata}-{q.destination_iata}:{q.advance_days}"
            if key not in self._indexed_cache:
                self._indexed_cache[key] = []
            self._indexed_cache[key].append(q)

        return self._cached_quotes

    def query_quotes(
        self,
        origin: str,
        destination: str,
        travel_date: Optional[date] = None,
        advance_days: Optional[int] = None,
        carrier_code: Optional[str] = None,
        source: Optional[str] = None
    ) -> List[RawFareQuote]:
        """
        Query quotes for given parameters. Falls back to synthesis if not in cache.
        """
        origin = origin.upper().strip()
        destination = destination.upper().strip()
        sector_key = f"{origin}-{destination}"

        if not self._cached_quotes and os.path.exists(self.seed_quotes_path):
            self.load_seed_dataset()

        # If cached, filter
        if self._cached_quotes:
            results = []
            for q in self._cached_quotes:
                if q.origin_iata == origin and q.destination_iata == destination:
                    if advance_days is not None and q.advance_days != advance_days:
                        continue
                    if travel_date is not None and q.travel_date != travel_date:
                        continue
                    if carrier_code is not None and q.carrier_code != carrier_code.upper().strip():
                        continue
                    if source is not None and q.source != source.lower().strip():
                        continue
                    results.append(q)
            if results:
                return results

        # Fallback to deterministic synthesis
        t_date = travel_date or (date.today() + timedelta(days=advance_days or 15))
        a_days = advance_days if advance_days is not None else (t_date - date.today()).days
        return self.generate_quotes_for_sector(
            origin=origin,
            destination=destination,
            travel_date=t_date,
            advance_days=a_days,
            carrier_code=carrier_code,
            source=source
        )
