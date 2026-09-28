"""
Strict Pydantic Data Models and Schemas for APIx Pipeline (Module B).
Defines schemas for RawFareQuote, DecomposedFare, CleanedFareQuote,
and related metadata/scoring structures.
"""

from datetime import date, datetime, time
from decimal import Decimal
from enum import Enum
import re
from typing import Any, Dict, List, Optional, Union
import uuid

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ==============================================================================
# ENUMS & CONSTANTS
# ==============================================================================

class CarrierCode(str, Enum):
    """Monitored DGCA domestic carriers."""
    INDIGO = "6E"
    AIR_INDIA = "AI"
    AIR_INDIA_EXPRESS = "IX"
    AKASA_AIR = "QP"
    SPICEJET = "SG"


class SourceType(str, Enum):
    """Origin source category."""
    AIRLINE = "airline"
    OTA = "ota"


class FareClass(str, Enum):
    """Cabin booking class."""
    ECONOMY = "economy"
    PREMIUM_ECONOMY = "premium_economy"
    BUSINESS = "business"


class RouteTier(str, Enum):
    """DGCA route network tier."""
    TRUNK = "Trunk"
    HIGH_DENSITY = "High-Density"
    REGIONAL = "Regional"
    SEASONAL = "Seasonal"


class OutlierReason(str, Enum):
    """Outlier flagging codes."""
    NONE = "NONE"
    DOMAIN_FLOOR = "DOMAIN_FLOOR"
    DOMAIN_CEILING = "DOMAIN_CEILING"
    STATISTICAL_IQR = "STATISTICAL_IQR"
    TEMPORAL_SURGE = "TEMPORAL_SURGE"
    CROSS_SOURCE_VARIANCE = "CROSS_SOURCE_VARIANCE"
    NON_BASKET_CONNECTING = "NON_BASKET_CONNECTING"


# ==============================================================================
# 1. RAW FARE QUOTE SCHEMA
# ==============================================================================

class RawFareQuote(BaseModel):
    """
    Ingested raw fare quote before cleaning, normalization, and statutory decomposition.
    Accepts direct scraper outputs as well as seed engine playback quotes.
    """
    model_config = ConfigDict(populate_by_name=True, extra="ignore")

    quote_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique quote identifier")
    source: str = Field(..., description="Source identifier e.g. indigo, makemytrip")
    source_type: str = Field(default="airline", description="'airline' or 'ota'")
    scrape_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(tz=None),
        description="Timestamp when the fare was scraped or played back"
    )
    origin_iata: str = Field(..., min_length=3, max_length=3, description="3-letter IATA origin airport code")
    destination_iata: str = Field(..., min_length=3, max_length=3, description="3-letter IATA destination airport code")
    carrier_code: str = Field(..., min_length=2, max_length=3, description="IATA 2-letter carrier code (e.g. 6E)")
    flight_number: str = Field(..., description="Carrier flight number (e.g. 6E-2041)")
    departure_time: time = Field(..., description="Scheduled flight departure time")
    arrival_time: time = Field(..., description="Scheduled flight arrival time")
    travel_date: date = Field(..., description="Scheduled flight travel date")
    advance_days: int = Field(..., ge=0, description="Advance booking horizon in days (travel_date - scrape_date)")
    duration_minutes: int = Field(..., gt=0, description="Scheduled flight duration in minutes")
    stops: int = Field(default=0, ge=0, description="Number of intermediate stops (0 for direct)")
    fare_class: str = Field(default="economy", description="Booking cabin class (economy, business, etc.)")
    currency: str = Field(default="INR", description="ISO-4217 currency code")

    raw_base_fare: Optional[Decimal] = Field(default=None, description="Unverified base fare if scraped")
    raw_fuel_surcharge: Optional[Decimal] = Field(default=None, description="Unverified fuel surcharge if scraped")
    raw_taxes: Optional[Decimal] = Field(default=None, description="Unverified taxes/GST if scraped")
    raw_udf: Optional[Decimal] = Field(default=None, description="Unverified User Development Fee if scraped")
    raw_psf: Optional[Decimal] = Field(default=None, description="Unverified Passenger Service Fee if scraped")
    raw_convenience_fee: Optional[Decimal] = Field(default=None, description="Unverified convenience fee if scraped")
    raw_total_fare: Decimal = Field(..., description="Total gross airfare charged to passenger")

    seats_available: Optional[int] = Field(default=None, ge=0, description="Remaining seats exposed by inventory")
    is_sold_out: bool = Field(default=False, description="True if no seats available or flight capacity exhausted")

    @model_validator(mode="before")
    @classmethod
    def remap_and_normalize_inputs(cls, data: Any) -> Any:
        """Remap aliases and alternative naming conventions across scrapers and test fixtures."""
        if isinstance(data, dict):
            # Origin & destination aliases
            if "origin" in data and "origin_iata" not in data:
                data["origin_iata"] = data["origin"]
            if "destination" in data and "destination_iata" not in data:
                data["destination_iata"] = data["destination"]

            # Carrier alias
            if "carrier" in data and "carrier_code" not in data:
                data["carrier_code"] = data["carrier"]

            # Fare aliases
            if "total_fare" in data and "raw_total_fare" not in data:
                data["raw_total_fare"] = data["total_fare"]
            if "base_fare" in data and "raw_base_fare" not in data:
                data["raw_base_fare"] = data["base_fare"]
            if "taxes" in data and "raw_taxes" not in data:
                data["raw_taxes"] = data["taxes"]
            if "udf" in data and "raw_udf" not in data:
                data["raw_udf"] = data["udf"]
            if "convenience_fee" in data and "raw_convenience_fee" not in data:
                data["raw_convenience_fee"] = data["convenience_fee"]

            # Date/timestamp aliases
            if "scrape_date" in data and "scrape_timestamp" not in data:
                data["scrape_timestamp"] = data["scrape_date"]
            if "search_datetime" in data and "scrape_timestamp" not in data:
                data["scrape_timestamp"] = data["search_datetime"]

            if "departure_datetime" in data:
                dep_dt = data["departure_datetime"]
                if isinstance(dep_dt, datetime):
                    if "departure_time" not in data:
                        data["departure_time"] = dep_dt.time()
                    if "travel_date" not in data:
                        data["travel_date"] = dep_dt.date()

            # Ensure uppercase string codes
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
            raise ValueError(f"Invalid IATA airport code: {v}. Must be exactly 3 uppercase letters.")
        return v

    @model_validator(mode="after")
    def validate_distinct_airports(self) -> "RawFareQuote":
        if self.origin_iata == self.destination_iata:
            raise ValueError(f"Origin and destination cannot be identical: {self.origin_iata}")
        return self

    @property
    def origin(self) -> str:
        return self.origin_iata

    @property
    def destination(self) -> str:
        return self.destination_iata

    @property
    def carrier(self) -> str:
        return self.carrier_code

    @property
    def total_fare(self) -> float:
        return float(self.raw_total_fare)


# ==============================================================================
# 2. DECOMPOSED FARE SCHEMA
# ==============================================================================

class DecomposedFare(BaseModel):
    """
    Itemized 6-component statutory fare breakdown for Indian domestic air travel.
    Total = Base Fare + Fuel Surcharge + GST (5%) + UDF + PSF + Convenience Fee
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    base_fare: Decimal = Field(..., description="Airline net commercial base tariff (INR)")
    fuel_surcharge: Decimal = Field(..., description="Airline fuel surcharge / YQ / YR (INR)")
    airport_taxes_gst: Decimal = Field(..., description="Statutory 5% GST on (Base + Fuel) for economy (INR)")
    user_development_fee: Decimal = Field(
        ...,
        alias="user_dev_fee",
        description="AERA approved User Development Fee for origin airport (INR)"
    )
    passenger_service_fee: Decimal = Field(
        ...,
        description="Statutory Passenger Service / Security Fee (INR)"
    )
    convenience_fee: Decimal = Field(
        default=Decimal("0.00"),
        description="OTA or booking channel transaction convenience fee (INR)"
    )
    total_fare: Decimal = Field(..., description="Total fare charged to passenger (INR)")

    @model_validator(mode="before")
    @classmethod
    def align_fee_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "user_dev_fee" in data and "user_development_fee" not in data:
                data["user_development_fee"] = data["user_dev_fee"]
            elif "user_development_fee" in data and "user_dev_fee" not in data:
                data["user_dev_fee"] = data["user_development_fee"]
            if "psf" in data and "passenger_service_fee" not in data:
                data["passenger_service_fee"] = data["psf"]
        return data

    @property
    def user_dev_fee(self) -> float:
        return float(self.user_development_fee)

    @property
    def reconstructed_total(self) -> Decimal:
        """Sum of all 6 individual statutory fare components."""
        return (
            self.base_fare
            + self.fuel_surcharge
            + self.airport_taxes_gst
            + self.user_development_fee
            + self.passenger_service_fee
            + self.convenience_fee
        )

    @property
    def reconciliation_difference(self) -> Decimal:
        """Arithmetic residual between Total Fare and reconstructed sum."""
        return abs(self.total_fare - self.reconstructed_total)

    def is_balanced(self, tolerance: float = 1.00) -> bool:
        """Returns True if the 6-component arithmetic identity holds within tolerance."""
        return float(self.reconciliation_difference) <= tolerance


# ==============================================================================
# 3. CLEANED FARE QUOTE SCHEMA
# ==============================================================================

class CleanedFareQuote(BaseModel):
    """
    Cleaned, normalized, verified, and scored fare quote ready for analytical storage
    and econometric index compilation (Module C).
    """
    model_config = ConfigDict(populate_by_name=True, extra="allow")

    quote_id: str = Field(..., description="Unique quote identifier")
    route_id: Optional[int] = Field(default=None, description="Optional foreign key to routes table")
    carrier_id: Optional[int] = Field(default=None, description="Optional foreign key to carriers table")
    sector: str = Field(..., description="Bidirectional route code e.g. 'DEL-BOM'")
    origin_iata: str = Field(..., min_length=3, max_length=3, description="3-letter IATA origin")
    destination_iata: str = Field(..., min_length=3, max_length=3, description="3-letter IATA destination")
    carrier_code: str = Field(..., min_length=2, max_length=3, description="2-letter IATA carrier code")
    source: str = Field(..., description="Source portal e.g. indigo, makemytrip")
    source_id: Optional[str] = Field(default=None, description="Optional source reference string")
    source_type: str = Field(default="airline", description="'airline' or 'ota'")
    scrape_timestamp: datetime = Field(default_factory=datetime.utcnow, description="UTC scrape timestamp")
    travel_date: date = Field(..., description="Date of travel")
    advance_days: int = Field(..., ge=0, description="Advance booking horizon (days)")
    flight_number: str = Field(..., description="Carrier flight number (e.g. 6E-2041)")
    departure_time: time = Field(default=time(8, 0), description="Scheduled departure time")
    arrival_time: time = Field(default=time(10, 30), description="Scheduled arrival time")
    duration_minutes: int = Field(default=120, gt=0, description="Duration in minutes")
    stops: int = Field(default=0, ge=0, description="Intermediate stops (0 = direct)")
    fare_class: str = Field(default="economy", description="Booking class (economy)")
    currency: str = Field(default="INR", description="Currency code")

    # 6 statutory fare components
    base_fare: Decimal = Field(..., description="Decomposed base fare (INR)")
    fuel_surcharge: Decimal = Field(..., description="Decomposed fuel surcharge (INR)")
    airport_taxes_gst: Decimal = Field(..., description="Decomposed GST (INR)")
    user_development_fee: Decimal = Field(
        ...,
        alias="user_dev_fee",
        description="Decomposed User Development Fee (INR)"
    )
    passenger_service_fee: Decimal = Field(..., description="Decomposed Passenger Service Fee (INR)")
    convenience_fee: Decimal = Field(default=Decimal("0.00"), description="Decomposed convenience fee (INR)")
    total_fare: Decimal = Field(..., description="Verified total airfare (INR)")

    # Nested decomposed fare model for interface interoperability
    decomposed_fare: Optional[DecomposedFare] = Field(
        default=None,
        description="Embedded DecomposedFare model for contract compatibility"
    )

    seats_available: Optional[int] = Field(default=None, ge=0, description="Seats remaining")
    is_sold_out: bool = Field(default=False, description="True if capacity is exhausted")
    is_outlier: bool = Field(default=False, description="True if flagged by statistical IQR or domain filters")
    outlier_reason: Optional[str] = Field(default=None, description="Reason code if flagged as outlier")
    data_quality_score: float = Field(..., ge=0.0, le=100.0, description="Composite Data Quality Score (0-100)")
    is_quarantined: bool = Field(default=False, description="True if DQS < 60 or rejected")

    @model_validator(mode="before")
    @classmethod
    def reconcile_and_unpack(cls, data: Any) -> Any:
        """Unpack nested decomposed_fare if provided, or construct missing decomposed_fare."""
        if isinstance(data, dict):
            # Quality score alias
            if "quality_score" in data and "data_quality_score" not in data:
                data["data_quality_score"] = data["quality_score"]

            # Advance window alias
            if "advance_window" in data and "advance_days" not in data:
                data["advance_days"] = data["advance_window"]

            # Sector & airport derivation
            if "sector" in data and ("origin_iata" not in data or "destination_iata" not in data):
                parts = data["sector"].split("-")
                if len(parts) == 2:
                    if "origin_iata" not in data:
                        data["origin_iata"] = parts[0]
                    if "destination_iata" not in data:
                        data["destination_iata"] = parts[1]
            elif "origin_iata" in data and "destination_iata" in data and "sector" not in data:
                data["sector"] = f"{data['origin_iata']}-{data['destination_iata']}"

            # Carrier code alias
            if "carrier" in data and "carrier_code" not in data:
                data["carrier_code"] = data["carrier"]

            # Unpack decomposed_fare if provided as model or dict
            decomp = data.get("decomposed_fare")
            if decomp is not None:
                if isinstance(decomp, BaseModel):
                    d_dict = decomp.model_dump()
                elif isinstance(decomp, dict):
                    d_dict = decomp
                else:
                    d_dict = {}

                for k, v in d_dict.items():
                    target_k = "user_development_fee" if k == "user_dev_fee" else k
                    if target_k not in data or data[target_k] is None:
                        data[target_k] = v
                if "total_fare" not in data or data["total_fare"] is None:
                    data["total_fare"] = d_dict.get("total_fare")

            # UDF aliases
            if "user_dev_fee" in data and "user_development_fee" not in data:
                data["user_development_fee"] = data["user_dev_fee"]
            elif "user_development_fee" in data and "user_dev_fee" not in data:
                data["user_dev_fee"] = data["user_development_fee"]

        return data

    @model_validator(mode="after")
    def ensure_nested_model(self) -> "CleanedFareQuote":
        """Ensure decomposed_fare nested object is instantiated."""
        if self.decomposed_fare is None:
            self.decomposed_fare = DecomposedFare(
                base_fare=self.base_fare,
                fuel_surcharge=self.fuel_surcharge,
                airport_taxes_gst=self.airport_taxes_gst,
                user_development_fee=self.user_development_fee,
                passenger_service_fee=self.passenger_service_fee,
                convenience_fee=self.convenience_fee,
                total_fare=self.total_fare,
            )
        return self

    @property
    def quality_score(self) -> float:
        return self.data_quality_score

    @property
    def user_dev_fee(self) -> float:
        return float(self.user_development_fee)

    @property
    def advance_window(self) -> int:
        return self.advance_days

    @property
    def carrier(self) -> str:
        return self.carrier_code

    @property
    def origin(self) -> str:
        return self.origin_iata

    @property
    def destination(self) -> str:
        return self.destination_iata

    @property
    def fare_base(self) -> float:
        return float(self.base_fare)

    @property
    def fare_fuel(self) -> float:
        return float(self.fuel_surcharge)

    @property
    def fare_gst(self) -> float:
        return float(self.airport_taxes_gst)

    @property
    def fare_udf(self) -> float:
        return float(self.user_development_fee)

    @property
    def fare_psf(self) -> float:
        return float(self.passenger_service_fee)

    @property
    def fare_fee(self) -> float:
        return float(self.convenience_fee)

    @property
    def dqs_score(self) -> float:
        return float(self.data_quality_score)


# ==============================================================================
# 4. DATA QUALITY SCORE BREAKDOWN
# ==============================================================================

class DQSBreakdown(BaseModel):
    """Component breakdown of the 0-100 Data Quality Score."""
    schema_completeness: float = Field(..., ge=0.0, le=25.0, description="Max 25 pts")
    decomposition_integrity: float = Field(..., ge=0.0, le=20.0, description="Max 20 pts")
    domain_and_bounds: float = Field(..., ge=0.0, le=25.0, description="Max 25 pts")
    cross_source_concordance: float = Field(..., ge=0.0, le=15.0, description="Max 15 pts")
    source_reliability: float = Field(..., ge=0.0, le=15.0, description="Max 15 pts")
    total_score: float = Field(..., ge=0.0, le=100.0, description="Overall DQS (0-100)")
    is_index_eligible: bool = Field(..., description="True if total_score >= 80.0")
    is_quarantined: bool = Field(..., description="True if total_score < 60.0")


# ==============================================================================
# 5. OUTLIER EVALUATION RESULT
# ==============================================================================

class OutlierEvaluation(BaseModel):
    """Detailed diagnostic result of multi-criteria outlier evaluation."""
    is_outlier: bool
    outlier_reason: Optional[str] = None
    flags: List[str] = Field(default_factory=list)
    iqr_lower_bound: Optional[float] = None
    iqr_upper_bound: Optional[float] = None
    temporal_pct_change: Optional[float] = None
    cross_source_deviation_pct: Optional[float] = None
