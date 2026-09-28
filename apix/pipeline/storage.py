"""
Relational Database Storage Layer & Analytical Store for APIx (Module B).
Uses SQLite / SQLAlchemy 2.0 to persist:
1. `routes`: DGCA 15 city-pairs (30 bidirectional sectors), distances, tiers, weights
2. `carriers`: Domestic airlines (6E, AI, IX, QP, SG), market shares, types
3. `fare_quotes`: Verified cleaned fare quotes with 6-component statutory decomposition & DQS
4. `daily_index`: Pre-computed elementary and route-level price index series
5. `atf_prices`: Aviation Turbine Fuel (ATF) historical and simulated prices
6. `cpi_reference`: Official MoSPI/NSO CPI reference series & transport weights

Provides high-performance analytical query functions:
- `insert_quotes(quotes)`
- `get_clean_quotes(start_date, end_date, min_dqs=80.0)`
"""

from datetime import date, datetime, time
from decimal import Decimal
import os
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import (
    Boolean,
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Time,
    and_,
    create_engine,
    event,
    func,
    select,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from apix.pipeline.schemas import CleanedFareQuote, DecomposedFare, RawFareQuote


Base = declarative_base()


# ==============================================================================
# 1. SQLALCHEMY ORM TABLE MODELS
# ==============================================================================

class Route(Base):
    """DGCA Domestic Route & City-Pair Dimension."""
    __tablename__ = "routes"

    route_id = Column(Integer, primary_key=True, autoincrement=True)
    sector = Column(String(16), unique=True, nullable=False, index=True) # e.g. 'DEL-BOM'
    origin_iata = Column(String(3), nullable=False, index=True)
    destination_iata = Column(String(3), nullable=False, index=True)
    origin_city = Column(String(64), nullable=False)
    destination_city = Column(String(64), nullable=False)
    distance_km = Column(Float, nullable=False)
    route_tier = Column(String(32), nullable=False)  # Trunk, High-Density, Regional, Seasonal
    dgca_weight = Column(Float, default=0.0, nullable=False) # Normalized basket weight
    is_active = Column(Boolean, default=True, nullable=False)

    quotes = relationship("FareQuoteRecord", back_populates="route_rel")


class Carrier(Base):
    """Domestic Airline Carrier Dimension."""
    __tablename__ = "carriers"

    carrier_id = Column(Integer, primary_key=True, autoincrement=True)
    iata_code = Column(String(3), unique=True, nullable=False, index=True) # e.g. '6E'
    name = Column(String(64), nullable=False)
    carrier_type = Column(String(16), nullable=False) # LCC or FSC
    dgca_market_share = Column(Float, nullable=False) # National market share
    is_active = Column(Boolean, default=True, nullable=False)

    quotes = relationship("FareQuoteRecord", back_populates="carrier_rel")


class FareQuoteRecord(Base):
    """Cleaned & Normalized Fare Quotes Analytical Fact Table."""
    __tablename__ = "fare_quotes"
    __table_args__ = (
        Index(
            "ix_fare_quotes_clean_search",
            "is_outlier",
            "is_quarantined",
            "sector",
            "travel_date",
            "advance_days",
            "carrier_code",
        ),
    )

    quote_id = Column(String(64), primary_key=True)
    scrape_timestamp = Column(DateTime, nullable=False, index=True)
    route_id = Column(Integer, ForeignKey("routes.route_id"), nullable=True)
    carrier_id = Column(Integer, ForeignKey("carriers.carrier_id"), nullable=True)
    sector = Column(String(16), nullable=False, index=True)
    origin_iata = Column(String(3), nullable=False, index=True)
    destination_iata = Column(String(3), nullable=False, index=True)
    carrier_code = Column(String(3), nullable=False, index=True)
    source = Column(String(32), nullable=False)
    source_type = Column(String(16), default="airline")
    travel_date = Column(Date, nullable=False, index=True)
    advance_days = Column(Integer, nullable=False, index=True)
    flight_number = Column(String(16), nullable=False, index=True)
    departure_time = Column(Time, nullable=True)
    arrival_time = Column(Time, nullable=True)
    duration_minutes = Column(Integer, default=120)
    stops = Column(Integer, default=0)
    fare_class = Column(String(16), default="economy")
    currency = Column(String(8), default="INR")

    # 6 statutory fare components
    base_fare = Column(Float, nullable=False)
    fuel_surcharge = Column(Float, nullable=False)
    airport_taxes_gst = Column(Float, nullable=False)
    user_dev_fee = Column(Float, nullable=False)
    passenger_service_fee = Column(Float, nullable=False)
    convenience_fee = Column(Float, default=0.0)
    total_fare = Column(Float, nullable=False)

    seats_available = Column(Integer, nullable=True)
    is_sold_out = Column(Boolean, default=False)
    is_outlier = Column(Boolean, default=False, index=True)
    outlier_reason = Column(String(64), nullable=True)
    data_quality_score = Column(Float, nullable=False, index=True)
    is_quarantined = Column(Boolean, default=False, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    route_rel = relationship("Route", back_populates="quotes")
    carrier_rel = relationship("Carrier", back_populates="quotes")


class DailyIndexRecord(Base):
    """Daily Elementary and Route Price Indices."""
    __tablename__ = "daily_index"

    index_id = Column(Integer, primary_key=True, autoincrement=True)
    computation_date = Column(Date, nullable=False, index=True)
    sector = Column(String(16), nullable=False, index=True)
    carrier_code = Column(String(3), nullable=True, index=True)
    advance_window = Column(Integer, nullable=False, index=True)
    index_value = Column(Float, nullable=False)
    quotes_count = Column(Integer, default=0)
    frequency = Column(String(16), default="daily")
    created_at = Column(DateTime, default=datetime.utcnow)


class AtfPriceRecord(Base):
    """Aviation Turbine Fuel (ATF) Reference Prices."""
    __tablename__ = "atf_prices"

    atf_id = Column(Integer, primary_key=True, autoincrement=True)
    effective_date = Column(Date, nullable=False, index=True)
    airport_metro = Column(String(32), nullable=False)
    price_per_kl = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class CpiReferenceRecord(Base):
    """Official CPI Reference Series."""
    __tablename__ = "cpi_reference"

    cpi_id = Column(Integer, primary_key=True, autoincrement=True)
    month_year = Column(String(7), unique=True, nullable=False) # 'YYYY-MM'
    cpi_headline_all_india = Column(Float, nullable=False)
    cpi_transport_component = Column(Float, nullable=False)
    transport_weight_pct = Column(Float, default=9.43)
    created_at = Column(DateTime, default=datetime.utcnow)


class MospiCpiRecord(Base):
    """Official MoSPI eSankhyiki Consumer Price Index & Airfare Statistics."""
    __tablename__ = "esankhyiki_cpi"

    record_id = Column(Integer, primary_key=True, autoincrement=True)
    base_year = Column(String(8), nullable=False, default="2024", index=True)
    series = Column(String(16), nullable=False, default="Current")
    year = Column(String(8), nullable=False, index=True)
    month = Column(String(16), nullable=False, index=True)
    state = Column(String(64), nullable=False, index=True)
    sector = Column(String(16), nullable=False, default="Combined")  # Rural, Urban, Combined
    division = Column(String(128), nullable=True)
    group_name = Column(String(128), nullable=True)
    item = Column(String(128), nullable=True)
    code = Column(String(32), nullable=True, index=True)
    index_value = Column(Float, nullable=False)
    inflation_pct = Column(Float, nullable=True)
    imputation = Column(String(8), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==============================================================================
# 2. REFERENCE SEED DATA (DGCA 30 SECTORS & 5 CARRIERS)
# ==============================================================================

SEED_ROUTES = [
    ("DEL-BOM", "DEL", "BOM", "Delhi", "Mumbai", 1148.0, "Trunk", 0.075),
    ("BOM-DEL", "BOM", "DEL", "Mumbai", "Delhi", 1148.0, "Trunk", 0.075),
    ("DEL-BLR", "DEL", "BLR", "Delhi", "Bengaluru", 1740.0, "Trunk", 0.050),
    ("BLR-DEL", "BLR", "DEL", "Bengaluru", "Delhi", 1740.0, "Trunk", 0.050),
    ("BOM-BLR", "BOM", "BLR", "Mumbai", "Bengaluru", 842.0, "Trunk", 0.045),
    ("BLR-BOM", "BLR", "BOM", "Bengaluru", "Mumbai", 842.0, "Trunk", 0.045),
    ("DEL-HYD", "DEL", "HYD", "Delhi", "Hyderabad", 1253.0, "Trunk", 0.0375),
    ("HYD-DEL", "HYD", "DEL", "Hyderabad", "Delhi", 1253.0, "Trunk", 0.0375),
    ("DEL-CCU", "DEL", "CCU", "Delhi", "Kolkata", 1305.0, "Trunk", 0.035),
    ("CCU-DEL", "CCU", "DEL", "Kolkata", "Delhi", 1305.0, "Trunk", 0.035),
    ("HYD-BOM", "HYD", "BOM", "Hyderabad", "Mumbai", 617.0, "High-Density", 0.025),
    ("BOM-HYD", "BOM", "HYD", "Mumbai", "Hyderabad", 617.0, "High-Density", 0.025),
    ("PNQ-DEL", "PNQ", "DEL", "Pune", "Delhi", 1173.0, "High-Density", 0.025),
    ("DEL-PNQ", "DEL", "PNQ", "Delhi", "Pune", 1173.0, "High-Density", 0.025),
    ("AMD-DEL", "AMD", "DEL", "Ahmedabad", "Delhi", 775.0, "High-Density", 0.0225),
    ("DEL-AMD", "DEL", "AMD", "Delhi", "Ahmedabad", 775.0, "High-Density", 0.0225),
    ("HYD-BLR", "HYD", "BLR", "Hyderabad", "Bengaluru", 500.0, "High-Density", 0.0225),
    ("BLR-HYD", "BLR", "HYD", "Bengaluru", "Hyderabad", 500.0, "High-Density", 0.0225),
    ("MAA-DEL", "MAA", "DEL", "Chennai", "Delhi", 1760.0, "High-Density", 0.020),
    ("DEL-MAA", "DEL", "MAA", "Delhi", "Chennai", 1760.0, "High-Density", 0.020),
    ("DEL-JAI", "DEL", "JAI", "Delhi", "Jaipur", 241.0, "Regional", 0.0175),
    ("JAI-DEL", "JAI", "DEL", "Jaipur", "Delhi", 241.0, "Regional", 0.0175),
    ("BOM-GOI", "BOM", "GOI", "Mumbai", "Goa", 435.0, "Regional", 0.020),
    ("GOI-BOM", "GOI", "BOM", "Goa", "Mumbai", 435.0, "Regional", 0.020),
    ("DEL-LKO", "DEL", "LKO", "Delhi", "Lucknow", 418.0, "Regional", 0.0175),
    ("LKO-DEL", "LKO", "DEL", "Lucknow", "Delhi", 418.0, "Regional", 0.0175),
    ("DEL-SXR", "DEL", "SXR", "Delhi", "Srinagar", 650.0, "Seasonal", 0.020),
    ("SXR-DEL", "SXR", "DEL", "Srinagar", "Delhi", 650.0, "Seasonal", 0.020),
    ("CCU-GAU", "CCU", "GAU", "Kolkata", "Guwahati", 510.0, "Regional", 0.0175),
    ("GAU-CCU", "GAU", "CCU", "Guwahati", "Kolkata", 510.0, "Regional", 0.0175),
]

SEED_CARRIERS = [
    ("6E", "IndiGo", "LCC", 0.62),
    ("AI", "Air India", "FSC", 0.14),
    ("IX", "Air India Express", "LCC", 0.07),
    ("QP", "Akasa Air", "LCC", 0.05),
    ("SG", "SpiceJet", "LCC", 0.04),
]


# ==============================================================================
# 3. STORAGE ENGINE CLASS
# ==============================================================================

try:
    from apix.config import settings
    DEFAULT_DB_URL = settings.DATABASE_URL
except Exception:
    DEFAULT_DB_URL = os.getenv("APIX_DATABASE_URL") or os.getenv("DATABASE_URL") or "sqlite:///apix.db"


class StorageEngine:
    """
    SQLite / SQLAlchemy analytical store manager for APIx.
    Provides schema management, transactional inserts, and analytical queries.
    """

    def __init__(self, db_url: Optional[str] = None):
        self.db_url = db_url or DEFAULT_DB_URL
        connect_args = {}
        if "sqlite" in self.db_url:
            connect_args["check_same_thread"] = False

        self.engine = create_engine(self.db_url, connect_args=connect_args)

        if "sqlite" in self.db_url and ":memory:" not in self.db_url:
            @event.listens_for(self.engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA journal_mode=WAL;")
                cursor.execute("PRAGMA synchronous=NORMAL;")
                cursor.execute("PRAGMA busy_timeout=5000;")
                cursor.close()

        self.SessionFactory = sessionmaker(bind=self.engine)
        self.init_db()

    def get_session(self) -> Session:
        """Returns a new SQLAlchemy Session."""
        return self.SessionFactory()

    def init_db(self) -> None:
        """Creates all required tables and seeds dimension tables if empty."""
        Base.metadata.create_all(self.engine)
        self._seed_dimensions_if_empty()

    def _seed_dimensions_if_empty(self) -> None:
        """Populates routes and carriers dimension tables if not yet seeded."""
        with self.get_session() as session:
            # Seed carriers
            if session.query(Carrier).count() == 0:
                for code, name, c_type, share in SEED_CARRIERS:
                    session.add(
                        Carrier(
                            iata_code=code,
                            name=name,
                            carrier_type=c_type,
                            dgca_market_share=share,
                            is_active=True,
                        )
                    )

            # Seed routes
            if session.query(Route).count() == 0:
                for sec, o, d, oc, dc, dist, tier, w in SEED_ROUTES:
                    session.add(
                        Route(
                            sector=sec,
                            origin_iata=o,
                            destination_iata=d,
                            origin_city=oc,
                            destination_city=dc,
                            distance_km=dist,
                            route_tier=tier,
                            dgca_weight=w,
                            is_active=True,
                        )
                    )
            session.commit()

    def insert_quotes(self, quotes: List[Union[CleanedFareQuote, RawFareQuote, Dict[str, Any]]]) -> int:
        """
        Inserts fare quotes into the database fact table.
        Automatically cleans RawFareQuote instances if provided uncleaned.

        Args:
            quotes: List of CleanedFareQuote, RawFareQuote, or quote dictionaries.

        Returns:
            Number of quotes inserted.
        """
        if not quotes:
            return 0

        from apix.pipeline.quality import clean_quote

        records_to_insert: List[FareQuoteRecord] = []
        with self.get_session() as session:
            # Build lookup cache for route and carrier IDs
            route_cache = {r.sector: r.route_id for r in session.query(Route).all()}
            carrier_cache = {c.iata_code: c.carrier_id for c in session.query(Carrier).all()}

            for item in quotes:
                # Duck-type and normalize diverse quote input representations
                if isinstance(item, CleanedFareQuote):
                    cleaned = item
                elif isinstance(item, RawFareQuote):
                    cleaned = clean_quote(item)
                elif hasattr(item, "raw_total_fare") and hasattr(item, "model_dump"):
                    raw = RawFareQuote(**item.model_dump())
                    cleaned = clean_quote(raw)
                elif hasattr(item, "model_dump"):
                    d = item.model_dump()
                    if "data_quality_score" in d or "base_fare" in d:
                        cleaned = CleanedFareQuote(**d)
                    else:
                        raw = RawFareQuote(**d)
                        cleaned = clean_quote(raw)
                elif isinstance(item, dict):
                    if "data_quality_score" in item or "base_fare" in item:
                        cleaned = CleanedFareQuote(**item)
                    else:
                        raw = RawFareQuote(**item)
                        cleaned = clean_quote(raw)
                elif hasattr(item, "raw_total_fare") or hasattr(item, "total_fare"):
                    d = {
                        k: getattr(item, k)
                        for k in dir(item)
                        if not k.startswith("_") and not callable(getattr(item, k))
                    }
                    if "data_quality_score" in d or "base_fare" in d:
                        cleaned = CleanedFareQuote(**d)
                    else:
                        raw = RawFareQuote(**d)
                        cleaned = clean_quote(raw)
                else:
                    continue

                r_id = route_cache.get(cleaned.sector)
                c_id = carrier_cache.get(cleaned.carrier_code)

                record = FareQuoteRecord(
                    quote_id=cleaned.quote_id,
                    scrape_timestamp=cleaned.scrape_timestamp,
                    route_id=r_id,
                    carrier_id=c_id,
                    sector=cleaned.sector,
                    origin_iata=cleaned.origin_iata,
                    destination_iata=cleaned.destination_iata,
                    carrier_code=cleaned.carrier_code,
                    source=cleaned.source,
                    source_type=cleaned.source_type,
                    travel_date=cleaned.travel_date,
                    advance_days=cleaned.advance_days,
                    flight_number=cleaned.flight_number,
                    departure_time=cleaned.departure_time,
                    arrival_time=cleaned.arrival_time,
                    duration_minutes=cleaned.duration_minutes,
                    stops=cleaned.stops,
                    fare_class=cleaned.fare_class,
                    currency=cleaned.currency,
                    base_fare=float(cleaned.base_fare),
                    fuel_surcharge=float(cleaned.fuel_surcharge),
                    airport_taxes_gst=float(cleaned.airport_taxes_gst),
                    user_dev_fee=float(cleaned.user_development_fee),
                    passenger_service_fee=float(cleaned.passenger_service_fee),
                    convenience_fee=float(cleaned.convenience_fee),
                    total_fare=float(cleaned.total_fare),
                    seats_available=cleaned.seats_available,
                    is_sold_out=cleaned.is_sold_out,
                    is_outlier=cleaned.is_outlier,
                    outlier_reason=cleaned.outlier_reason,
                    data_quality_score=cleaned.data_quality_score,
                    is_quarantined=cleaned.is_quarantined,
                )
                session.merge(record)  # Idempotent upsert
                records_to_insert.append(record)

            session.commit()
            return len(records_to_insert)

    def get_clean_quotes(
        self,
        start_date: Optional[Union[date, str]] = None,
        end_date: Optional[Union[date, str]] = None,
        min_dqs: float = 80.0,
        sector: Optional[str] = None,
        carrier: Optional[str] = None,
        advance_window: Optional[int] = None,
    ) -> List[CleanedFareQuote]:
        """
        Retrieves verified clean quotes meeting institutional CPI criteria:
        - data_quality_score >= min_dqs (default 80.0)
        - is_outlier == False
        - is_quarantined == False

        Interface contract for Module B ↔ Module C (Econometric Engine).
        """
        # Convert date strings if needed
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

        with self.get_session() as session:
            stmt = select(FareQuoteRecord).where(
                and_(
                    FareQuoteRecord.data_quality_score >= min_dqs,
                    FareQuoteRecord.is_outlier == False,
                    FareQuoteRecord.is_quarantined == False,
                )
            )

            if start_date is not None:
                stmt = stmt.where(FareQuoteRecord.travel_date >= start_date)
            if end_date is not None:
                stmt = stmt.where(FareQuoteRecord.travel_date <= end_date)
            if sector is not None:
                stmt = stmt.where(FareQuoteRecord.sector == sector.upper().strip())
            if carrier is not None:
                stmt = stmt.where(FareQuoteRecord.carrier_code == carrier.upper().strip())
            if advance_window is not None:
                stmt = stmt.where(FareQuoteRecord.advance_days == advance_window)

            results = session.execute(stmt).scalars().all()

            output: List[CleanedFareQuote] = []
            for r in results:
                decomp = DecomposedFare(
                    base_fare=Decimal(str(r.base_fare)),
                    fuel_surcharge=Decimal(str(r.fuel_surcharge)),
                    airport_taxes_gst=Decimal(str(r.airport_taxes_gst)),
                    user_development_fee=Decimal(str(r.user_dev_fee)),
                    passenger_service_fee=Decimal(str(r.passenger_service_fee)),
                    convenience_fee=Decimal(str(r.convenience_fee)),
                    total_fare=Decimal(str(r.total_fare)),
                )
                quote = CleanedFareQuote(
                    quote_id=r.quote_id,
                    route_id=r.route_id,
                    carrier_id=r.carrier_id,
                    sector=r.sector,
                    origin_iata=r.origin_iata,
                    destination_iata=r.destination_iata,
                    carrier_code=r.carrier_code,
                    source=r.source,
                    source_type=r.source_type,
                    scrape_timestamp=r.scrape_timestamp,
                    travel_date=r.travel_date,
                    advance_days=r.advance_days,
                    flight_number=r.flight_number,
                    departure_time=r.departure_time or time(8, 0),
                    arrival_time=r.arrival_time or time(10, 30),
                    duration_minutes=r.duration_minutes or 120,
                    stops=r.stops,
                    fare_class=r.fare_class,
                    currency=r.currency,
                    base_fare=decomp.base_fare,
                    fuel_surcharge=decomp.fuel_surcharge,
                    airport_taxes_gst=decomp.airport_taxes_gst,
                    user_development_fee=decomp.user_development_fee,
                    passenger_service_fee=decomp.passenger_service_fee,
                    convenience_fee=decomp.convenience_fee,
                    total_fare=decomp.total_fare,
                    decomposed_fare=decomp,
                    seats_available=r.seats_available,
                    is_sold_out=r.is_sold_out,
                    is_outlier=r.is_outlier,
                    outlier_reason=r.outlier_reason,
                    data_quality_score=r.data_quality_score,
                    is_quarantined=r.is_quarantined,
                )
                output.append(quote)

            return output

    def get_route_summary(
        self,
        sector: str,
        advance_window: Optional[int] = None,
        travel_date: Optional[date] = None,
    ) -> Dict[str, Any]:
        """Calculates route-level analytical summary stats across carriers."""
        with self.get_session() as session:
            stmt = select(
                FareQuoteRecord.carrier_code,
                func.count(FareQuoteRecord.quote_id).label("count"),
                func.avg(FareQuoteRecord.total_fare).label("avg_fare"),
                func.min(FareQuoteRecord.total_fare).label("min_fare"),
                func.max(FareQuoteRecord.total_fare).label("max_fare"),
            ).where(
                and_(
                    FareQuoteRecord.sector == sector.upper().strip(),
                    FareQuoteRecord.data_quality_score >= 80.0,
                    FareQuoteRecord.is_outlier == False,
                )
            )

            if advance_window is not None:
                stmt = stmt.where(FareQuoteRecord.advance_days == advance_window)
            if travel_date is not None:
                stmt = stmt.where(FareQuoteRecord.travel_date == travel_date)

            stmt = stmt.group_by(FareQuoteRecord.carrier_code)
            rows = session.execute(stmt).all()

            carriers_summary = {}
            for row in rows:
                carriers_summary[row.carrier_code] = {
                    "count": row.count,
                    "avg_fare": round(float(row.avg_fare), 2),
                    "min_fare": round(float(row.min_fare), 2),
                    "max_fare": round(float(row.max_fare), 2),
                }

            return {
                "sector": sector,
                "advance_window": advance_window,
                "travel_date": str(travel_date) if travel_date else None,
                "carriers": carriers_summary,
            }

    def store_esankhyiki_records(self, records: List[Dict[str, Any]]) -> int:
        """Stores official MoSPI eSankhyiki CPI records in SQLite."""
        if not records:
            return 0
        inserted = 0
        with self.get_session() as session:
            for r in records:
                idx_val = float(r.get("index", 100.0))
                inf_val = float(r["inflation"]) if r.get("inflation") is not None else None
                rec = MospiCpiRecord(
                    base_year=str(r.get("base_year", "2024")),
                    series=str(r.get("series", "Current")),
                    year=str(r.get("year", "2026")),
                    month=str(r.get("month", "August")),
                    state=str(r.get("state", "All India")),
                    sector=str(r.get("sector", "Combined")),
                    division=r.get("division"),
                    group_name=r.get("group"),
                    item=r.get("item"),
                    code=r.get("code"),
                    index_value=idx_val,
                    inflation_pct=inf_val,
                    imputation=r.get("imputation"),
                )
                session.add(rec)
                inserted += 1
            session.commit()
        return inserted

    def get_esankhyiki_cpi_benchmarks(self, base_year: str = "2024") -> Dict[str, Any]:
        """Returns official eSankhyiki CPI benchmarks for Airfare and Transport."""
        with self.get_session() as session:
            air_stmt = select(MospiCpiRecord).where(
                and_(
                    MospiCpiRecord.base_year == base_year,
                    MospiCpiRecord.state == "All India",
                    MospiCpiRecord.sector == "Combined",
                    MospiCpiRecord.code == "07.3.3.1.2.01",
                )
            ).order_by(MospiCpiRecord.record_id.desc()).limit(1)
            air_row = session.execute(air_stmt).scalar_one_or_none()

            trans_stmt = select(MospiCpiRecord).where(
                and_(
                    MospiCpiRecord.base_year == base_year,
                    MospiCpiRecord.state == "All India",
                    MospiCpiRecord.sector == "Combined",
                    MospiCpiRecord.code == "07",
                )
            ).order_by(MospiCpiRecord.record_id.desc()).limit(1)
            trans_row = session.execute(trans_stmt).scalar_one_or_none()

            return {
                "base_year": base_year,
                "airfare": {
                    "index": air_row.index_value if air_row else 135.49,
                    "inflation_pct": air_row.inflation_pct if air_row else 20.85,
                    "month": air_row.month if air_row else "August",
                    "year": air_row.year if air_row else "2026",
                    "code": "07.3.3.1.2.01",
                    "item": "Airfare (Domestic)",
                },
                "transport": {
                    "index": trans_row.index_value if trans_row else 105.90,
                    "inflation_pct": trans_row.inflation_pct if trans_row else 4.60,
                    "month": trans_row.month if trans_row else "August",
                    "year": trans_row.year if trans_row else "2026",
                    "code": "07",
                    "division": "Transport",
                },
                "cpi_weight_pct": 9.43,
            }


# ==============================================================================
# 4. MODULE-LEVEL FUNCTIONAL API
# ==============================================================================

_DEFAULT_ENGINE: Optional[StorageEngine] = None


def get_storage_engine(db_url: Optional[str] = None) -> StorageEngine:
    """Returns singleton default StorageEngine or creates one with custom db_url."""
    global _DEFAULT_ENGINE
    if db_url is not None:
        return StorageEngine(db_url=db_url)
    if _DEFAULT_ENGINE is None:
        default_db = os.environ.get("APIX_DATABASE_URL", "sqlite:///apix.db")
        _DEFAULT_ENGINE = StorageEngine(db_url=default_db)
    return _DEFAULT_ENGINE


def insert_quotes(
    quotes: List[Union[CleanedFareQuote, RawFareQuote, Dict[str, Any]]],
    db_url: Optional[str] = None,
) -> int:
    """Global functional wrapper for quote ingestion into database."""
    engine = get_storage_engine(db_url=db_url)
    return engine.insert_quotes(quotes)


def get_clean_quotes(
    start_date: Optional[Union[date, str]] = None,
    end_date: Optional[Union[date, str]] = None,
    min_dqs: float = 80.0,
    sector: Optional[str] = None,
    carrier: Optional[str] = None,
    advance_window: Optional[int] = None,
    db_url: Optional[str] = None,
) -> List[CleanedFareQuote]:
    """Global functional wrapper for retrieving verified clean quotes."""
    engine = get_storage_engine(db_url=db_url)
    return engine.get_clean_quotes(
        start_date=start_date,
        end_date=end_date,
        min_dqs=min_dqs,
        sector=sector,
        carrier=carrier,
        advance_window=advance_window,
    )
