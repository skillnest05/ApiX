"""
Data Quality Scoring (DQS), Sold-Out Tracking & Deduplication Engine (Module B).
Implements:
1. Multi-criteria 0-100 Data Quality Score (DQS >= 80 threshold for index inclusion)
2. Capacity exhaustion and sold-out flight tracking (preserves non-zero price signal)
3. Canonical deduplication across flight instances and multiple scraping sources
4. End-to-end quote normalization pipeline
"""

from collections import defaultdict
from datetime import date, datetime, time
from decimal import Decimal
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from apix.pipeline.decomposition import decompose_fare, decompose_quote
from apix.pipeline.outlier import (
    evaluate_outlier_status,
    filter_iqr_bounds,
    is_domain_ceiling_violation,
    is_domain_floor_violation,
)
from apix.pipeline.schemas import (
    CleanedFareQuote,
    DecomposedFare,
    DQSBreakdown,
    RawFareQuote,
)


DIRECT_AIRLINE_SOURCES = {
    "indigo", "airindia", "airindiaexpress", "akasa", "spicejet",
    "6e", "ai", "ix", "qp", "sg"
}


# ==============================================================================
# 1. 0-100 DATA QUALITY SCORE (DQS) ALGORITHM
# ==============================================================================

def compute_dqs_breakdown(
    raw: Union[RawFareQuote, Any],
    decomp: Union[DecomposedFare, Dict[str, Any], Any],
    is_outlier: bool = False,
    cohort_median: Optional[float] = None,
    is_synthetic_imputation: bool = False,
) -> DQSBreakdown:
    """
    Computes detailed sub-component breakdown of the 0-100 Data Quality Score.

    Weights:
    1. Schema Completeness: W = 0.25 (Max 25 pts)
    2. Decomposition Integrity: W = 0.20 (Max 20 pts)
    3. Domain & Outlier Bounds: W = 0.25 (Max 25 pts)
    4. Cross-Source Concordance: W = 0.15 (Max 15 pts)
    5. Source Reliability: W = 0.15 (Max 15 pts)
    """
    # --------------------------------------------------------------------------
    # 1. Schema Completeness (Max 25 pts)
    # --------------------------------------------------------------------------
    s_schema = 25.0
    fn = getattr(raw, "flight_number", None) or (raw.get("flight_number") if isinstance(raw, dict) else None)
    arr_t = getattr(raw, "arrival_time", None) or (raw.get("arrival_time") if isinstance(raw, dict) else None)
    dur = getattr(raw, "duration_minutes", None) or (raw.get("duration_minutes") if isinstance(raw, dict) else 120)

    if not fn:
        s_schema -= 25.0
    if arr_t is None:
        s_schema -= 10.0
    if dur is not None and dur <= 0:
        s_schema -= 15.0
    s_schema = max(0.0, min(25.0, s_schema))

    # --------------------------------------------------------------------------
    # 2. Decomposition Integrity (Max 20 pts)
    # --------------------------------------------------------------------------
    if isinstance(decomp, DecomposedFare):
        reconstructed = float(decomp.reconstructed_total)
        total_val = float(decomp.total_fare)
    elif isinstance(decomp, dict):
        reconstructed = sum([
            float(decomp.get("base_fare", 0.0)),
            float(decomp.get("fuel_surcharge", 0.0)),
            float(decomp.get("airport_taxes_gst", 0.0)),
            float(decomp.get("user_dev_fee", decomp.get("user_development_fee", 0.0))),
            float(decomp.get("passenger_service_fee", 0.0)),
            float(decomp.get("convenience_fee", 0.0)),
        ])
        total_val = float(decomp.get("total_fare", 0.0))
    else:
        reconstructed = 0.0
        total_val = float(getattr(raw, "raw_total_fare", 0.0))

    decomp_diff = abs(reconstructed - total_val)
    if decomp_diff <= 1.00:
        s_decomp = 12.0 if is_synthetic_imputation else 20.0
    elif decomp_diff <= 5.00:
        s_decomp = 12.0
    elif decomp_diff <= 50.00:
        s_decomp = 5.0
    else:
        s_decomp = 0.0

    # --------------------------------------------------------------------------
    # 3. Domain & Outlier Bounds (Max 25 pts)
    # --------------------------------------------------------------------------
    raw_fare_val = float(getattr(raw, "raw_total_fare", total_val))
    if raw_fare_val < 999.0 or raw_fare_val > 25000.0:
        s_bounds = 0.0
    elif not is_outlier:
        s_bounds = 25.0
    else:
        s_bounds = 10.0

    # --------------------------------------------------------------------------
    # 4. Cross-Source Concordance (Max 15 pts)
    # --------------------------------------------------------------------------
    if cohort_median is not None and cohort_median > 0.0:
        dev_pct = abs(raw_fare_val - cohort_median) / cohort_median
        if dev_pct <= 0.05:
            s_cross = 15.0
        elif dev_pct <= 0.20:
            s_cross = 10.0
        elif dev_pct <= 0.50:
            s_cross = 5.0
        else:
            s_cross = 0.0
    else:
        s_cross = 15.0  # Baseline concordance score when single source

    # --------------------------------------------------------------------------
    # 5. Source Reliability (Max 15 pts)
    # --------------------------------------------------------------------------
    src = str(getattr(raw, "source", "airline") or "airline").lower().strip()
    if src in DIRECT_AIRLINE_SOURCES:
        s_source = 15.0
    elif src in ["makemytrip", "cleartrip", "easemytrip", "ixigo", "mmt", "ct", "ixi", "emt"]:
        s_source = 13.0
    else:
        s_source = 8.0

    total_score = round(s_schema + s_decomp + s_bounds + s_cross + s_source, 2)
    total_score = max(0.0, min(100.0, total_score))

    return DQSBreakdown(
        schema_completeness=s_schema,
        decomposition_integrity=s_decomp,
        domain_and_bounds=s_bounds,
        cross_source_concordance=s_cross,
        source_reliability=s_source,
        total_score=total_score,
        is_index_eligible=total_score >= 80.0,
        is_quarantined=total_score < 60.0,
    )


def compute_dqs_score(
    raw: Union[RawFareQuote, Any],
    decomp: Union[DecomposedFare, Dict[str, Any], Any],
    is_outlier: bool = False,
    cohort_median: Optional[float] = None,
    is_synthetic_imputation: bool = False,
) -> float:
    """
    Computes overall 0-100 Data Quality Score (DQS).
    Matches both production and test harness signatures.
    """
    breakdown = compute_dqs_breakdown(
        raw=raw,
        decomp=decomp,
        is_outlier=is_outlier,
        cohort_median=cohort_median,
        is_synthetic_imputation=is_synthetic_imputation,
    )
    return breakdown.total_score


# ==============================================================================
# 2. SOLD-OUT FLIGHT CAPACITY TRACKING
# ==============================================================================

def handle_sold_out(
    quote: RawFareQuote,
    reservation_ceiling: float = 16500.0,
) -> RawFareQuote:
    """
    Handles capacity exhaustion:
    - Marks is_sold_out = True and seats_available = 0.
    - Prevents total_fare from being corrupted to ₹0.00.
    - If total_fare is missing or <= 0, assigns reservation ceiling tariff.
    """
    is_exhausted = quote.is_sold_out or (quote.seats_available is not None and quote.seats_available == 0)

    if is_exhausted:
        curr_fare = float(quote.raw_total_fare)
        safe_fare = curr_fare if curr_fare > 0.0 else reservation_ceiling
        return quote.model_copy(
            update={
                "is_sold_out": True,
                "seats_available": 0,
                "raw_total_fare": Decimal(str(safe_fare)),
            }
        )
    return quote


# ==============================================================================
# 3. CANONICAL DEDUPLICATION
# ==============================================================================

def canonical_flight_key(quote: Union[RawFareQuote, CleanedFareQuote, Any]) -> Tuple[str, str, str, str, date]:
    """Generates canonical key: (origin, destination, carrier, normalized_flight_number, travel_date)."""
    fn = getattr(quote, "flight_number", "") or ""
    norm_fn = re.sub(r'[^A-Z0-9]', '', str(fn).upper())
    return (
        str(getattr(quote, "origin_iata", "")).upper().strip(),
        str(getattr(quote, "destination_iata", "")).upper().strip(),
        str(getattr(quote, "carrier_code", "")).upper().strip(),
        norm_fn,
        getattr(quote, "travel_date", date.today()),
    )


def deduplicate_quotes(quotes: List[Any]) -> List[Any]:
    """
    Deduplicates multiple quotes for the identical flight instance across multiple sources.
    Priority hierarchy:
    1. Direct airline quote preferred over OTA quote.
    2. Lowest fare preferred if multiple same-tier quotes exist.
    """
    grouped: Dict[Tuple[str, str, str, str, date], List[Any]] = defaultdict(list)
    for q in quotes:
        key = canonical_flight_key(q)
        grouped[key].append(q)

    deduped: List[Any] = []
    for key, group in grouped.items():
        if len(group) == 1:
            deduped.append(group[0])
            continue

        # Sort group: direct airlines first, then by lowest fare
        def priority_score(item: Any) -> Tuple[int, float]:
            src = str(getattr(item, "source", "") or "").lower().strip()
            is_direct = 0 if src in DIRECT_AIRLINE_SOURCES else 1
            raw_f = getattr(item, "raw_total_fare", None)
            if raw_f is None:
                raw_f = getattr(item, "total_fare", 0.0)
            return (is_direct, float(raw_f))

        best_quote = min(group, key=priority_score)
        deduped.append(best_quote)

    return deduped



# ==============================================================================
# 4. NORMALIZATION & CLEANING PIPELINE
# ==============================================================================

def clean_quote(
    raw: RawFareQuote,
    cohort_fares: Optional[List[float]] = None,
    cohort_median: Optional[float] = None,
    prev_fare: Optional[float] = None,
    udf_rates: Optional[Dict[str, float]] = None,
) -> CleanedFareQuote:
    """
    Transforms a single RawFareQuote into a fully verified CleanedFareQuote:
    1. Handles sold out capacity status
    2. Decomposes into 6 statutory components
    3. Evaluates outlier status (floor, ceiling, IQR, temporal, cross-source)
    4. Computes 0-100 Data Quality Score (DQS)
    5. Sets quarantine flag if DQS < 60 or hard domain violation
    """
    # 1. Capacity exhaustion check
    processed_raw = handle_sold_out(raw)

    # 2. Statutory decomposition
    decomp = decompose_quote(processed_raw, udf_rates=udf_rates)

    # 3. Outlier evaluation
    outlier_res = evaluate_outlier_status(
        fare=processed_raw.raw_total_fare,
        cohort_fares=cohort_fares,
        prev_fare=prev_fare,
        cohort_median=cohort_median,
    )

    # 4. Compute DQS Score
    dqs = compute_dqs_score(
        raw=processed_raw,
        decomp=decomp,
        is_outlier=outlier_res.is_outlier,
        cohort_median=cohort_median,
    )

    is_quarantined = dqs < 60.0 or (
        outlier_res.is_outlier and ("DOMAIN_FLOOR" in outlier_res.flags or "DOMAIN_CEILING" in outlier_res.flags)
    )

    sector_str = f"{processed_raw.origin_iata}-{processed_raw.destination_iata}"

    return CleanedFareQuote(
        quote_id=processed_raw.quote_id,
        sector=sector_str,
        origin_iata=processed_raw.origin_iata,
        destination_iata=processed_raw.destination_iata,
        carrier_code=processed_raw.carrier_code,
        source=processed_raw.source,
        source_type=processed_raw.source_type,
        scrape_timestamp=processed_raw.scrape_timestamp,
        travel_date=processed_raw.travel_date,
        advance_days=processed_raw.advance_days,
        flight_number=processed_raw.flight_number,
        departure_time=processed_raw.departure_time,
        arrival_time=processed_raw.arrival_time,
        duration_minutes=processed_raw.duration_minutes,
        stops=processed_raw.stops,
        fare_class=processed_raw.fare_class,
        currency=processed_raw.currency,
        base_fare=decomp.base_fare,
        fuel_surcharge=decomp.fuel_surcharge,
        airport_taxes_gst=decomp.airport_taxes_gst,
        user_development_fee=decomp.user_development_fee,
        passenger_service_fee=decomp.passenger_service_fee,
        convenience_fee=decomp.convenience_fee,
        total_fare=decomp.total_fare,
        decomposed_fare=decomp,
        seats_available=processed_raw.seats_available,
        is_sold_out=processed_raw.is_sold_out,
        is_outlier=outlier_res.is_outlier,
        outlier_reason=outlier_res.outlier_reason,
        data_quality_score=dqs,
        is_quarantined=is_quarantined,
    )


def clean_quotes_batch(
    raw_quotes: List[RawFareQuote],
    deduplicate: bool = True,
    udf_rates: Optional[Dict[str, float]] = None,
) -> List[CleanedFareQuote]:
    """
    Cleans a batch of RawFareQuote instances:
    1. Deduplicates multiple quotes for identical flight instances.
    2. Groups into homogeneous cohorts (sector, advance_days, dow) to compute cohort medians and IQR bounds.
    3. Normalizes and validates each quote.
    """
    if deduplicate:
        quotes_to_process = deduplicate_quotes(raw_quotes)
    else:
        quotes_to_process = raw_quotes

    # Group fares into homogeneous cohorts: (sector, advance_days, dow)
    cohort_groups: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for q in quotes_to_process:
        sec = f"{q.origin_iata}-{q.destination_iata}"
        dow = q.travel_date.weekday()
        cohort_groups[(sec, q.advance_days, dow)].append(float(q.raw_total_fare))

    # Compute cohort medians
    cohort_medians: Dict[Tuple[str, int, int], float] = {
        k: float(np.median(v)) for k, v in cohort_groups.items() if len(v) > 0
    }

    cleaned_list: List[CleanedFareQuote] = []
    for q in quotes_to_process:
        sec = f"{q.origin_iata}-{q.destination_iata}"
        dow = q.travel_date.weekday()
        c_fares = cohort_groups.get((sec, q.advance_days, dow))
        c_med = cohort_medians.get((sec, q.advance_days, dow))

        cleaned = clean_quote(
            raw=q,
            cohort_fares=c_fares,
            cohort_median=c_med,
            udf_rates=udf_rates,
        )
        cleaned_list.append(cleaned)

    return cleaned_list
