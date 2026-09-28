"""
Stage 1 Econometric Index: Jevons Elementary Price Index (Module C).
Computes geometric mean of price relatives in log-space for homogeneous cohorts:
    (Sector r, Carrier c, Advance Window w).
Guarantees Fisher Time-Reversal Property, Commensurability, and Monotonicity.
"""

from datetime import date, datetime, time
import math
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from apix.pipeline.schemas import CleanedFareQuote


class JevonsCohortResult(BaseModel):
    """Result of Stage 1 Jevons Elementary Index computation for a homogeneous cohort."""
    model_config = ConfigDict(frozen=True)

    sector: str = Field(..., description="Directional route e.g. DEL-BOM")
    carrier_code: str = Field(..., description="Carrier IATA code e.g. 6E")
    advance_window: int = Field(..., description="Advance booking window in days: 1, 7, 15, 30, 45")
    base_quote_count: int = Field(..., ge=1, description="Number of valid quotes in base period")
    current_quote_count: int = Field(..., ge=1, description="Number of valid quotes in comparison period")
    matched_quote_count: int = Field(default=0, ge=0, description="Number of matched flight/time pairs")
    base_geometric_mean: float = Field(..., gt=0.0, description="Geometric mean of base period prices")
    current_geometric_mean: float = Field(..., gt=0.0, description="Geometric mean of current period prices")
    index_value: float = Field(..., gt=0.0, description="Jevons price relative (e.g. 1.05 = +5% price change)")
    fare_component: str = Field(default="total_fare", description="'total_fare' or 'base_fare'")


def compute_geometric_mean(prices: Union[np.ndarray, Sequence[float]]) -> float:
    """
    Computes geometric mean of strictly positive prices in natural log-space:
        G = exp( (1/n) * sum( ln(p_i) ) )

    Raises:
        ValueError: If price vector is empty or contains non-positive values.
    """
    arr = np.asarray(prices, dtype=float)
    if arr.size == 0:
        raise ValueError("Price vector must not be empty")
    if np.any(arr <= 0.0):
        raise ValueError("Prices must be strictly positive")
    return float(np.exp(np.mean(np.log(arr))))


def compute_jevons_index(
    p0: Union[np.ndarray, Sequence[float]],
    pt: Union[np.ndarray, Sequence[float]]
) -> float:
    """
    Computes Stage 1 Jevons elementary price index:
        I_{r,c,w}^{0,t} = exp( (1/n) * sum( ln(pt / p0) ) )

    When lengths match: evaluates mean log price relative.
    When lengths differ: evaluates ratio of cohort geometric means:
        exp( mean(ln(pt)) - mean(ln(p0)) )

    Guarantees Fisher Time-Reversal Property:
        P_J(0, t) * P_J(t, 0) == 1.0 +- 1e-6.
    
    Raises:
        ValueError: If either vector is empty or contains non-positive values.
    """
    p0_arr = np.asarray(p0, dtype=float)
    pt_arr = np.asarray(pt, dtype=float)

    if p0_arr.size == 0 or pt_arr.size == 0:
        raise ValueError("Cohort vectors must not be empty")
    if np.any(p0_arr <= 0.0) or np.any(pt_arr <= 0.0):
        raise ValueError("Prices must be strictly positive")

    if p0_arr.size == pt_arr.size:
        return float(np.exp(np.mean(np.log(pt_arr / p0_arr))))
    else:
        log_mean_pt = np.mean(np.log(pt_arr))
        log_mean_p0 = np.mean(np.log(p0_arr))
        return float(np.exp(log_mean_pt - log_mean_p0))


def _extract_fare(quote: Any, fare_component: str = "total_fare") -> float:
    """Extracts scalar numeric fare from quote object or dict."""
    if hasattr(quote, fare_component):
        val = getattr(quote, fare_component)
        return float(val)
    elif isinstance(quote, dict) and fare_component in quote:
        return float(quote[fare_component])
    elif hasattr(quote, "raw_total_fare") and fare_component == "total_fare":
        return float(quote.raw_total_fare)
    elif hasattr(quote, "total_fare"):
        return float(quote.total_fare)
    raise AttributeError(f"Cannot extract fare component '{fare_component}' from quote: {quote}")


def calculate_cohort_jevons(
    base_quotes: Sequence[Any],
    current_quotes: Sequence[Any],
    sector: str,
    carrier_code: str,
    advance_window: int,
    fare_component: str = "total_fare",
    match_flights: bool = True
) -> Optional[JevonsCohortResult]:
    """
    Filters base and current quotes for the given (sector, carrier_code, advance_window),
    matches quotes if requested, and calculates the Jevons elementary index.

    Returns:
        JevonsCohortResult if quotes exist in both periods, or None otherwise.
    """
    sec = sector.upper().strip()
    carr = carrier_code.upper().strip()

    def matches_cohort(q: Any) -> bool:
        q_sec = getattr(q, "sector", None)
        if q_sec is None and hasattr(q, "origin_iata") and hasattr(q, "destination_iata"):
            q_sec = f"{q.origin_iata}-{q.destination_iata}"
        elif isinstance(q, dict):
            q_sec = q.get("sector") or f"{q.get('origin_iata', '')}-{q.get('destination_iata', '')}"

        q_carr = getattr(q, "carrier_code", None) or getattr(q, "carrier", None)
        if isinstance(q, dict) and q_carr is None:
            q_carr = q.get("carrier_code") or q.get("carrier")

        q_win = getattr(q, "advance_days", None)
        if q_win is None:
            q_win = getattr(q, "advance_window", None)
        if isinstance(q, dict) and q_win is None:
            q_win = q.get("advance_days") or q.get("advance_window")

        if q_sec is None or q_carr is None or q_win is None:
            return False

        return (
            str(q_sec).upper().strip() == sec
            and str(q_carr).upper().strip() == carr
            and int(q_win) == advance_window
        )

    q0 = [q for q in base_quotes if matches_cohort(q)]
    qt = [q for q in current_quotes if matches_cohort(q)]

    if not q0 or not qt:
        return None

    p0_all = np.array([_extract_fare(q, fare_component) for q in q0], dtype=float)
    pt_all = np.array([_extract_fare(q, fare_component) for q in qt], dtype=float)

    if np.any(p0_all <= 0.0) or np.any(pt_all <= 0.0):
        # Exclude non-positive if present
        p0_all = p0_all[p0_all > 0.0]
        pt_all = pt_all[pt_all > 0.0]
        if p0_all.size == 0 or pt_all.size == 0:
            return None

    base_geom = compute_geometric_mean(p0_all)
    curr_geom = compute_geometric_mean(pt_all)

    matched_pairs = 0
    if match_flights:
        # Match by (flight_number, departure_time)
        q0_map: Dict[Tuple[str, str], List[float]] = {}
        for q in q0:
            fn = str(getattr(q, "flight_number", "")).strip().upper()
            dep = str(getattr(q, "departure_time", "")).strip()
            k = (fn, dep)
            fare_val = _extract_fare(q, fare_component)
            if fare_val > 0.0:
                q0_map.setdefault(k, []).append(fare_val)

        paired_p0: List[float] = []
        paired_pt: List[float] = []
        for q in qt:
            fn = str(getattr(q, "flight_number", "")).strip().upper()
            dep = str(getattr(q, "departure_time", "")).strip()
            k = (fn, dep)
            fare_val = _extract_fare(q, fare_component)
            if fare_val > 0.0 and k in q0_map and q0_map[k]:
                matched_p0 = q0_map[k].pop(0)
                paired_p0.append(matched_p0)
                paired_pt.append(fare_val)

        if paired_p0:
            matched_pairs = len(paired_p0)
            index_val = compute_jevons_index(paired_p0, paired_pt)
        else:
            index_val = curr_geom / base_geom
    else:
        index_val = curr_geom / base_geom

    return JevonsCohortResult(
        sector=sec,
        carrier_code=carr,
        advance_window=advance_window,
        base_quote_count=len(q0),
        current_quote_count=len(qt),
        matched_quote_count=matched_pairs,
        base_geometric_mean=round(base_geom, 4),
        current_geometric_mean=round(curr_geom, 4),
        index_value=round(index_val, 6),
        fare_component=fare_component,
    )


def compute_all_elementary_indices(
    base_quotes: Sequence[Any],
    current_quotes: Sequence[Any],
    sectors: Optional[Sequence[str]] = None,
    advance_windows: Optional[Sequence[int]] = None,
    fare_component: str = "total_fare",
    match_flights: bool = True
) -> Dict[Tuple[str, str, int], JevonsCohortResult]:
    """
    Computes Stage 1 Jevons elementary index for all populated cohorts:
        (sector, carrier_code, advance_window).

    Returns:
        Mapping: (sector, carrier_code, advance_window) -> JevonsCohortResult.
    """
    results: Dict[Tuple[str, str, int], JevonsCohortResult] = {}
    cohort_keys: Set[Tuple[str, str, int]] = set()

    def extract_cohort_key(q: Any) -> Optional[Tuple[str, str, int]]:
        sec = getattr(q, "sector", None)
        if sec is None and hasattr(q, "origin_iata") and hasattr(q, "destination_iata"):
            sec = f"{q.origin_iata}-{q.destination_iata}"
        elif isinstance(q, dict):
            sec = q.get("sector") or f"{q.get('origin_iata', '')}-{q.get('destination_iata', '')}"

        carr = getattr(q, "carrier_code", None) or getattr(q, "carrier", None)
        if isinstance(q, dict) and carr is None:
            carr = q.get("carrier_code") or q.get("carrier")

        win = getattr(q, "advance_days", None)
        if win is None:
            win = getattr(q, "advance_window", None)
        if isinstance(q, dict) and win is None:
            win = q.get("advance_days") or q.get("advance_window")

        if sec and carr and win is not None:
            return (str(sec).upper().strip(), str(carr).upper().strip(), int(win))
        return None

    for q in base_quotes:
        k = extract_cohort_key(q)
        if k:
            cohort_keys.add(k)
    for q in current_quotes:
        k = extract_cohort_key(q)
        if k:
            cohort_keys.add(k)

    for sec, carr, win in cohort_keys:
        if sectors and sec not in sectors:
            continue
        if advance_windows and win not in advance_windows:
            continue

        res = calculate_cohort_jevons(
            base_quotes=base_quotes,
            current_quotes=current_quotes,
            sector=sec,
            carrier_code=carr,
            advance_window=win,
            fare_component=fare_component,
            match_flights=match_flights,
        )
        if res is not None:
            results[(sec, carr, win)] = res

    return results


class JevonsEngine:
    """
    Object-oriented engine wrapper for Stage 1 Jevons Elementary Price Index.
    Provides methods for cohort computation and full basket execution.
    """

    @staticmethod
    def compute_index(
        p0: Union[np.ndarray, Sequence[float]],
        pt: Union[np.ndarray, Sequence[float]]
    ) -> float:
        """Computes Jevons elementary index between two price vectors."""
        return compute_jevons_index(p0, pt)

    @staticmethod
    def compute_geometric_mean(prices: Union[np.ndarray, Sequence[float]]) -> float:
        """Computes geometric mean of price vector."""
        return compute_geometric_mean(prices)

    @staticmethod
    def calculate_cohort(
        base_quotes: Sequence[Any],
        current_quotes: Sequence[Any],
        sector: str,
        carrier_code: str,
        advance_window: int,
        fare_component: str = "total_fare",
        match_flights: bool = True
    ) -> Optional[JevonsCohortResult]:
        """Calculates cohort Jevons index."""
        return calculate_cohort_jevons(
            base_quotes=base_quotes,
            current_quotes=current_quotes,
            sector=sector,
            carrier_code=carrier_code,
            advance_window=advance_window,
            fare_component=fare_component,
            match_flights=match_flights,
        )

    @staticmethod
    def compute_all(
        base_quotes: Sequence[Any],
        current_quotes: Sequence[Any],
        sectors: Optional[Sequence[str]] = None,
        advance_windows: Optional[Sequence[int]] = None,
        fare_component: str = "total_fare",
        match_flights: bool = True
    ) -> Dict[Tuple[str, str, int], JevonsCohortResult]:
        """Calculates all elementary cohort indices."""
        return compute_all_elementary_indices(
            base_quotes=base_quotes,
            current_quotes=current_quotes,
            sectors=sectors,
            advance_windows=advance_windows,
            fare_component=fare_component,
            match_flights=match_flights,
        )
