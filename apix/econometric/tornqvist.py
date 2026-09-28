"""
Stage 2 Econometric Index: Superlative Törnqvist Route-Level Index (Module C).
Aggregates carrier elementary indices across domestic airlines for sector-window (r, w):
    ln P_T^{0,t}(r,w) = sum_{c in C^{0 cap t}} ((w_c^0 + w_c^t) / 2) * ln(I_{r,c,w}^{0,t})
Integrates DGCA domestic market shares with dynamic active set normalization for carrier entry/exit.
Guarantees Fisher Time-Reversal Property, Monopoly Route Reduction, and Scale Homogeneity.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from apix.econometric.jevons import JevonsCohortResult


# Canonical DGCA Domestic Market Shares (August 2026 Reference)
DEFAULT_DGCA_CARRIER_SHARES = {
    "6E": 0.62,   # IndiGo
    "AI": 0.14,   # Air India
    "IX": 0.07,   # Air India Express
    "QP": 0.05,   # Akasa Air
    "SG": 0.04,   # SpiceJet
    "OTHERS": 0.08 # Other regional carriers
}


class TornqvistRouteResult(BaseModel):
    """Result of Stage 2 Törnqvist Superlative Index aggregation for a sector-window."""
    model_config = ConfigDict(frozen=True)

    sector: str = Field(..., description="Directional route e.g. DEL-BOM")
    advance_window: int = Field(..., description="Advance booking window in days: 1, 7, 15, 30, 45")
    active_carriers: List[str] = Field(..., description="Active carriers C^{0 cap t} included in aggregation")
    carrier_indices: Dict[str, float] = Field(..., description="Elementary Jevons indices per carrier")
    carrier_weights_base: Dict[str, float] = Field(..., description="Normalized carrier weights in period 0")
    carrier_weights_curr: Dict[str, float] = Field(..., description="Normalized carrier weights in period t")
    carrier_average_weights: Dict[str, float] = Field(..., description="Harmonized Törnqvist weights (w0 + wt)/2")
    index_value: float = Field(..., gt=0.0, description="Törnqvist price relative P_T^{0,t}(r,w)")
    is_monopoly: bool = Field(default=False, description="True if only 1 carrier operated on sector-window")
    missing_carriers: List[str] = Field(default_factory=list, description="Carriers without valid quotes")


def load_carrier_market_shares(weights_path: Optional[Union[str, Path]] = None) -> Dict[str, float]:
    """
    Loads carrier domestic market shares from dgca_weights.json.
    Falls back to canonical DEFAULT_DGCA_CARRIER_SHARES if file is missing or unparseable.
    """
    if weights_path is None:
        base_dir = Path(__file__).resolve().parent.parent.parent
        weights_path = base_dir / "data" / "dgca_weights.json"
    else:
        weights_path = Path(weights_path)

    if weights_path.exists():
        try:
            with open(weights_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                shares_dict = data.get("carrier_market_shares", {})
                loaded: Dict[str, float] = {}
                for k, v in shares_dict.items():
                    if isinstance(v, dict) and "market_share" in v:
                        loaded[str(k).upper().strip()] = float(v["market_share"])
                    elif isinstance(v, (int, float)):
                        loaded[str(k).upper().strip()] = float(v)
                if loaded:
                    return loaded
        except Exception:
            pass

    return dict(DEFAULT_DGCA_CARRIER_SHARES)


def compute_tornqvist_index(
    elementary_indices: Union[np.ndarray, Sequence[float]],
    w0: Union[np.ndarray, Sequence[float]],
    wt: Union[np.ndarray, Sequence[float]]
) -> float:
    """
    Computes Stage 2 Törnqvist superlative index:
        ln(P_T) = sum_{c in C} ((w0_norm + wt_norm)/2) * ln(I_rel)
        P_T = exp(ln(P_T))

    Normalizes weights over active operating carriers so sum(w0_norm) == 1.0 and sum(wt_norm) == 1.0.
    Satisfies Fisher Time-Reversal Property:
        P_T(0, t) * P_T(t, 0) == 1.0 +- 1e-6.
    
    Raises:
        ValueError: On empty inputs, dimension mismatches, non-positive relatives, or non-positive weight sums.
    """
    I_rel = np.asarray(elementary_indices, dtype=float)
    w0_arr = np.asarray(w0, dtype=float)
    wt_arr = np.asarray(wt, dtype=float)

    if I_rel.size == 0 or w0_arr.size == 0 or wt_arr.size == 0:
        raise ValueError("Inputs to Törnqvist index must not be empty")
    if I_rel.size != w0_arr.size or I_rel.size != wt_arr.size:
        raise ValueError(
            f"Mismatched dimensions: I_rel({I_rel.size}), w0({w0_arr.size}), wt({wt_arr.size})"
        )
    if np.any(I_rel <= 0.0):
        raise ValueError("Elementary price relatives must be strictly positive")

    sum_w0 = float(np.sum(w0_arr))
    sum_wt = float(np.sum(wt_arr))
    if sum_w0 <= 0.0 or sum_wt <= 0.0:
        raise ValueError("Weight vectors must sum to strictly positive values")

    # Monopoly reduction optimization
    if I_rel.size == 1:
        return float(I_rel[0])

    # Normalize weights over active carrier subset
    w0_norm = w0_arr / sum_w0
    wt_norm = wt_arr / sum_wt
    w_bar = 0.5 * (w0_norm + wt_norm)

    ln_PT = float(np.sum(w_bar * np.log(I_rel)))
    return float(np.exp(ln_PT))


def aggregate_route_tornqvist(
    elementary_map: Dict[Any, Any],
    sector: str,
    advance_window: int,
    carrier_weights_base: Optional[Dict[str, float]] = None,
    carrier_weights_curr: Optional[Dict[str, float]] = None
) -> Optional[TornqvistRouteResult]:
    """
    Identifies active carriers C^{0 cap t} for (sector, advance_window),
    normalizes carrier market shares over the active set, and computes the Törnqvist index.

    Returns:
        TornqvistRouteResult if at least one active carrier is present, or None otherwise.
    """
    sec = sector.upper().strip()
    if carrier_weights_base is None:
        carrier_weights_base = load_carrier_market_shares()
    if carrier_weights_curr is None:
        carrier_weights_curr = dict(carrier_weights_base)

    active_carriers: List[str] = []
    carrier_indices: Dict[str, float] = {}
    w0_raw: List[float] = []
    wt_raw: List[float] = []
    missing_carriers: List[str] = []

    all_known_carriers = ["6E", "AI", "IX", "QP", "SG"]

    # Also discover any additional carriers in elementary_map for this sector-window
    for k in elementary_map.keys():
        if isinstance(k, tuple) and len(k) == 3:
            s_cand, c_cand, w_cand = k
            if str(s_cand).upper().strip() == sec and int(w_cand) == advance_window:
                c_code = str(c_cand).upper().strip()
                if c_code not in all_known_carriers:
                    all_known_carriers.append(c_code)

    for carr in all_known_carriers:
        key = (sec, carr, advance_window)
        res = elementary_map.get(key)
        if res is not None:
            val = getattr(res, "index_value", None)
            if val is None and isinstance(res, (int, float)):
                val = float(res)
            elif val is None and isinstance(res, dict) and "index_value" in res:
                val = float(res["index_value"])

            if val is not None and val > 0.0:
                active_carriers.append(carr)
                carrier_indices[carr] = float(val)
                w0_raw.append(carrier_weights_base.get(carr, 0.05))
                wt_raw.append(carrier_weights_curr.get(carr, 0.05))
            else:
                missing_carriers.append(carr)
        else:
            missing_carriers.append(carr)

    if not active_carriers:
        return None

    pt_val = compute_tornqvist_index(
        elementary_indices=[carrier_indices[c] for c in active_carriers],
        w0=w0_raw,
        wt=wt_raw,
    )

    sum_w0 = sum(w0_raw)
    sum_wt = sum(wt_raw)
    w0_norm_dict = {c: round(w / sum_w0, 6) for c, w in zip(active_carriers, w0_raw)}
    wt_norm_dict = {c: round(w / sum_wt, 6) for c, w in zip(active_carriers, wt_raw)}
    w_bar_dict = {c: round(0.5 * (w0_norm_dict[c] + wt_norm_dict[c]), 6) for c in active_carriers}

    return TornqvistRouteResult(
        sector=sec,
        advance_window=advance_window,
        active_carriers=active_carriers,
        carrier_indices={c: round(v, 6) for c, v in carrier_indices.items()},
        carrier_weights_base=w0_norm_dict,
        carrier_weights_curr=wt_norm_dict,
        carrier_average_weights=w_bar_dict,
        index_value=round(pt_val, 6),
        is_monopoly=(len(active_carriers) == 1),
        missing_carriers=missing_carriers,
    )


def compute_all_route_indices(
    elementary_map: Dict[Any, Any],
    sectors: Optional[Sequence[str]] = None,
    advance_windows: Optional[Sequence[int]] = None,
    carrier_weights_base: Optional[Dict[str, float]] = None,
    carrier_weights_curr: Optional[Dict[str, float]] = None
) -> Dict[Tuple[str, int], TornqvistRouteResult]:
    """
    Computes Stage 2 Törnqvist index for all (sector, advance_window) pairs.
    Returns mapping: (sector, advance_window) -> TornqvistRouteResult.
    """
    if advance_windows is None:
        advance_windows = [1, 7, 15, 30, 45]

    if sectors is None:
        found_sectors = set()
        for k in elementary_map.keys():
            if isinstance(k, tuple) and len(k) >= 1:
                found_sectors.add(str(k[0]).upper().strip())
        sectors = sorted(found_sectors)

    results: Dict[Tuple[str, int], TornqvistRouteResult] = {}
    for sec in sectors:
        for win in advance_windows:
            res = aggregate_route_tornqvist(
                elementary_map=elementary_map,
                sector=sec,
                advance_window=win,
                carrier_weights_base=carrier_weights_base,
                carrier_weights_curr=carrier_weights_curr,
            )
            if res is not None:
                results[(sec, win)] = res

    return results


class TornqvistEngine:
    """
    Object-oriented engine wrapper for Stage 2 Superlative Törnqvist Route Index.
    """

    @staticmethod
    def compute_index(
        elementary_indices: Union[np.ndarray, Sequence[float]],
        w0: Union[np.ndarray, Sequence[float]],
        wt: Union[np.ndarray, Sequence[float]]
    ) -> float:
        """Computes Törnqvist superlative index for arbitrary arrays."""
        return compute_tornqvist_index(elementary_indices, w0, wt)

    @staticmethod
    def load_market_shares(weights_path: Optional[Union[str, Path]] = None) -> Dict[str, float]:
        """Loads market shares from reference JSON."""
        return load_carrier_market_shares(weights_path)

    @staticmethod
    def compute_route_index(
        elementary_map: Dict[Any, Any],
        sector: str,
        advance_window: int,
        carrier_weights_base: Optional[Dict[str, float]] = None,
        carrier_weights_curr: Optional[Dict[str, float]] = None
    ) -> Optional[TornqvistRouteResult]:
        """Aggregates active carriers for a single sector and window."""
        return aggregate_route_tornqvist(
            elementary_map=elementary_map,
            sector=sector,
            advance_window=advance_window,
            carrier_weights_base=carrier_weights_base,
            carrier_weights_curr=carrier_weights_curr,
        )

    @staticmethod
    def compute_all_routes(
        elementary_map: Dict[Any, Any],
        sectors: Optional[Sequence[str]] = None,
        advance_windows: Optional[Sequence[int]] = None,
        carrier_weights_base: Optional[Dict[str, float]] = None,
        carrier_weights_curr: Optional[Dict[str, float]] = None
    ) -> Dict[Tuple[str, int], TornqvistRouteResult]:
        """Aggregates all sector-window pairs."""
        return compute_all_route_indices(
            elementary_map=elementary_map,
            sectors=sectors,
            advance_windows=advance_windows,
            carrier_weights_base=carrier_weights_base,
            carrier_weights_curr=carrier_weights_curr,
        )
