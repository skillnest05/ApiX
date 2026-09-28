"""
Stage 3: National Modified Laspeyres Price Index Aggregator (Module C).
Combines Stage 2 Törnqvist route-level superlative indices across 30 directional
DGCA sectors and 5 advance booking windows using fixed empirical passenger traffic weights.

Formula:
    APIx^t = [ sum_{r=1}^{30} sum_{w=1}^{5} omega_{r,w} * P_T^{0,t}(r, w) ] * 100

Axiomatic Invariants:
    1. Strict Unit Normalization: sum_{r,w} omega_{r,w} == 1.0000000 +- 1e-7
    2. Base Period Scaling: In base period, APIx == 100.000000
    3. Homogeneity of Degree 1: Uniform price shock lambda scales APIx by lambda
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# 1. PYDANTIC SCHEMAS & DATA MODELS
# ==============================================================================

class Stage3Result(BaseModel):
    """Result of Stage 3 National Laspeyres Index computation."""
    model_config = ConfigDict(frozen=True)

    apix_index: float = Field(..., description="National APIx Index value (e.g. 104.85)")
    raw_composite_relative: float = Field(..., description="Composite price relative before 100x scaling (e.g. 1.0485)")
    base_value: float = Field(default=100.0, description="Base period index value (100.0)")
    total_observed_weight: float = Field(..., description="Sum of weights for observed cells (1.0 if complete)")
    is_fully_observed: bool = Field(..., description="True if all 150 cells were observed")
    missing_cells_count: int = Field(default=0, ge=0, description="Count of unobserved cells")
    sector_contributions: Dict[str, float] = Field(..., description="Sector-wise additive contributions to index")
    window_contributions: Dict[str, float] = Field(..., description="Window-wise additive contributions to index")
    tier_contributions: Dict[str, float] = Field(..., description="Route-tier additive contributions to index")


# ==============================================================================
# 2. NATIONAL WEIGHT MATRIX CONTAINER
# ==============================================================================

class NationalWeightMatrix:
    """
    Manages the 30-sector x 5-window fixed weight matrix Omega in R^{30 x 5}.
    omega_{r,w} = (PAX_r * alpha_w) / sum_{r',w'} (PAX_{r'} * alpha_{w'})
    Guarantees strict normalization sum_{r,w} omega_{r,w} == 1.0000000 +- 1e-7.
    """

    DEFAULT_WINDOWS = [1, 7, 15, 30, 45]
    WINDOW_STR_TO_DAYS = {
        "T+1": 1, "T+7": 7, "T+15": 15, "T+30": 30, "T+45": 45,
        "1": 1, "7": 7, "15": 15, "30": 30, "45": 45,
        "T_1": 1, "T_7": 7, "T_15": 15, "T_30": 30, "T_45": 45,
    }

    def __init__(
        self,
        dgca_path: Optional[Union[str, Path]] = None,
        booking_path: Optional[Union[str, Path]] = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent.parent
        self.dgca_path = Path(dgca_path) if dgca_path else base_dir / "data" / "dgca_weights.json"
        self.booking_path = Path(booking_path) if booking_path else base_dir / "data" / "booking_distribution.json"

        self.sectors: List[str] = []
        self.sector_metadata: Dict[str, Dict[str, Any]] = {}
        self.sector_weights: Dict[str, float] = {}
        self.window_weights: Dict[int, float] = {}
        self.matrix: Dict[Tuple[str, int], float] = {}
        self._numpy_matrix: Optional[np.ndarray] = None

        self._load_and_build()

    def _load_and_build(self) -> None:
        """Loads reference JSON files, computes composite weights, and verifies normalization."""
        if not self.dgca_path.exists():
            raise FileNotFoundError(f"DGCA weights file not found: {self.dgca_path}")
        if not self.booking_path.exists():
            raise FileNotFoundError(f"Booking distribution file not found: {self.booking_path}")

        with open(self.dgca_path, "r", encoding="utf-8") as f:
            dgca_data = json.load(f)
        with open(self.booking_path, "r", encoding="utf-8") as f:
            booking_data = json.load(f)

        # 1. Parse Sectors
        raw_sectors = dgca_data.get("sectors", {})
        if len(raw_sectors) != 30:
            raise ValueError(f"Expected exactly 30 sectors in DGCA reference, found {len(raw_sectors)}")

        self.sectors = sorted(list(raw_sectors.keys()))
        raw_sector_weights = {}
        for s_code in self.sectors:
            s_info = raw_sectors[s_code]
            self.sector_metadata[s_code] = s_info
            raw_sector_weights[s_code] = float(s_info.get("weight", 0.0))

        # Normalize sector weights to exactly 1.0
        sum_sw = sum(raw_sector_weights.values())
        if sum_sw <= 0:
            raise ValueError("Sum of sector weights must be strictly positive")
        self.sector_weights = {s: w / sum_sw for s, w in raw_sector_weights.items()}

        # 2. Parse Advance Booking Windows
        dist = booking_data.get("distribution", {})
        raw_window_weights = {}
        for w_key, w_weight in dist.items():
            days = self.WINDOW_STR_TO_DAYS.get(str(w_key).upper().strip())
            if days is not None:
                raw_window_weights[days] = float(w_weight)

        if len(raw_window_weights) != 5:
            # Fallback to days_distribution
            days_dist = booking_data.get("days_distribution", {})
            for d_str, w_weight in days_dist.items():
                raw_window_weights[int(d_str)] = float(w_weight)

        sum_ww = sum(raw_window_weights.values())
        if sum_ww <= 0:
            raise ValueError("Sum of booking window weights must be strictly positive")
        self.window_weights = {d: w / sum_ww for d, w in raw_window_weights.items()}

        # 3. Construct Composite Matrix Omega in R^{30 x 5}
        self.matrix = {}
        for s in self.sectors:
            for w in self.DEFAULT_WINDOWS:
                self.matrix[(s, w)] = self.sector_weights[s] * self.window_weights[w]

        # Verify strict normalization
        total_sum = sum(self.matrix.values())
        if abs(total_sum - 1.0) > 1e-7:
            raise ValueError(f"Weight matrix normalization identity violated: sum = {total_sum}")

        # Build numpy array (30, 5)
        w_r_vec = np.array([self.sector_weights[s] for s in self.sectors], dtype=float)
        alpha_vec = np.array([self.window_weights[w] for w in self.DEFAULT_WINDOWS], dtype=float)
        self._numpy_matrix = np.outer(w_r_vec, alpha_vec)

    def get_weight(self, sector: str, window: Union[int, str]) -> float:
        """Retrieves composite cell weight omega_{r,w}."""
        w_days = self.WINDOW_STR_TO_DAYS.get(str(window).upper().strip(), int(window) if str(window).isdigit() else 15)
        return self.matrix.get((sector.upper().strip(), w_days), 0.0)

    def to_numpy(self) -> np.ndarray:
        """Returns 30x5 NumPy weight matrix."""
        assert self._numpy_matrix is not None
        return self._numpy_matrix.copy()

    def get_tier(self, sector: str) -> str:
        """Returns route tier for sector (Trunk, High-Density, Regional, Seasonal)."""
        return self.sector_metadata.get(sector.upper().strip(), {}).get("route_tier", "Trunk")


# Singleton default weight matrix
_DEFAULT_WEIGHT_MATRIX: Optional[NationalWeightMatrix] = None


def get_default_weight_matrix() -> NationalWeightMatrix:
    """Returns singleton default NationalWeightMatrix."""
    global _DEFAULT_WEIGHT_MATRIX
    if _DEFAULT_WEIGHT_MATRIX is None:
        _DEFAULT_WEIGHT_MATRIX = NationalWeightMatrix()
    return _DEFAULT_WEIGHT_MATRIX


def load_national_weights(
    dgca_path: Optional[Union[str, Path]] = None,
    booking_path: Optional[Union[str, Path]] = None
) -> NationalWeightMatrix:
    """Factory loader for NationalWeightMatrix."""
    return NationalWeightMatrix(dgca_path=dgca_path, booking_path=booking_path)


# ==============================================================================
# 3. STAGE 3 CALCULATION FUNCTION
# ==============================================================================

def calculate_national_laspeyres(
    route_window_indices: Union[
        Dict[Tuple[str, int], float],
        Dict[Tuple[str, str], float],
        np.ndarray,
        Sequence[Any]
    ],
    weight_matrix: Optional[NationalWeightMatrix] = None,
    impute_missing: bool = True,
    fallback_index: float = 1.0,
) -> Stage3Result:
    """
    Computes Stage 3 National Modified Laspeyres Price Index:
        APIx^t = [ sum_{r,w} omega_{r,w} * P_T^{0,t}(r,w) ] * 100

    Args:
        route_window_indices: Dictionary of (sector, window_days) -> index,
                              NumPy array of shape (30, 5), or sequence of TornqvistRouteResult.
        weight_matrix: Optional custom NationalWeightMatrix.
        impute_missing: If True, renormalizes over observed cells.
        fallback_index: Default relative for unobserved cells if not renormalizing.

    Returns:
        Stage3Result with national index, contributions, and metadata.
    """
    if weight_matrix is None:
        weight_matrix = get_default_weight_matrix()

    # Case 1: NumPy 2D array of shape (30, 5)
    if isinstance(route_window_indices, np.ndarray):
        arr = np.asarray(route_window_indices, dtype=float)
        if arr.shape != (30, 5):
            raise ValueError(f"NumPy array input must have shape (30, 5), got {arr.shape}")
        if np.any(arr <= 0):
            raise ValueError("All price index relatives in NumPy array must be strictly positive")

        omega = weight_matrix.to_numpy()
        raw_rel = float(np.sum(omega * arr))
        apix = raw_rel * 100.0

        sector_contribs = {}
        for i, s in enumerate(weight_matrix.sectors):
            sector_contribs[s] = round(float(np.sum(omega[i, :] * arr[i, :]) * 100.0), 6)

        window_contribs = {}
        for j, w in enumerate(weight_matrix.DEFAULT_WINDOWS):
            window_contribs[f"T+{w}"] = round(float(np.sum(omega[:, j] * arr[:, j]) * 100.0), 6)

        tier_contribs: Dict[str, float] = {}
        for s, c in sector_contribs.items():
            tier = weight_matrix.get_tier(s)
            tier_contribs[tier] = round(tier_contribs.get(tier, 0.0) + c, 6)

        return Stage3Result(
            apix_index=round(apix, 4),
            raw_composite_relative=round(raw_rel, 6),
            total_observed_weight=1.0,
            is_fully_observed=True,
            missing_cells_count=0,
            sector_contributions=sector_contribs,
            window_contributions=window_contribs,
            tier_contributions=tier_contribs,
        )

    # Case 2: Dictionary or Sequence input
    cell_indices: Dict[Tuple[str, int], float] = {}

    if isinstance(route_window_indices, dict):
        for k, v in route_window_indices.items():
            if isinstance(k, tuple) and len(k) == 2:
                s, w = k
                w_days = weight_matrix.WINDOW_STR_TO_DAYS.get(str(w).upper().strip(), int(w) if str(w).isdigit() else 15)
                val = getattr(v, "index_value", v)
                if isinstance(val, dict) and "index_value" in val:
                    val = val["index_value"]
                val = float(val)
                if val <= 0:
                    raise ValueError(f"Price relative for cell ({s}, {w}) must be strictly positive, got {val}")
                cell_indices[(str(s).upper().strip(), w_days)] = val
    else:
        # Sequence of objects (e.g. TornqvistRouteResult)
        for item in route_window_indices:
            s = getattr(item, "sector", None)
            w = getattr(item, "advance_window", None)
            val = getattr(item, "index_value", None)
            if s is not None and w is not None and val is not None:
                w_days = weight_matrix.WINDOW_STR_TO_DAYS.get(str(w).upper().strip(), int(w) if str(w).isdigit() else 15)
                f_val = float(val)
                if f_val <= 0:
                    raise ValueError(f"Price relative for cell ({s}, {w}) must be strictly positive, got {f_val}")
                cell_indices[(str(s).upper().strip(), w_days)] = f_val

    if len(cell_indices) == 0:
        raise ValueError("Cannot calculate Stage 3 index from empty route-window dataset")

    # Evaluate completeness & weights
    total_observed_weight = 0.0
    missing_cells = []
    weighted_sum = 0.0
    sector_contribs = {s: 0.0 for s in weight_matrix.sectors}
    window_contribs = {f"T+{w}": 0.0 for w in weight_matrix.DEFAULT_WINDOWS}
    tier_contribs = {}

    for s in weight_matrix.sectors:
        for w in weight_matrix.DEFAULT_WINDOWS:
            cell_key = (s, w)
            omega_rw = weight_matrix.matrix[cell_key]

            if cell_key in cell_indices:
                idx_val = cell_indices[cell_key]
                total_observed_weight += omega_rw
                contrib = omega_rw * idx_val * 100.0
                weighted_sum += contrib
                sector_contribs[s] += contrib
                window_contribs[f"T+{w}"] += contrib
                tier = weight_matrix.get_tier(s)
                tier_contribs[tier] = tier_contribs.get(tier, 0.0) + contrib
            else:
                missing_cells.append(cell_key)

    is_complete = len(missing_cells) == 0

    # Renormalization over observed cells if partially observed
    if not is_complete and impute_missing:
        if total_observed_weight <= 0:
            raise ValueError("Observed weight is zero; cannot compute index")

        scale_factor = 1.0 / total_observed_weight
        weighted_sum *= scale_factor
        sector_contribs = {s: round(c * scale_factor, 6) for s, c in sector_contribs.items()}
        window_contribs = {w: round(c * scale_factor, 6) for w, c in window_contribs.items()}
        tier_contribs = {t: round(c * scale_factor, 6) for t, c in tier_contribs.items()}
    else:
        sector_contribs = {s: round(c, 6) for s, c in sector_contribs.items()}
        window_contribs = {w: round(c, 6) for w, c in window_contribs.items()}
        tier_contribs = {t: round(c, 6) for t, c in tier_contribs.items()}

    raw_rel = weighted_sum / 100.0

    return Stage3Result(
        apix_index=round(weighted_sum, 4),
        raw_composite_relative=round(raw_rel, 6),
        total_observed_weight=round(total_observed_weight, 7),
        is_fully_observed=is_complete,
        missing_cells_count=len(missing_cells),
        sector_contributions=sector_contribs,
        window_contributions=window_contribs,
        tier_contributions=tier_contribs,
    )


class LaspeyresEngine:
    """
    Object-oriented engine wrapper for Stage 3 National Modified Laspeyres Index.
    """

    def __init__(self, weight_matrix: Optional[NationalWeightMatrix] = None):
        self.weight_matrix = weight_matrix or get_default_weight_matrix()

    def get_weight_matrix(self) -> NationalWeightMatrix:
        """Returns active NationalWeightMatrix."""
        return self.weight_matrix

    def compute_national_index(
        self,
        route_window_indices: Union[
            Dict[Tuple[str, int], float],
            Dict[Tuple[str, str], float],
            np.ndarray,
            Sequence[Any]
        ],
        impute_missing: bool = True,
        fallback_index: float = 1.0,
    ) -> Stage3Result:
        """Computes national APIx composite index."""
        return calculate_national_laspeyres(
            route_window_indices=route_window_indices,
            weight_matrix=self.weight_matrix,
            impute_missing=impute_missing,
            fallback_index=fallback_index,
        )
