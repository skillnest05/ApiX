"""
Multilateral Rolling Year GEKS (RYGEKS / CCDI) Engine with Mean Splice (Module C).
Eliminates high-frequency chain drift in daily/weekly/monthly airfare price series
while providing strict historical non-revisability required by NSO and RBI.

Formulas:
    Multilateral GEKS (CCDI):
        P_{GEKS}^{0,t} = prod_{k in W} ( P_T^{k,t} / P_T^{k,0} )^{1/T}
    Mean Splice Linking:
        ln P_{published}^t = ln g_{T-1} + (1 / (T-1)) * sum_{j=0}^{T-2} ( ln P_{published}^{t-T+1+j} - ln g_j )

Axiomatic Invariants:
    1. Multilateral Transitivity: P^{a,b} * P^{b,c} == P^{a,c} +- 1e-6
    2. Zero Chain Drift: Cyclic price loop (P0 -> P1 -> P0) returns strictly 1.000000
    3. Non-Revisability: Published historical values are immutable
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


# ==============================================================================
# 1. PYDANTIC SCHEMAS & DATA MODELS
# ==============================================================================

class RYGEKSResult(BaseModel):
    """Result of a single RYGEKS publication period update."""
    model_config = ConfigDict(frozen=True)

    period_id: Any = Field(..., description="Date or period identifier")
    published_index: float = Field(..., description="Non-revisable spliced index value (e.g. 104.85)")
    window_length: int = Field(..., description="Active window length T used in multilateral computation")
    splice_method: str = Field(..., description="Splicing method applied ('mean', 'movement', 'window', 'half', 'base')")
    level_adjustment_factor: float = Field(..., description="Splice scale factor exp(C)")
    is_expanding_window: bool = Field(..., description="True if initial window is expanding (t < T)")


# ==============================================================================
# 2. CORE MATHEMATICAL GEKS FUNCTIONS
# ==============================================================================

def compute_geks_vector(bilateral_matrix: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
    """
    Computes transitive multilateral GEKS (CCDI) index vector for window of length T:
        P_{GEKS}^{0,t} = prod_{k=0}^{T-1} ( P[k, t] / P[k, 0] )^{1/T}

    Evaluated in natural log-space for numerical stability:
        ln g_t = (1 / T) * sum_{k=0}^{T-1} ( ln P[k, t] - ln P[k, 0] )

    Args:
        bilateral_matrix: T x T square matrix where P[j, k] is price relative from period j to k.

    Returns:
        1D NumPy array of length T with multilateral indices normalized to g[0] = 1.000000.
    """
    P_T = np.asarray(bilateral_matrix, dtype=float)
    if P_T.ndim != 2 or P_T.shape[0] != P_T.shape[1]:
        raise ValueError(f"Bilateral matrix must be a square 2D matrix, got shape {P_T.shape}")

    T = P_T.shape[0]
    if T == 0:
        raise ValueError("Bilateral matrix cannot be empty")
    if T == 1:
        return np.array([1.0], dtype=float)
    if np.any(P_T <= 0.0):
        raise ValueError("All elements in bilateral matrix must be strictly positive")

    ln_P = np.log(P_T)

    # ln g_t = (1/T) * sum_k [ ln P[k, t] - ln P[k, 0] ]
    ln_geks = np.zeros(T, dtype=float)
    for t in range(T):
        diff = ln_P[:, t] - ln_P[:, 0]
        ln_geks[t] = np.mean(diff)

    geks_vector = np.exp(ln_geks)
    # Ensure exact machine normalization at base
    geks_vector[0] = 1.0
    return geks_vector


def compute_geks_pairwise(
    bilateral_dict: Dict[Tuple[Any, Any], float],
    periods: Sequence[Any],
    base_period: Any,
    current_period: Any,
) -> float:
    """
    Computes bilateral GEKS relative P_{GEKS}^{base, current} over period set.
    """
    p_list = list(periods)
    T = len(p_list)
    if T == 0:
        raise ValueError("Periods sequence cannot be empty")

    log_sum = 0.0
    for k in p_list:
        P_k_curr = bilateral_dict.get((k, current_period), 1.0)
        P_k_base = bilateral_dict.get((k, base_period), 1.0)
        if P_k_curr <= 0.0 or P_k_base <= 0.0:
            raise ValueError("Bilateral relatives must be strictly positive")
        log_sum += (math.log(P_k_curr) - math.log(P_k_base))

    return math.exp(log_sum / T)


# ==============================================================================
# 3. SPLICING METHODS
# ==============================================================================

def splice_movement(
    published_series: Dict[Any, float],
    window_geks: np.ndarray,
    window_periods: Sequence[Any],
) -> float:
    """Movement Splice: Links onto adjacent period t-1."""
    prev_period = window_periods[-2]
    return float(published_series[prev_period] * (window_geks[-1] / window_geks[-2]))


def splice_window(
    published_series: Dict[Any, float],
    window_geks: np.ndarray,
    window_periods: Sequence[Any],
) -> float:
    """Window Splice: Links onto oldest period in window t-T+1."""
    oldest_period = window_periods[0]
    return float(published_series[oldest_period] * (window_geks[-1] / window_geks[0]))


def splice_half(
    published_series: Dict[Any, float],
    window_geks: np.ndarray,
    window_periods: Sequence[Any],
) -> float:
    """Half Splice: Geometric mean of Movement Splice and Window Splice."""
    p_ms = splice_movement(published_series, window_geks, window_periods)
    p_ws = splice_window(published_series, window_geks, window_periods)
    return float(math.sqrt(p_ms * p_ws))


def splice_mean(
    published_series: Dict[Any, float],
    window_geks: np.ndarray,
    window_periods: Sequence[Any],
) -> Tuple[float, float]:
    """
    Mean Splice (Diewert & Fox, 2022):
    Unweighted geometric mean of all T-1 possible links across overlapping periods.

    Returns:
        Tuple of (spliced_index_value, level_adjustment_factor)
    """
    T = len(window_periods)
    overlap_count = T - 1
    if overlap_count <= 0:
        return float(published_series[window_periods[0]]), 1.0

    # ln P_t = ln g_{T-1} + (1 / (T-1)) * sum_{j=0}^{T-2} [ ln P_pub[k] - ln g_j ]
    log_diffs = []
    for j in range(overlap_count):
        hist_p = window_periods[j]
        pub_val = published_series[hist_p]
        log_diff = math.log(pub_val) - math.log(window_geks[j])
        log_diffs.append(log_diff)

    mean_log_diff = float(np.mean(log_diffs))
    adj_factor = math.exp(mean_log_diff)
    spliced_val = float(window_geks[-1] * adj_factor)

    return spliced_val, adj_factor


# ==============================================================================
# 4. RYGEKS STATEFUL ENGINE
# ==============================================================================

class RYGEKSEngine:
    """
    Stateful Rolling Year GEKS Engine.
    Maintains chronological rolling window, computes multilateral GEKS, and links
    new periods via Mean Splice to produce an immutable, non-revisable published series.
    """

    def __init__(
        self,
        window_length: int = 13,
        base_value: float = 100.0,
        splice_method: str = "mean",
    ):
        if window_length < 2:
            raise ValueError(f"Window length must be >= 2, got {window_length}")

        self.window_length = window_length
        self.base_value = float(base_value)
        self.splice_method = splice_method.lower().strip()

        self.periods: List[Any] = []
        self.published_series: Dict[Any, float] = {}
        self.bilateral_history: Dict[Tuple[Any, Any], float] = {}
        self.results_history: List[RYGEKSResult] = []

    def compute_multilateral_series(self, bilateral_matrix: Union[np.ndarray, Sequence[Sequence[float]]]) -> np.ndarray:
        """Computes multilateral GEKS index vector."""
        return compute_geks_vector(bilateral_matrix)

    def apply_mean_splice(
        self,
        published_series: Dict[Any, float],
        window_geks: np.ndarray,
        window_periods: Sequence[Any],
    ) -> Tuple[float, float]:
        """Applies Mean Splice linking."""
        return splice_mean(published_series, window_geks, window_periods)

    def add_first_period(self, period_id: Any) -> RYGEKSResult:
        """Initializes the time series with the baseline period (APIx = base_value)."""
        if len(self.periods) > 0:
            raise ValueError("add_first_period can only be called on an empty engine")

        self.periods.append(period_id)
        self.published_series[period_id] = self.base_value
        self.bilateral_history[(period_id, period_id)] = 1.0

        res = RYGEKSResult(
            period_id=period_id,
            published_index=self.base_value,
            window_length=1,
            splice_method="base",
            level_adjustment_factor=1.0,
            is_expanding_window=True,
        )
        self.results_history.append(res)
        return res

    def add_period(
        self,
        period_id: Any,
        pairwise_relatives: Dict[Any, float],
    ) -> RYGEKSResult:
        """
        Advances the rolling window with a new period and applies Mean Splice.

        Args:
            period_id: Identifier for the new period (date, string, integer).
            pairwise_relatives: Dict mapping existing historical periods in window
                                to the bilateral price relative from historical to new period:
                                P[hist, period_id].
        """
        if period_id in self.published_series:
            raise ValueError(f"Period {period_id} has already been published; cannot overwrite")
        if len(self.periods) == 0:
            return self.add_first_period(period_id)

        self.periods.append(period_id)
        self.bilateral_history[(period_id, period_id)] = 1.0

        for p_hist, rel in pairwise_relatives.items():
            f_rel = float(rel)
            if f_rel <= 0:
                raise ValueError(f"Bilateral price relative from {p_hist} to {period_id} must be positive, got {f_rel}")
            self.bilateral_history[(p_hist, period_id)] = f_rel
            self.bilateral_history[(period_id, p_hist)] = 1.0 / f_rel

        # Determine active window W_t
        active_w_len = min(len(self.periods), self.window_length)
        window_periods = self.periods[-active_w_len:]
        is_expanding = len(self.periods) < self.window_length

        # Construct bilateral matrix for window
        T = len(window_periods)
        P_T = np.zeros((T, T), dtype=float)
        for i, p_i in enumerate(window_periods):
            for j, p_j in enumerate(window_periods):
                if (p_i, p_j) in self.bilateral_history:
                    P_T[i, j] = self.bilateral_history[(p_i, p_j)]
                else:
                    # Indirect path via base of window
                    p_base = window_periods[0]
                    rel_base_j = self.bilateral_history.get((p_base, p_j), 1.0)
                    rel_base_i = self.bilateral_history.get((p_base, p_i), 1.0)
                    P_T[i, j] = rel_base_j / rel_base_i

        # Compute multilateral GEKS on window
        window_geks = compute_geks_vector(P_T)

        # Splice into published historical series
        if self.splice_method == "mean":
            spliced_val, adj_factor = splice_mean(self.published_series, window_geks, window_periods)
        elif self.splice_method == "movement":
            spliced_val = splice_movement(self.published_series, window_geks, window_periods)
            adj_factor = spliced_val / window_geks[-1]
        elif self.splice_method == "window":
            spliced_val = splice_window(self.published_series, window_geks, window_periods)
            adj_factor = spliced_val / window_geks[-1]
        elif self.splice_method == "half":
            spliced_val = splice_half(self.published_series, window_geks, window_periods)
            adj_factor = spliced_val / window_geks[-1]
        else:
            raise ValueError(f"Unknown splice method: {self.splice_method}")

        self.published_series[period_id] = round(spliced_val, 4)

        res = RYGEKSResult(
            period_id=period_id,
            published_index=round(spliced_val, 4),
            window_length=T,
            splice_method=self.splice_method,
            level_adjustment_factor=round(adj_factor, 6),
            is_expanding_window=is_expanding,
        )
        self.results_history.append(res)
        return res

    def calculate_series(
        self,
        periods: Sequence[Any],
        pairwise_relatives_map: Optional[Dict[Tuple[Any, Any], float]] = None,
        period_index_levels: Optional[Dict[Any, float]] = None,
    ) -> List[RYGEKSResult]:
        """
        Executes multilateral Rolling Year GEKS (RYGEKS) with Mean Splice linking
        across an ordered sequence of periods.
        """
        p_list = list(periods)
        if not p_list:
            return []

        if len(self.periods) == 0:
            self.add_first_period(p_list[0])

        start_idx = 1 if p_list[0] == self.periods[0] else 0

        for i in range(start_idx, len(p_list)):
            p_cur = p_list[i]
            if p_cur in self.published_series:
                continue

            active_hist = self.periods[-self.window_length:]
            pairwise: Dict[Any, float] = {}

            for p_h in active_hist:
                if pairwise_relatives_map and (p_h, p_cur) in pairwise_relatives_map:
                    pairwise[p_h] = float(pairwise_relatives_map[(p_h, p_cur)])
                elif period_index_levels and p_cur in period_index_levels and p_h in period_index_levels:
                    idx_cur = float(period_index_levels[p_cur])
                    idx_h = float(period_index_levels[p_h])
                    pairwise[p_h] = idx_cur / idx_h if idx_h > 0 else 1.0
                else:
                    pairwise[p_h] = 1.0

            self.add_period(p_cur, pairwise)

        return list(self.results_history)

    def get_published_series(self) -> Dict[Any, float]:
        """Returns copy of immutable published series."""
        return self.published_series.copy()


def calculate_rygeks_series(
    start_date: Union[date, str],
    end_date: Union[date, str],
    window_length: int = 13,
    base_value: float = 100.0,
    daily_indices: Optional[Dict[Any, float]] = None,
) -> List[RYGEKSResult]:
    """
    Public Module C contract:
        calculate_rygeks_series(start_date, end_date) -> List[RYGEKSResult]

    Generates or computes RYGEKS multilateral series for a date range.
    """
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()

    engine = RYGEKSEngine(window_length=window_length, base_value=base_value, splice_method="mean")

    # Generate daily or period steps
    cur_date = start_date
    date_list = []
    while cur_date <= end_date:
        date_list.append(cur_date.isoformat())
        cur_date += timedelta(days=1)

    if not date_list:
        return []

    # Initialize first period
    engine.add_first_period(date_list[0])

    for i in range(1, len(date_list)):
        p_cur = date_list[i]
        # Build pairwise relatives to recent periods in window
        active_hist = date_list[max(0, i - window_length):i]
        pairwise = {}

        if daily_indices and p_cur in daily_indices:
            cur_idx = daily_indices[p_cur]
            for p_h in active_hist:
                h_idx = daily_indices.get(p_h, base_value)
                pairwise[p_h] = cur_idx / h_idx
        else:
            # Deterministic simulation if empirical daily indices not supplied
            # Slight trend + day-of-week oscillation
            step = i
            base_rel = 1.0 + 0.001 * step + 0.003 * math.sin(2 * math.pi * step / 7)
            for j, p_h in enumerate(reversed(active_hist)):
                hist_step = step - 1 - j
                h_rel = 1.0 + 0.001 * hist_step + 0.003 * math.sin(2 * math.pi * hist_step / 7)
                pairwise[p_h] = base_rel / h_rel

        engine.add_period(p_cur, pairwise)

    return list(engine.results_history)
