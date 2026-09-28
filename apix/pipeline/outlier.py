"""
Outlier Filtering & Plausibility Engine for APIx Pipeline (Module B).
Implements:
1. Statistical IQR Cohort Filter (Q1 - 1.5*IQR to Q3 + 1.5*IQR per homogeneous cohort)
2. Domain Floor Guard (reject fares < 999 INR)
3. Domain Ceiling Guard (reject domestic economy fares > 25,000 INR)
4. Temporal Jump Detector (>300% day-over-day price surge)
5. Cross-Source Median Variance Check (>50% departure from cohort median)
6. Multi-Source Consensus Scoring (Kendall's W Concordance)
"""

from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from apix.pipeline.schemas import OutlierEvaluation, OutlierReason


# ==============================================================================
# STATISTICAL IQR COHORT FILTER
# ==============================================================================

def filter_iqr_bounds(fares: Union[np.ndarray, List[float]]) -> Tuple[float, float]:
    """
    Computes statistical Interquartile Range (IQR) bounds [LB, UB] for a homogeneous cohort:
    LB = max(999.0, Q1 - 1.5 * IQR)
    UB = min(25000.0, Q3 + 1.5 * IQR)

    Args:
        fares: Array-like collection of numeric fares in the cohort.

    Returns:
        Tuple of (lower_bound, upper_bound) in INR.
    """
    arr = np.asarray(fares, dtype=float)
    if len(arr) == 0:
        return 999.0, 25000.0

    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1

    if iqr < 1.0:
        # Zero-IQR protection: uniform or near-uniform fare cohorts.
        # Expand bounds by +/- 15% around the median to prevent false outlier rejection.
        med = float(np.median(arr))
        lb = max(999.0, med * 0.85)
        ub = min(25000.0, med * 1.15)
        return float(min(lb, ub)), float(max(lb, ub))

    lb = max(999.0, q1 - 1.5 * iqr)
    ub = min(25000.0, q3 + 1.5 * iqr)
    return float(lb), float(ub)


# ==============================================================================
# DOMAIN BOUNDARY GUARDS
# ==============================================================================

def is_domain_floor_violation(fare: Union[float, Decimal]) -> bool:
    """
    Checks if total fare falls below the Indian aviation statutory minimum floor (₹999).
    Fares < ₹999 indicate promotional coupon glitches, infant fare leaks, or zero base fare errors.
    """
    val = float(fare)
    return val < 999.0


def is_domain_ceiling_violation(fare: Union[float, Decimal]) -> bool:
    """
    Checks if domestic economy fare exceeds the statutory maximum ceiling (₹25,000).
    Fares > ₹25,000 on domestic sectors indicate Business Class cabin leaks or international transit legs.
    """
    val = float(fare)
    return val > 25000.0


# ==============================================================================
# TEMPORAL JUMP & CROSS-SOURCE MEDIAN CHECKS
# ==============================================================================

def is_temporal_surge_violation(
    prev_fare: Union[float, Decimal],
    curr_fare: Union[float, Decimal],
    threshold_pct: float = 300.0
) -> bool:
    """
    Flags price jumps exceeding threshold_pct (default 300.0%) from previous day's fare
    for the identical flight instance and travel date.
    Strictly flags if pct_change > threshold_pct.
    """
    p_prev = float(prev_fare)
    p_curr = float(curr_fare)
    if p_prev <= 0.0:
        return True
    pct_change = ((p_curr - p_prev) / p_prev) * 100.0
    return pct_change > threshold_pct


def is_cross_source_variance_violation(
    source_fare: Union[float, Decimal],
    cohort_median: Union[float, Decimal],
    threshold_pct: float = 50.0
) -> bool:
    """
    Flags quotes deviating by >50.0% from the cross-source median fare for the same flight.
    Strictly flags if dev_pct > threshold_pct.
    """
    p_src = float(source_fare)
    p_med = float(cohort_median)
    if p_med <= 0.0:
        return True
    dev_pct = (abs(p_src - p_med) / p_med) * 100.0
    return dev_pct > threshold_pct


# ==============================================================================
# MULTI-SOURCE CONSENSUS SCORING (KENDALL'S W)
# ==============================================================================

def compute_kendall_w(rankings_matrix: np.ndarray) -> float:
    """
    Computes Kendall's Coefficient of Concordance (W) across multiple sources:
    W = 12 * sum( (R_i - R_bar)^2 ) / ( m^2 * (n^3 - n) )
    where:
        m = number of raters/sources (rows)
        n = number of subjects/flights (columns)
    """
    mat = np.asarray(rankings_matrix, dtype=float)
    if mat.ndim != 2:
        return 1.0
    m, n = mat.shape
    if n <= 1 or m < 1:
        return 1.0

    r_sums = np.sum(mat, axis=0)
    r_bar = np.mean(r_sums)
    s = np.sum((r_sums - r_bar) ** 2)
    denom = (m ** 2) * (n ** 3 - n)
    if denom == 0:
        return 1.0

    w = (12.0 * s) / denom
    return float(np.clip(w, 0.0, 1.0))


# ==============================================================================
# COMPREHENSIVE OUTLIER EVALUATOR
# ==============================================================================

class OutlierDetector:
    """
    Multi-criteria outlier evaluation engine combining domain rules,
    cohort statistical IQR, temporal surge checks, and cross-source consistency.
    """

    def __init__(
        self,
        floor_limit: float = 999.0,
        ceiling_limit: float = 25000.0,
        temporal_surge_threshold: float = 300.0,
        cross_source_threshold: float = 50.0,
        min_cohort_sample_size: int = 5,
    ):
        self.floor_limit = floor_limit
        self.ceiling_limit = ceiling_limit
        self.temporal_surge_threshold = temporal_surge_threshold
        self.cross_source_threshold = cross_source_threshold
        self.min_cohort_sample_size = min_cohort_sample_size

    def evaluate_fare(
        self,
        fare: Union[float, Decimal],
        cohort_fares: Optional[List[float]] = None,
        prev_fare: Optional[float] = None,
        cohort_median: Optional[float] = None,
    ) -> OutlierEvaluation:
        """
        Runs comprehensive outlier evaluation on a single fare amount.
        """
        val = float(fare)
        flags: List[str] = []
        is_outlier = False
        primary_reason: Optional[str] = None
        lb, ub = None, None
        temporal_pct = None
        cross_dev_pct = None

        # 1. Domain Floor Check
        if is_domain_floor_violation(val):
            is_outlier = True
            flags.append("DOMAIN_FLOOR")
            if not primary_reason:
                primary_reason = OutlierReason.DOMAIN_FLOOR.value

        # 2. Domain Ceiling Check
        if is_domain_ceiling_violation(val):
            is_outlier = True
            flags.append("DOMAIN_CEILING")
            if not primary_reason:
                primary_reason = OutlierReason.DOMAIN_CEILING.value

        # 3. Statistical IQR Cohort Check
        if cohort_fares is not None and len(cohort_fares) >= self.min_cohort_sample_size:
            lb, ub = filter_iqr_bounds(cohort_fares)
            if val < lb or val > ub:
                is_outlier = True
                flags.append("STATISTICAL_IQR")
                if not primary_reason:
                    primary_reason = OutlierReason.STATISTICAL_IQR.value

        # 4. Temporal Surge Check
        if prev_fare is not None and prev_fare > 0.0:
            temporal_pct = ((val - prev_fare) / prev_fare) * 100.0
            if is_temporal_surge_violation(prev_fare, val, self.temporal_surge_threshold):
                is_outlier = True
                flags.append("TEMPORAL_SURGE")
                if not primary_reason:
                    primary_reason = OutlierReason.TEMPORAL_SURGE.value

        # 5. Cross-Source Median Variance Check
        if cohort_median is not None and cohort_median > 0.0:
            cross_dev_pct = (abs(val - cohort_median) / cohort_median) * 100.0
            if is_cross_source_variance_violation(val, cohort_median, self.cross_source_threshold):
                is_outlier = True
                flags.append("CROSS_SOURCE_VARIANCE")
                if not primary_reason:
                    primary_reason = OutlierReason.CROSS_SOURCE_VARIANCE.value

        return OutlierEvaluation(
            is_outlier=is_outlier,
            outlier_reason=primary_reason if is_outlier else None,
            flags=flags,
            iqr_lower_bound=lb,
            iqr_upper_bound=ub,
            temporal_pct_change=temporal_pct,
            cross_source_deviation_pct=cross_dev_pct,
        )


def evaluate_outlier_status(
    fare: Union[float, Decimal],
    cohort_fares: Optional[List[float]] = None,
    prev_fare: Optional[float] = None,
    cohort_median: Optional[float] = None,
) -> OutlierEvaluation:
    """Convenience module function for evaluating outlier status."""
    detector = OutlierDetector()
    return detector.evaluate_fare(
        fare=fare,
        cohort_fares=cohort_fares,
        prev_fare=prev_fare,
        cohort_median=cohort_median,
    )
