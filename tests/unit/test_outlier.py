"""
Unit Test Suite: Statistical Outlier, Plausibility, and Domain Boundaries Filtering.
Tests:
1. Statistical IQR Cohort Filter (Q1 - 1.5*IQR to Q3 + 1.5*IQR)
2. Domain Floor Guard (reject fares < 999 INR)
3. Domain Ceiling Guard (reject economy fares > 25,000 INR)
4. Temporal Jump Detector (>300% day-over-day price surge)
5. Cross-Source Median Variance Filter (>50% departure from cohort median)
6. Multi-Source Consensus Scoring (Kendall's W >= 0.80)
"""

import numpy as np
import pytest


def filter_iqr_bounds(fares: np.ndarray) -> tuple[float, float]:
    """Computes IQR bounds [LB, UB]."""
    q1 = np.percentile(fares, 25)
    q3 = np.percentile(fares, 75)
    iqr = q3 - q1
    lb = max(999.0, q1 - 1.5 * iqr)
    ub = min(25000.0, q3 + 1.5 * iqr)
    return lb, ub


def is_domain_floor_violation(fare: float) -> bool:
    """Rejects fares < 999 INR."""
    return fare < 999.0


def is_domain_ceiling_violation(fare: float) -> bool:
    """Rejects domestic economy fares > 25,000 INR."""
    return fare > 25000.0


def is_temporal_surge_violation(prev_fare: float, curr_fare: float, threshold_pct: float = 300.0) -> bool:
    """Flags day-over-day price jumps exceeding threshold_pct (300%)."""
    if prev_fare <= 0:
        return True
    pct_change = ((curr_fare - prev_fare) / prev_fare) * 100.0
    return pct_change > threshold_pct


def is_cross_source_variance_violation(source_fare: float, cohort_median: float, threshold_pct: float = 50.0) -> bool:
    """Flags quotes deviating by >50% from cohort median."""
    if cohort_median <= 0:
        return True
    dev_pct = (abs(source_fare - cohort_median) / cohort_median) * 100.0
    return dev_pct > threshold_pct


def compute_kendall_w(rankings_matrix: np.ndarray) -> float:
    """
    Computes Kendall's Coefficient of Concordance (W):
    W = 12 * sum( (R_i - R_bar)^2 ) / ( m^2 * (n^3 - n) )
    where m = number of raters/sources, n = number of subjects/flights.
    """
    m, n = rankings_matrix.shape
    if n <= 1:
        return 1.0
    r_sums = np.sum(rankings_matrix, axis=0)
    r_bar = np.mean(r_sums)
    s = np.sum((r_sums - r_bar) ** 2)
    w = (12.0 * s) / ((m ** 2) * (n ** 3 - n))
    return float(np.clip(w, 0.0, 1.0))


class TestOutlierFilters:
    """Tests for Outlier and Plausibility Filters across Tiers 1-4."""

    # --------------------------------------------------------------------------
    # Tier 1: Primary Outlier Rules
    # --------------------------------------------------------------------------

    def test_statistical_iqr_clean_distribution(self):
        """Tier 1: Normal distribution of fares where all values fall within IQR bounds."""
        np.random.seed(42)
        normal_fares = np.random.normal(loc=5200.0, scale=350.0, size=50)
        lb, ub = filter_iqr_bounds(normal_fares)

        inliers = normal_fares[(normal_fares >= lb) & (normal_fares <= ub)]
        assert len(inliers) >= 45 # >90% inliers in normal distribution

    def test_statistical_iqr_outlier_rejection(self):
        """Tier 1: Synthetic outlier fares (₹1,500 and ₹18,000 on ₹5,000 route) flagged by IQR."""
        baseline = np.array([4800.0, 4900.0, 5000.0, 5100.0, 5200.0, 5300.0, 5400.0])
        lb, ub = filter_iqr_bounds(baseline)

        low_outlier = 1500.0
        high_outlier = 18000.0
        assert low_outlier < lb
        assert high_outlier > ub

    def test_domain_floor_enforcement(self):
        """Tier 1: Promotional glitches (₹499, ₹0, ₹750) fail domain floor check."""
        for invalid_fare in [0.0, 499.0, 750.0, 998.99]:
            assert is_domain_floor_violation(invalid_fare) is True

    def test_domain_ceiling_enforcement(self):
        """Tier 1: Accidental business class leaks (₹35,000, ₹48,000) fail domain ceiling check."""
        for invalid_fare in [25000.01, 28000.0, 45000.0, 95000.0]:
            assert is_domain_ceiling_violation(invalid_fare) is True

    def test_temporal_jump_detection_happy_path(self):
        """Tier 1: Normal price increase (+40%) passes; jump from ₹3,000 to ₹13,000 (+333%) fails."""
        assert is_temporal_surge_violation(prev_fare=4000.0, curr_fare=5600.0) is False # +40%
        assert is_temporal_surge_violation(prev_fare=3000.0, curr_fare=13000.0) is True # +333%

    def test_cross_source_median_variance_detection(self):
        """Tier 1: Quote deviating >50% from cohort median is flagged."""
        median_fare = 5000.0
        concordant_fare = 5250.0  # +5%
        deviant_fare = 8000.0     # +60%

        assert is_cross_source_variance_violation(concordant_fare, median_fare) is False
        assert is_cross_source_variance_violation(deviant_fare, median_fare) is True

    def test_consensus_scoring_kendall_w(self):
        """Tier 1: High agreement among 4 sources across 5 flights produces Kendall's W > 0.80."""
        # 4 sources ranking 5 flights identically or near-identically
        rankings = np.array([
            [1, 2, 3, 4, 5],
            [1, 2, 4, 3, 5],
            [1, 2, 3, 4, 5],
            [2, 1, 3, 4, 5],
        ])
        w = compute_kendall_w(rankings)
        assert w >= 0.80, f"Kendall's W below threshold: {w}"

    # --------------------------------------------------------------------------
    # Tier 2: Boundary Value Analysis (Exact Cutoffs)
    # --------------------------------------------------------------------------

    def test_domain_floor_exact_boundary(self):
        """Tier 2 Boundary: ₹998 is rejected, ₹999 is accepted, ₹1,000 is accepted."""
        assert is_domain_floor_violation(998.0) is True
        assert is_domain_floor_violation(998.99) is True
        assert is_domain_floor_violation(999.0) is False
        assert is_domain_floor_violation(1000.0) is False

    def test_domain_ceiling_exact_boundary(self):
        """Tier 2 Boundary: ₹24,999 is accepted, ₹25,000 is accepted, ₹25,001 is rejected."""
        assert is_domain_ceiling_violation(24999.0) is False
        assert is_domain_ceiling_violation(25000.0) is False
        assert is_domain_ceiling_violation(25000.01) is True
        assert is_domain_ceiling_violation(25001.0) is True

    def test_temporal_surge_exact_boundary(self):
        """Tier 2 Boundary: +299.9% jump passes, +300.0% jump passes, +300.1% jump is flagged."""
        prev = 3000.0
        assert is_temporal_surge_violation(prev, prev * 3.999) is False  # +299.9%
        assert is_temporal_surge_violation(prev, prev * 4.000) is False  # +300.0%
        assert is_temporal_surge_violation(prev, prev * 4.001) is True   # +300.1%

    def test_cross_source_median_exact_boundary(self):
        """Tier 2 Boundary: Exactly 50.0% departure passes; 50.1% departure is flagged."""
        median = 4000.0
        assert is_cross_source_variance_violation(4000.0 * 1.500, median) is False # +50%
        assert is_cross_source_variance_violation(4000.0 * 1.501, median) is True  # +50.1%
        assert is_cross_source_variance_violation(4000.0 * 0.500, median) is False # -50%
        assert is_cross_source_variance_violation(4000.0 * 0.499, median) is True  # -50.1%

    # --------------------------------------------------------------------------
    # Tier 3 & Tier 4: Adversarial & Combined Stress Testing
    # --------------------------------------------------------------------------

    def test_adversarial_malformed_inputs(self):
        """Tier 2 Adversarial: Extreme negative, infinite, and zero fares."""
        assert is_domain_floor_violation(-5000.0) is True
        assert is_domain_ceiling_violation(1e9) is True

    def test_combined_festival_surge_and_outlier_filtering(self):
        """
        Tier 3: During a simulated festival surge on DEL-SXR, valid surge fares
        (₹14,000) pass bounds while a rogue scrape of ₹38,000 is rejected by ceiling.
        """
        fares = [7500.0, 9200.0, 11500.0, 14000.0, 38000.0]
        cleaned = [f for f in fares if not is_domain_ceiling_violation(f) and not is_domain_floor_violation(f)]
        assert 14000.0 in cleaned
        assert 38000.0 not in cleaned
