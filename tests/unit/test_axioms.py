"""
Unit Test Suite: Mathematical Axioms & Econometric Invariants.
Enforces zero-tolerance precision criteria on:
1. Fisher Time-Reversal Property of Jevons (P_J^{0,t} * P_J^{t,0} = 1.0 +- 1e-6)
2. Fisher Time-Reversal Property of Törnqvist (P_T^{0,t} * P_T^{t,0} = 1.0 +- 1e-6)
3. GEKS Multilateral Transitivity (P^{a,b} * P^{b,c} = P^{a,c} +- 1e-6)
4. GEKS Zero Chain-Drift Resistance on Cyclic Price Loops
5. Strict Weight Normalization Sum (sum(omega_{r,w}) = 1.0 +- 1e-7)
6. Monotonicity & Commensurability Axioms
"""

import math
import numpy as np
import pytest

from tests.conftest import (
    oracle_jevons,
    oracle_tornqvist,
    oracle_laspeyres,
    oracle_geks,
    DGCA_CITY_PAIRS,
    ADVANCE_WINDOWS,
    CARRIER_MARKET_SHARES,
)


class TestJevonsAxioms:
    """Mathematical axiom verification for Stage 1 Jevons Elementary Index."""

    def test_jevons_time_reversal_identity(self):
        """
        Axiom 1: Fisher Time-Reversal Test:
        P_J^{0,t} * P_J^{t,0} == 1.0 within 1e-6 tolerance.
        """
        p0 = np.array([2500.0, 3400.0, 4200.0, 5100.0, 6800.0])
        pt = np.array([2800.0, 3200.0, 4900.0, 5300.0, 7100.0])

        P_0t = oracle_jevons(p0, pt)
        P_t0 = oracle_jevons(pt, p0)

        product = P_0t * P_t0
        assert abs(product - 1.0) < 1e-6, f"Time reversal violated: {product} (diff: {abs(product - 1.0)})"

    def test_jevons_time_reversal_randomized_vectors(self):
        """Tier 2 Boundary: Time reversal holds across 10 distinct random price distributions."""
        np.random.seed(101)
        for i in range(10):
            n = np.random.randint(5, 50)
            p0 = np.random.uniform(1500.0, 18000.0, size=n)
            pt = np.random.uniform(1500.0, 18000.0, size=n)

            P_0t = oracle_jevons(p0, pt)
            P_t0 = oracle_jevons(pt, p0)
            product = P_0t * P_t0
            assert abs(product - 1.0) < 1e-6, f"Random trial {i} failed: {product}"

    def test_jevons_single_quote_cohort(self):
        """Tier 2 Corner Case: Single quote (n=1) simplifies to price ratio."""
        p0 = np.array([4500.0])
        pt = np.array([5400.0])
        P_0t = oracle_jevons(p0, pt)
        assert abs(P_0t - 1.20) < 1e-6

        # Time reversal holds for n=1
        P_t0 = oracle_jevons(pt, p0)
        assert abs(P_0t * P_t0 - 1.0) < 1e-6

    def test_jevons_commensurability_scale_invariance(self):
        """
        Axiom: Commensurability (Scale Invariance):
        Scaling quotes by factor lambda leaves Jevons invariant.
        """
        p0 = np.array([3000.0, 4500.0, 6000.0])
        pt = np.array([3300.0, 4200.0, 6600.0])
        P_base = oracle_jevons(p0, pt)

        for scale_factor in [0.5, 2.0, 10.0, 100.0]:
            P_scaled = oracle_jevons(scale_factor * p0, scale_factor * pt)
            assert abs(P_base - P_scaled) < 1e-6

    def test_jevons_proportionality_in_current_prices(self):
        """Axiom: Proportionality in current prices: P(p0, lambda * pt) == lambda * P(p0, pt)."""
        p0 = np.array([3000.0, 4000.0, 5000.0])
        pt = np.array([3300.0, 4400.0, 5500.0])
        P_base = oracle_jevons(p0, pt)

        lambda_mult = 1.25
        P_mult = oracle_jevons(p0, lambda_mult * pt)
        assert abs(P_mult - (lambda_mult * P_base)) < 1e-6

    def test_jevons_identity_axiom(self):
        """Axiom: Identity axiom: P(p, p) == 1.000000."""
        p = np.array([2100.0, 3900.0, 7800.0, 12000.0])
        assert abs(oracle_jevons(p, p) - 1.0) < 1e-6

    def test_jevons_monotonicity(self):
        """
        Axiom: Monotonicity:
        If any price in comparison period increases without any decrease, index must increase.
        """
        p0 = np.array([3000.0, 4000.0, 5000.0])
        pt1 = np.array([3000.0, 4000.0, 5000.0]) # equal
        pt2 = np.array([3000.0, 4200.0, 5000.0]) # one price increased

        P1 = oracle_jevons(p0, pt1)
        P2 = oracle_jevons(p0, pt2)
        assert P2 > P1
        assert abs(P1 - 1.0) < 1e-6
        assert P2 > 1.0

    def test_jevons_empty_cohort_raises_error(self):
        """Tier 2 Corner Case: Zero quotes raises ValueError."""
        with pytest.raises(ValueError):
            oracle_jevons(np.array([]), np.array([]))

    def test_jevons_non_positive_price_raises_error(self):
        """Tier 2 Boundary: Non-positive price trapped."""
        with pytest.raises(ValueError):
            oracle_jevons(np.array([4000.0, 0.0]), np.array([4200.0, 4500.0]))
        with pytest.raises(ValueError):
            oracle_jevons(np.array([4000.0, 5000.0]), np.array([4200.0, -100.0]))


class TestTornqvistAxioms:
    """Mathematical axiom verification for Stage 2 Törnqvist Superlative Index."""

    def test_tornqvist_time_reversal_identity(self):
        """
        Axiom 1: Fisher Time-Reversal for Törnqvist:
        P_T^{0,t} * P_T^{t,0} == 1.0 within 1e-6 tolerance.
        """
        I_rel = np.array([1.08, 0.95, 1.12, 1.03, 0.98])
        w0 = np.array([0.62, 0.14, 0.07, 0.05, 0.12])
        wt = np.array([0.60, 0.15, 0.08, 0.06, 0.11])

        # Forward Törnqvist
        PT_0t = oracle_tornqvist(I_rel, w0, wt)
        # Backward Törnqvist (relative is 1 / I_rel, period weights reversed)
        PT_t0 = oracle_tornqvist(1.0 / I_rel, wt, w0)

        product = PT_0t * PT_t0
        assert abs(product - 1.0) < 1e-6, f"Törnqvist time reversal failed: {product}"

    def test_tornqvist_monopoly_route_reduction(self):
        """
        Tier 2 Corner Case: Single carrier on route (market share = 1.0).
        Törnqvist index simplifies exactly to the carrier's elementary index.
        """
        I_rel = np.array([1.185])
        w0 = np.array([1.0])
        wt = np.array([1.0])

        PT = oracle_tornqvist(I_rel, w0, wt)
        assert abs(PT - 1.185) < 1e-6

    def test_tornqvist_symmetric_price_movement(self):
        """Tier 2 Corner Case: If all carriers change price by exactly factor k, index is k."""
        k = 1.15
        I_rel = np.array([k, k, k, k, k])
        w0 = np.array([0.65, 0.15, 0.10, 0.05, 0.05])
        wt = np.array([0.60, 0.20, 0.08, 0.06, 0.06])

        PT = oracle_tornqvist(I_rel, w0, wt)
        assert abs(PT - k) < 1e-6

    def test_carrier_entry_exit_normalization(self):
        r"""
        Tier 3 Interaction: Carrier exits in period t (share drops to 0).
        Normalized over operating carrier subset C^{0 \cap t}, weights sum to 1.0.
        """
        # 3 carriers operating in base, 1 exits in comparison
        I_rel = np.array([1.05, 1.10])
        w0_raw = np.array([0.60, 0.20]) # Carrier 3 had 0.20 in base, omitted here
        wt_raw = np.array([0.75, 0.25])

        PT = oracle_tornqvist(I_rel, w0_raw, wt_raw)
        assert PT > 1.0
        # Time reversal holds over active subset
        PT_rev = oracle_tornqvist(1.0 / I_rel, wt_raw, w0_raw)
        assert abs(PT * PT_rev - 1.0) < 1e-6


class TestGEKSAxioms:
    """Multilateral GEKS (CCDI) Transitivity & Chain-Drift Resistance."""

    def test_geks_multilateral_transitivity(self):
        """
        Axiom 2: Multilateral Transitivity (Circularity):
        For any 3 periods a, b, c: P^{a,b} * P^{b,c} == P^{a,c} +- 1e-6.
        """
        # 3-period bilateral matrix
        # P_T[j, k] represents price movement from period j to period k
        P_T = np.array([
            [1.000, 1.060, 1.120],
            [1.0 / 1.060, 1.000, 1.0566],
            [1.0 / 1.120, 1.0 / 1.0566, 1.000]
        ])

        geks_series = oracle_geks(P_T)
        P_01 = geks_series[1] / geks_series[0]
        P_12 = geks_series[2] / geks_series[1]
        P_02 = geks_series[2] / geks_series[0]

        assert abs((P_01 * P_12) - P_02) < 1e-6, f"GEKS transitivity failed: P_01*P_12={P_01*P_12}, P_02={P_02}"

    def test_geks_zero_chain_drift_on_cyclic_price_loop(self):
        """
        Axiom 3: Chain-Drift Invariance:
        When prices bounce away and return to exact initial state (P0 -> P1 -> P0),
        GEKS strictly returns 1.000000 (standard chained bilateral drift eliminated).
        """
        # Period 0: Baseline (1.0)
        # Period 1: Surge +25%
        # Period 2: Return to Baseline
        P_T = np.array([
            [1.00, 1.25, 1.00],
            [1.0 / 1.25, 1.00, 1.0 / 1.25],
            [1.00, 1.25, 1.00]
        ])

        geks_series = oracle_geks(P_T)
        P_02 = geks_series[2] / geks_series[0]
        assert abs(P_02 - 1.0) < 1e-6, f"Chain drift observed in GEKS: {P_02}"

    def test_geks_4_period_multi_season_matrix(self):
        """Tier 2 Stress: 4-period seasonal cycle transitivity across all pairs."""
        P_T = np.array([
            [1.00, 1.04, 1.08, 1.02],
            [1.0 / 1.04, 1.00, 1.038, 0.980],
            [1.0 / 1.08, 1.0 / 1.038, 1.00, 0.944],
            [1.0 / 1.02, 1.0 / 0.980, 1.0 / 0.944, 1.00]
        ])

        geks = oracle_geks(P_T)
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    P_ij = geks[j] / geks[i]
                    P_jk = geks[k] / geks[j]
                    P_ik = geks[k] / geks[i]
                    assert abs((P_ij * P_jk) - P_ik) < 1e-6


class TestWeightNormalization:
    """Stage 3 National Laspeyres Weight Normalization & Invariance Proofs."""

    def test_strict_weight_sum_identity(self, national_weights_matrix):
        """
        Axiom 4: Strict Unit Normalization:
        sum(omega_{r,w}) == 1.0000000 across 30 directional sectors and 5 windows
        within numerical tolerance 1e-7.
        """
        total_weight = float(np.sum(national_weights_matrix))
        assert abs(total_weight - 1.0) < 1e-7, f"Weight sum invalid: {total_weight}"

    def test_30_sectors_coverage(self, dgca_sectors):
        """Tier 1 Feature: Exact 30 directional sectors (15 bidirectional pairs)."""
        assert len(dgca_sectors) == 30
        assert len(set(dgca_sectors)) == 30
        # Check bidirectional symmetry
        for s in dgca_sectors:
            o, d = s.split("-")
            rev = f"{d}-{o}"
            assert rev in dgca_sectors

    def test_5_advance_windows(self):
        """Tier 1 Feature: 5 advance windows summing to 1.000000."""
        windows = list(ADVANCE_WINDOWS.keys())
        assert windows == [1, 7, 15, 30, 45]
        total_alpha = sum(ADVANCE_WINDOWS.values())
        assert abs(total_alpha - 1.0) < 1e-7

    def test_national_laspeyres_base_period_scaling(self, national_weights_matrix):
        """Stage 3 Base Normalization: In base period where P_T(r,w) = 1.0, APIx == 100.0."""
        base_indices = np.ones((30, 5), dtype=float)
        apix_base = oracle_laspeyres(base_indices, national_weights_matrix)
        assert abs(apix_base - 100.0) < 1e-6

    def test_national_laspeyres_homogeneous_scaling(self, national_weights_matrix):
        """Homogeneity of degree 1: A uniform +10% price rise yields APIx == 110.0."""
        shocked_indices = np.full((30, 5), 1.10, dtype=float)
        apix_shocked = oracle_laspeyres(shocked_indices, national_weights_matrix)
        assert abs(apix_shocked - 110.0) < 1e-6
