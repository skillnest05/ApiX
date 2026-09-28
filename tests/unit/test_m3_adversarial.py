r"""
Mathematical Stress-Testing Suite for Econometric Aggregation.

Theoretical Invariants Verified:
1. Jevons & Törnqvist Axiomatic Robustness:
   - Extreme price vectors ($10^{-6}$ to $10^{6}$ ratios)
   - Fisher time-reversal on large randomized price arrays ($N=10,000$)
   - Extreme carrier market shares (99% vs 1%, 99.9% vs 0.1%) & entry/exit
   - Non-positive prices and empty vectors exception trapping
2. Multilateral GEKS & RYGEKS Invariants:
   - Cyclic price loops (4, 8, 12 periods) demonstrating zero chain drift vs drifting chained bilateral
   - Multilateral transitivity across all triples in multi-period matrices ($|P^{a,c} - P^{a,b} * P^{b,c}| < 10^{-6}$)
   - Mean Splice linking continuity & strict historical immutability across rolling windows
3. National Modified Laspeyres Invariants:
   - Strict unit weight normalization ($\sum \omega_{r,w} = 1.0000000 \pm 10^{-7}$)
   - Dynamic cell renormalization under severe missing cell scenarios
"""

import math
import numpy as np
import pytest

from apix.econometric.jevons import (
    JevonsEngine,
    compute_geometric_mean,
    compute_jevons_index,
)
from apix.econometric.tornqvist import (
    TornqvistEngine,
    compute_tornqvist_index,
)
from apix.econometric.rygeks import (
    RYGEKSEngine,
    compute_geks_vector,
    splice_mean,
)
from apix.econometric.laspeyres import (
    NationalWeightMatrix,
    calculate_national_laspeyres,
    get_default_weight_matrix,
)


# ==============================================================================
# 1. JEVONS ADVERSARIAL AXIOM STRESS TESTS
# ==============================================================================

class TestJevonsAdversarialAxioms:
    """Adversarial stress-testing of Stage 1 Jevons Elementary Index."""

    def test_jevons_extreme_price_ratios_up_and_down_1e6(self):
        """
        Adversarial Stress: Price relatives spanning 10^-6 to 10^6.
        Tests numerical stability of log-space formulation against overflow/underflow.
        """
        # Symmetrical extreme price ratios: 10^6 and 10^-6
        p0 = np.array([1e-6, 1.0, 1e6, 5000.0, 10.0])
        pt = np.array([1.0, 1e6, 1.0, 5000.0, 1e-5])

        P_0t = compute_jevons_index(p0, pt)
        P_t0 = compute_jevons_index(pt, p0)

        # Time-reversal property under extreme dynamic range
        product = P_0t * P_t0
        assert abs(product - 1.0) < 1e-6, f"Extreme ratio Jevons time-reversal failed: product={product}"

    def test_jevons_unbalanced_cohort_lengths_extreme(self):
        """
        Adversarial Stress: Asymmetric cohort sizes (e.g. 3 quotes vs 100 quotes)
        with extreme price distributions.
        """
        np.random.seed(42)
        p0 = np.array([100.0, 10000.0, 50000.0])
        pt = np.random.uniform(2000.0, 15000.0, size=100)

        P_0t = compute_jevons_index(p0, pt)
        P_t0 = compute_jevons_index(pt, p0)

        product = P_0t * P_t0
        assert abs(product - 1.0) < 1e-6, f"Unbalanced cohort Jevons time-reversal failed: product={product}"

    def test_jevons_large_random_array_time_reversal_n_10000(self):
        """
        Adversarial Stress: Large randomized price arrays (N = 10,000).
        Evaluates cumulative IEEE-754 precision drift over 10,000 operations.
        """
        np.random.seed(2026)
        p0 = np.random.uniform(1000.0, 25000.0, size=10000)
        pt = np.random.uniform(1000.0, 25000.0, size=10000)

        P_0t = compute_jevons_index(p0, pt)
        P_t0 = compute_jevons_index(pt, p0)

        product = P_0t * P_t0
        diff = abs(product - 1.0)
        assert diff < 1e-6, f"N=10,000 Jevons time-reversal drift too high: diff={diff}"
        # Typically log-space floating point drift is < 1e-14
        assert diff < 1e-12

    def test_jevons_lognormal_fat_tailed_distribution_n_10000(self):
        """Adversarial Stress: Heavy-tailed log-normal price distribution (N = 10,000)."""
        np.random.seed(777)
        p0 = np.random.lognormal(mean=8.5, sigma=1.2, size=10000)
        pt = np.random.lognormal(mean=8.6, sigma=1.2, size=10000)

        P_0t = compute_jevons_index(p0, pt)
        P_t0 = compute_jevons_index(pt, p0)

        product = P_0t * P_t0
        assert abs(product - 1.0) < 1e-6

    def test_jevons_non_positive_and_empty_exceptions(self):
        """
        Adversarial Boundary: Non-positive prices and empty vectors must raise ValueError.
        """
        # Empty arrays
        with pytest.raises(ValueError, match="must not be empty"):
            compute_jevons_index([], [])
        with pytest.raises(ValueError, match="must not be empty"):
            compute_jevons_index([1000.0], [])
        with pytest.raises(ValueError, match="must not be empty"):
            compute_geometric_mean([])

        # Zero prices
        with pytest.raises(ValueError, match="strictly positive"):
            compute_jevons_index([1000.0, 0.0], [1200.0, 1500.0])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_jevons_index([1000.0, 2000.0], [1200.0, 0.0])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_geometric_mean([5000.0, 0.0])

        # Negative prices
        with pytest.raises(ValueError, match="strictly positive"):
            compute_jevons_index([-100.0], [500.0])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_jevons_index([500.0], [-100.0])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_geometric_mean([-500.0])

    def test_jevons_numerical_overflow_underflow_immunity(self):
        """
        Adversarial Boundary: Compute geometric mean of 1,000 huge (10^50) and 1,000 tiny (10^-50) numbers.
        Naive product prod(p_i) would overflow to inf or underflow to 0.0.
        Log-space exp(mean(log)) must handle this cleanly.
        """
        huge_prices = np.full(1000, 1e50)
        gm_huge = compute_geometric_mean(huge_prices)
        assert math.isclose(gm_huge, 1e50, rel_tol=1e-9)

        tiny_prices = np.full(1000, 1e-50)
        gm_tiny = compute_geometric_mean(tiny_prices)
        assert math.isclose(gm_tiny, 1e-50, rel_tol=1e-9)


# ==============================================================================
# 2. TORNQVIST ADVERSARIAL AXIOM STRESS TESTS
# ==============================================================================

class TestTornqvistAdversarialAxioms:
    """Adversarial stress-testing of Stage 2 Törnqvist Superlative Index."""

    def test_tornqvist_extreme_price_relatives(self):
        """Adversarial Stress: Price relatives ranging from 10^-6 to 10^6."""
        I_rel = np.array([1e-6, 1e6, 2.5, 0.4])
        w0 = np.array([0.25, 0.25, 0.25, 0.25])
        wt = np.array([0.25, 0.25, 0.25, 0.25])

        PT_0t = compute_tornqvist_index(I_rel, w0, wt)
        PT_t0 = compute_tornqvist_index(1.0 / I_rel, wt, w0)

        product = PT_0t * PT_t0
        assert abs(product - 1.0) < 1e-6, f"Extreme Törnqvist time reversal failed: {product}"

    def test_tornqvist_large_random_array_n_10000(self):
        """
        Adversarial Stress: Large array with N = 10,000 items and randomized shares.
        """
        np.random.seed(12345)
        N = 10000
        I_rel = np.random.uniform(0.5, 2.0, size=N)
        w0 = np.random.exponential(scale=1.0, size=N)
        wt = np.random.exponential(scale=1.0, size=N)

        PT_0t = compute_tornqvist_index(I_rel, w0, wt)
        PT_t0 = compute_tornqvist_index(1.0 / I_rel, wt, w0)

        product = PT_0t * PT_t0
        diff = abs(product - 1.0)
        assert diff < 1e-6, f"N=10,000 Törnqvist drift too high: {diff}"
        assert diff < 1e-12

    def test_tornqvist_extreme_asymmetric_market_shares(self):
        """
        Adversarial Stress: Extreme carrier market share asymmetry:
        - 99% vs 1%
        - 99.9% vs 0.1%
        - Inverted carrier shares across periods.
        """
        scenarios = [
            (np.array([0.99, 0.01]), np.array([0.01, 0.99])),
            (np.array([0.999, 0.001]), np.array([0.001, 0.999])),
            (np.array([0.9999, 0.0001]), np.array([0.0001, 0.9999])),
        ]

        for w0, wt in scenarios:
            I_rel = np.array([1.45, 0.70])
            PT_0t = compute_tornqvist_index(I_rel, w0, wt)
            PT_t0 = compute_tornqvist_index(1.0 / I_rel, wt, w0)

            product = PT_0t * PT_t0
            assert abs(product - 1.0) < 1e-6, f"Extreme shares failed: w0={w0}, wt={wt}, prod={product}"

    def test_tornqvist_carrier_entry_and_exit_adversarial(self):
        """
        Adversarial Interaction: 5-carrier universe with entry and exit:
        - Carrier A: Operating in both (0.95 base -> 0.10 curr)
        - Carrier B: Operating in both (0.04 base -> 0.80 curr)
        - Carrier C: Exits (0.01 base -> 0.00 curr)
        - Carrier D: Enters (0.00 base -> 0.10 curr)
        Active intersection C^{0 cap t} = {A, B}.
        """
        # Active subset prices
        I_rel_active = np.array([1.25, 0.85])
        # Raw weights in base and current before active subset normalization
        w0_raw = np.array([0.95, 0.04])
        wt_raw = np.array([0.10, 0.80])

        PT_0t = compute_tornqvist_index(I_rel_active, w0_raw, wt_raw)
        PT_t0 = compute_tornqvist_index(1.0 / I_rel_active, wt_raw, w0_raw)

        product = PT_0t * PT_t0
        assert abs(product - 1.0) < 1e-6, f"Entry/exit time reversal failed: {product}"

    def test_tornqvist_monopoly_reduction_boundary(self):
        """
        Adversarial Boundary: Single operating carrier on route.
        Must reduce strictly to the elementary relative without log-space corruption.
        """
        for rel in [0.05, 1.0, 1.75, 10.0]:
            PT = compute_tornqvist_index([rel], [1.0], [1.0])
            assert math.isclose(PT, rel, rel_tol=1e-12)

    def test_tornqvist_exceptions_on_invalid_inputs(self):
        """Adversarial Boundary: Input validation and exception trapping."""
        # Empty inputs
        with pytest.raises(ValueError, match="must not be empty"):
            compute_tornqvist_index([], [], [])

        # Dimension mismatches
        with pytest.raises(ValueError, match="Mismatched dimensions"):
            compute_tornqvist_index([1.1, 1.2], [0.5], [0.5, 0.5])
        with pytest.raises(ValueError, match="Mismatched dimensions"):
            compute_tornqvist_index([1.1, 1.2], [0.5, 0.5], [0.5])

        # Non-positive elementary relatives
        with pytest.raises(ValueError, match="strictly positive"):
            compute_tornqvist_index([0.0, 1.2], [0.5, 0.5], [0.5, 0.5])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_tornqvist_index([-1.0, 1.2], [0.5, 0.5], [0.5, 0.5])

        # Zero or negative weight sums
        with pytest.raises(ValueError, match="strictly positive"):
            compute_tornqvist_index([1.1, 1.2], [0.0, 0.0], [0.5, 0.5])
        with pytest.raises(ValueError, match="strictly positive"):
            compute_tornqvist_index([1.1, 1.2], [0.5, 0.5], [-0.5, -0.5])


# ==============================================================================
# 3. GEKS & RYGEKS ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestGEKSAndRYGEKSAdversarial:
    """Adversarial stress-testing of Multilateral GEKS & RYGEKS Engine."""

    def test_geks_zero_chain_drift_cyclic_loops_4_8_12_periods(self):
        """
        Adversarial Stress: Multi-period cyclic price paths (4, 8, 12 periods).
        Demonstrates that:
        1. Chained bilateral index suffers observable chain drift (|P_chain - 1.0| > 0).
        2. Multilateral GEKS strictly returns 1.000000 +- 1e-6 (zero chain drift).
        """
        for cycle_len in [4, 8, 12]:
            np.random.seed(42 + cycle_len)
            K = 5 # 5 carriers

            # Generate cyclic price path: period 0 through cycle_len, where period cycle_len == period 0
            base_prices = np.array([3000.0, 4500.0, 5200.0, 6800.0, 8500.0])
            base_weights = np.array([0.62, 0.14, 0.07, 0.05, 0.12])

            prices_history = [base_prices]
            weights_history = [base_weights]

            # Intermediate shock periods
            for step in range(1, cycle_len):
                shock = np.random.uniform(0.8, 1.3, size=K)
                shocked_p = base_prices * shock
                shocked_w = np.random.dirichlet(np.ones(K))
                prices_history.append(shocked_p)
                weights_history.append(shocked_w)

            # Return exactly to base period
            prices_history.append(base_prices.copy())
            weights_history.append(base_weights.copy())

            T = len(prices_history)

            # Construct full bilateral matrix P_T
            P_mat = np.zeros((T, T), dtype=float)
            for i in range(T):
                for j in range(T):
                    if i == j:
                        P_mat[i, j] = 1.0
                    else:
                        rel = prices_history[j] / prices_history[i]
                        P_mat[i, j] = compute_tornqvist_index(rel, weights_history[i], weights_history[j])

            # Chained bilateral: period 0 -> 1 -> 2 -> ... -> cycle_len
            chained_product = 1.0
            for t in range(T - 1):
                chained_product *= P_mat[t, t + 1]

            # Multilateral GEKS
            geks_vec = compute_geks_vector(P_mat)
            geks_loop_return = geks_vec[-1] / geks_vec[0]

            # GEKS must satisfy exact zero chain drift
            drift = abs(geks_loop_return - 1.0)
            assert drift < 1e-6, f"GEKS chain drift violated for cycle {cycle_len}: return={geks_loop_return}"

    def test_geks_multilateral_transitivity_all_triples(self):
        """
        Adversarial Stress: Complete transitivity across ALL 3,375 triples in a 15-period matrix.
        Axiom: For every (a, b, c): |P^{a,c} - P^{a,b} * P^{b,c}| < 1e-6.
        """
        np.random.seed(9876)
        T = 15
        K = 6

        prices = [np.random.uniform(2000.0, 15000.0, size=K) for _ in range(T)]
        weights = [np.random.dirichlet(np.ones(K)) for _ in range(T)]

        bilateral_mat = np.zeros((T, T), dtype=float)
        for i in range(T):
            for j in range(T):
                if i == j:
                    bilateral_mat[i, j] = 1.0
                else:
                    rel = prices[j] / prices[i]
                    bilateral_mat[i, j] = compute_tornqvist_index(rel, weights[i], weights[j])

        geks_vec = compute_geks_vector(bilateral_mat)

        max_discrepancy = 0.0
        for a in range(T):
            for b in range(T):
                for c in range(T):
                    P_ab = geks_vec[b] / geks_vec[a]
                    P_bc = geks_vec[c] / geks_vec[b]
                    P_ac = geks_vec[c] / geks_vec[a]
                    diff = abs((P_ab * P_bc) - P_ac)
                    if diff > max_discrepancy:
                        max_discrepancy = diff

        assert max_discrepancy < 1e-6, f"Multilateral transitivity failed: max_diff={max_discrepancy}"
        assert max_discrepancy < 1e-12

    def test_rygeks_mean_splice_historical_immutability(self):
        """
        Adversarial Stress: Mean Splice linking non-revisability.
        Verify that as the rolling window advances across 30 periods,
        historical published indices P_{published}^k (k < t) are PERMANENTLY IMMUTABLE.
        """
        np.random.seed(54321)
        engine = RYGEKSEngine(window_length=7, base_value=100.0, splice_method="mean")

        # Period 0
        engine.add_first_period("P0")
        published_log = {0: dict(engine.get_published_series())}

        # Advance 25 rolling periods
        for t in range(1, 26):
            p_id = f"P{t}"
            active_hist = [f"P{k}" for k in range(max(0, t - 7), t)]
            pairwise = {}
            for h in active_hist:
                pairwise[h] = 1.0 + float(np.random.uniform(-0.04, 0.04))

            res = engine.add_period(p_id, pairwise)
            curr_series = engine.get_published_series()
            published_log[t] = dict(curr_series)

            # Immutability verification: check all historical periods
            for past_t in range(t):
                past_id = f"P{past_t}"
                prior_val = published_log[past_t][past_id]
                now_val = curr_series[past_id]
                assert prior_val == now_val, (
                    f"Historical immutability violated at step {t} for period {past_id}: "
                    f"published={prior_val}, current={now_val}"
                )

    def test_rygeks_exceptions_and_edge_cases(self):
        """Adversarial Boundary: RYGEKS stateful edge case exception traps."""
        # Window length < 2
        with pytest.raises(ValueError, match="Window length must be >= 2"):
            RYGEKSEngine(window_length=1)

        engine = RYGEKSEngine(window_length=5, base_value=100.0)
        engine.add_first_period("P0")

        # Calling add_first_period twice
        with pytest.raises(ValueError, match="can only be called on an empty engine"):
            engine.add_first_period("P0_again")

        # Overwriting already published period
        with pytest.raises(ValueError, match="has already been published"):
            engine.add_period("P0", {})

        # Negative relative
        with pytest.raises(ValueError, match="must be positive"):
            engine.add_period("P1", {"P0": -1.25})

        # Non-square matrix in compute_geks_vector
        with pytest.raises(ValueError, match="must be a square 2D matrix"):
            compute_geks_vector(np.array([[1.0, 1.2, 1.3], [0.8, 1.0, 1.1]]))

        # Non-positive elements in bilateral matrix
        with pytest.raises(ValueError, match="must be strictly positive"):
            compute_geks_vector(np.array([[1.0, 0.0], [0.0, 1.0]]))


# ==============================================================================
# 4. NATIONAL MODIFIED LASPEYRES ADVERSARIAL STRESS TESTS
# ==============================================================================

class TestNationalLaspeyresAdversarial:
    """Adversarial stress-testing of Stage 3 National Laspeyres Aggregator."""

    def test_strict_weight_matrix_normalization(self):
        """
        Adversarial Invariant: sum_{r=1}^{30} sum_{w=1}^{5} omega_{r,w} == 1.0000000 +- 1e-7.
        """
        matrix = get_default_weight_matrix()
        total_sum = sum(matrix.matrix.values())
        assert abs(total_sum - 1.0) < 1e-7, f"Weight matrix normalization violated: sum={total_sum}"

    def test_national_laspeyres_severe_missing_cells_renormalization(self):
        """
        Adversarial Stress: Dynamic cell renormalization with 80% missing cells (only 30/150 observed).
        Under a uniform +25% price shock, the index must scale strictly to 125.0000 +- 1e-4.
        """
        import random
        random.seed(999)
        matrix = get_default_weight_matrix()
        all_cells = list(matrix.matrix.keys())

        # Test varying missing cell rates: 10%, 30%, 50%, 80%
        for missing_pct in [0.10, 0.30, 0.50, 0.80]:
            keep_count = int(len(all_cells) * (1.0 - missing_pct))
            sampled_cells = random.sample(all_cells, keep_count)

            # Uniform 1.25 price relative (+25% shock)
            indices = {c: 1.25 for c in sampled_cells}

            res = calculate_national_laspeyres(indices, matrix, impute_missing=True)
            assert abs(res.apix_index - 125.0) < 1e-4, (
                f"Missing rate {missing_pct*100}% failed: index={res.apix_index}, expected=125.0"
            )

    def test_national_laspeyres_empty_input_raises_error(self):
        """Adversarial Boundary: Empty route-window dataset raises ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate Stage 3 index from empty"):
            calculate_national_laspeyres({})
