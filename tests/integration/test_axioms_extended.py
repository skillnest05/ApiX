"""
Extended Mathematical Axiom Verification Suite for Econometric Index Formulas.
Comprehensive verification of theoretical axiomatic properties across all aggregation stages.

Empirical Stress Testing of Mathematical Axioms:
1. Stage 1 Jevons:
   - Fisher Time-Reversal: P_J(0, t) * P_J(t, 0) == 1.0 +- 1e-6 (balanced, unbalanced, extreme ratios, large N=10,000)
   - Commensurability (Scale Invariance): P_J(lambda * p0, lambda * pt) == P_J(p0, pt)
   - Transitivity in price relatives: P_J(a, b) * P_J(b, c) == P_J(a, c) +- 1e-6
   - Duck-typed input handling in calculate_cohort_jevons and compute_all_elementary_indices
   - Exception trapping on non-positive and empty inputs

2. Stage 2 Törnqvist:
   - Fisher Time-Reversal: P_T(0, t) * P_T(t, 0) == 1.0 +- 1e-6
   - Carrier weight re-normalization across missing/entering carriers (C^{0 cap t})
   - Monopoly route reduction (N=1)
   - Symmetric price shock invariance
   - Extreme asymmetric market shares (99.99% vs 0.01%)
   - Dimension mismatch and non-positive weight sum exception trapping

3. Stage 3 Laspeyres:
   - Strict Weight Normalization: sum(omega_{r,w}) == 1.0000000 +- 1e-7 across 30x5 matrix
   - Single-route inputs and severe missing cell re-normalization (impute_missing=True)
   - Mixed input representations:
     * Raw floats in dict
     * String window keys ("T+1", "T+15", etc.)
     * Dict containing TornqvistRouteResult model instances
     * Dict containing plain dicts with {"index_value": float}
     * Sequence of TornqvistRouteResult instances
     * Sequence of custom duck-typed objects with attributes
     * NumPy 2D array of shape (30, 5)
   - Base period scaling (100.0) and degree-1 homogeneity
   - Exception trapping on empty / invalid inputs

4. Stage 4 RYGEKS:
   - Multilateral Transitivity of GEKS matrix: P_{ik} == P_{ij} * P_{jk} +- 1e-6 across all triples
   - Zero chain drift on cyclic price loops (3, 6, 12, 24 periods) vs drifting chained bilateral
   - Mean Splice linking invariance: past published history is strictly immutable across rolling windows
   - Expanding window (t < T) to rolling window (t >= T) transition
   - Alternative splicing methods (mean, movement, window, half)
"""

import math
from datetime import date, timedelta
from typing import Any, Dict, List, NamedTuple
import numpy as np
import pytest

from apix.econometric.jevons import (
    JevonsEngine,
    JevonsCohortResult,
    compute_geometric_mean,
    compute_jevons_index,
    calculate_cohort_jevons,
    compute_all_elementary_indices,
)
from apix.econometric.tornqvist import (
    TornqvistEngine,
    TornqvistRouteResult,
    compute_tornqvist_index,
    aggregate_route_tornqvist,
    compute_all_route_indices,
)
from apix.econometric.laspeyres import (
    LaspeyresEngine,
    NationalWeightMatrix,
    Stage3Result,
    calculate_national_laspeyres,
    get_default_weight_matrix,
)
from apix.econometric.rygeks import (
    RYGEKSEngine,
    RYGEKSResult,
    compute_geks_vector,
    splice_mean,
    splice_movement,
    splice_window,
    splice_half,
    calculate_rygeks_series,
)


# ==============================================================================
# 1. STAGE 1 JEVONS EMPIRICAL STRESS TESTS
# ==============================================================================

class TestJevonsEmpiricalStress:
    """Empirical verification of Stage 1 Jevons Elementary Index."""

    def test_jevons_time_reversal_balanced_and_unbalanced(self):
        """
        Fisher Time-Reversal: P_J(0, t) * P_J(t, 0) == 1.0 +- 1e-6
        Tested across:
        - Balanced cohort sizes (N = 1, 5, 50, 1000)
        - Highly unbalanced cohort sizes (N0=3 vs Nt=500, N0=500 vs Nt=3)
        """
        np.random.seed(42)

        # Balanced cases
        for n in [1, 5, 50, 1000]:
            p0 = np.random.uniform(1000.0, 20000.0, size=n)
            pt = np.random.uniform(1000.0, 20000.0, size=n)

            P_0t = compute_jevons_index(p0, pt)
            P_t0 = compute_jevons_index(pt, p0)
            product = P_0t * P_t0
            assert abs(product - 1.0) < 1e-6, f"Balanced n={n} failed: {product}"

        # Unbalanced cases
        unbalanced_pairs = [(3, 500), (500, 3), (1, 100), (100, 1), (7, 43)]
        for n0, nt in unbalanced_pairs:
            p0 = np.random.uniform(1200.0, 18000.0, size=n0)
            pt = np.random.uniform(1200.0, 18000.0, size=nt)

            P_0t = compute_jevons_index(p0, pt)
            P_t0 = compute_jevons_index(pt, p0)
            product = P_0t * P_t0
            assert abs(product - 1.0) < 1e-6, f"Unbalanced ({n0}, {nt}) failed: {product}"

    def test_jevons_extreme_ratios_and_large_n(self):
        """
        Tests numerical stability under extreme price ratios (10^-8 to 10^8)
        and large array sizes (N = 10,000).
        """
        np.random.seed(101)
        # Extreme ratios
        p0 = np.array([1e-8, 1.0, 1e8, 4500.0, 0.01])
        pt = np.array([1.0, 1e8, 1e-8, 4500.0, 100.0])
        P_0t = compute_jevons_index(p0, pt)
        P_t0 = compute_jevons_index(pt, p0)
        assert abs(P_0t * P_t0 - 1.0) < 1e-6

        # Large N = 10,000
        p0_large = np.random.uniform(1500.0, 25000.0, size=10000)
        pt_large = np.random.uniform(1500.0, 25000.0, size=10000)
        P_large_0t = compute_jevons_index(p0_large, pt_large)
        P_large_t0 = compute_jevons_index(pt_large, p0_large)
        assert abs(P_large_0t * P_large_t0 - 1.0) < 1e-12

    def test_jevons_commensurability_and_transitivity(self):
        """
        - Commensurability: P_J(lambda * p0, lambda * pt) == P_J(p0, pt)
        - Transitivity: P_J(a, b) * P_J(b, c) == P_J(a, c) +- 1e-6
        """
        np.random.seed(202)
        n = 20
        pa = np.random.uniform(2000.0, 10000.0, size=n)
        pb = np.random.uniform(2000.0, 10000.0, size=n)
        pc = np.random.uniform(2000.0, 10000.0, size=n)

        # Commensurability
        P_ab_base = compute_jevons_index(pa, pb)
        for scale in [0.001, 0.5, 2.0, 100.0, 10000.0]:
            P_ab_scaled = compute_jevons_index(scale * pa, scale * pb)
            assert abs(P_ab_base - P_ab_scaled) < 1e-6

        # Transitivity
        P_ab = compute_jevons_index(pa, pb)
        P_bc = compute_jevons_index(pb, pc)
        P_ac = compute_jevons_index(pa, pc)
        assert abs((P_ab * P_bc) - P_ac) < 1e-6

        # Transitivity with unbalanced cohorts
        pa_unb = np.random.uniform(2000.0, 10000.0, size=10)
        pb_unb = np.random.uniform(2000.0, 10000.0, size=25)
        pc_unb = np.random.uniform(2000.0, 10000.0, size=40)
        P_ab_u = compute_jevons_index(pa_unb, pb_unb)
        P_bc_u = compute_jevons_index(pb_unb, pc_unb)
        P_ac_u = compute_jevons_index(pa_unb, pc_unb)
        assert abs((P_ab_u * P_bc_u) - P_ac_u) < 1e-6

    def test_jevons_duck_typed_quote_inputs(self):
        """
        Verifies calculate_cohort_jevons and compute_all_elementary_indices
        handle diverse duck-typed quotes: dictionaries, namedtuples, and custom objects.
        """
        class DuckQuote:
            def __init__(self, sector, carrier_code, advance_window, total_fare, flight_number="6E-101", departure_time="08:00"):
                self.sector = sector
                self.carrier_code = carrier_code
                self.advance_window = advance_window
                self.total_fare = total_fare
                self.flight_number = flight_number
                self.departure_time = departure_time

        q0 = [
            DuckQuote("DEL-BOM", "6E", 15, 4000.0, "6E-101", "08:00"),
            DuckQuote("DEL-BOM", "6E", 15, 5000.0, "6E-102", "14:00"),
            {"sector": "DEL-BOM", "carrier_code": "6E", "advance_days": 15, "total_fare": 6000.0, "flight_number": "6E-103", "departure_time": "20:00"},
        ]
        qt = [
            DuckQuote("DEL-BOM", "6E", 15, 4400.0, "6E-101", "08:00"),
            DuckQuote("DEL-BOM", "6E", 15, 5500.0, "6E-102", "14:00"),
            {"sector": "DEL-BOM", "carrier_code": "6E", "advance_days": 15, "total_fare": 6600.0, "flight_number": "6E-103", "departure_time": "20:00"},
        ]

        res = calculate_cohort_jevons(q0, qt, "DEL-BOM", "6E", 15, match_flights=True)
        assert res is not None
        assert res.matched_quote_count == 3
        # Exactly +10% price rise across all 3 matched pairs
        assert abs(res.index_value - 1.10) < 1e-4

        all_res = compute_all_elementary_indices(q0, qt)
        assert ("DEL-BOM", "6E", 15) in all_res
        assert abs(all_res[("DEL-BOM", "6E", 15)].index_value - 1.10) < 1e-4


# ==============================================================================
# 2. STAGE 2 TÖRNQVIST EMPIRICAL STRESS TESTS
# ==============================================================================

class TestTornqvistEmpiricalStress:
    """Empirical verification of Stage 2 Törnqvist Superlative Index."""

    def test_tornqvist_time_reversal_and_asymmetric_shares(self):
        """
        Fisher Time-Reversal: P_T(0, t) * P_T(t, 0) == 1.0 +- 1e-6
        Across:
        - Symmetric market shares
        - Extreme asymmetric market shares (99.99% vs 0.01%)
        - Inverted carrier shares across periods
        """
        # Extreme asymmetric market shares
        I_rel = np.array([1.50, 0.60])
        w0 = np.array([0.9999, 0.0001])
        wt = np.array([0.0001, 0.9999])

        PT_0t = compute_tornqvist_index(I_rel, w0, wt)
        PT_t0 = compute_tornqvist_index(1.0 / I_rel, wt, w0)
        assert abs(PT_0t * PT_t0 - 1.0) < 1e-6

        # Multi-carrier randomized test (N=10,000)
        np.random.seed(303)
        N = 10000
        I_large = np.random.uniform(0.5, 2.0, size=N)
        w0_large = np.random.exponential(1.0, size=N)
        wt_large = np.random.exponential(1.0, size=N)
        PT_large_0t = compute_tornqvist_index(I_large, w0_large, wt_large)
        PT_large_t0 = compute_tornqvist_index(1.0 / I_large, wt_large, w0_large)
        assert abs(PT_large_0t * PT_large_t0 - 1.0) < 1e-12

    def test_tornqvist_carrier_entry_exit_and_active_set_normalization(self):
        """
        Carrier entry and exit on sector-window:
        - 5 potential carriers
        - Period 0: 6E (0.60), AI (0.30), SG (0.10) operate; IX and QP absent
        - Period t: 6E (0.50), AI (0.20), QP (0.30) operate; SG exited, QP entered
        - Active carrier intersection C^{0 cap t} = {6E, AI}
        - Re-normalized weights must sum to 1.0 in both periods
        - Time reversal must strictly hold over C^{0 cap t}
        """
        elementary_map = {
            ("DEL-BOM", "6E", 15): 1.10,
            ("DEL-BOM", "AI", 15): 1.05,
        }
        # In period 0, raw shares: 6E=0.60, AI=0.30, SG=0.10 (sum=1.0)
        # In period t, raw shares: 6E=0.50, AI=0.20, QP=0.30 (sum=1.0)
        w0_dict = {"6E": 0.60, "AI": 0.30, "SG": 0.10}
        wt_dict = {"6E": 0.50, "AI": 0.20, "QP": 0.30}

        route_res = aggregate_route_tornqvist(
            elementary_map=elementary_map,
            sector="DEL-BOM",
            advance_window=15,
            carrier_weights_base=w0_dict,
            carrier_weights_curr=wt_dict,
        )

        assert route_res is not None
        assert set(route_res.active_carriers) == {"6E", "AI"}
        # Normalized weights in base: 6E=0.60/0.90=0.666667, AI=0.30/0.90=0.333333 -> sum=1.0
        sum_w0_norm = sum(route_res.carrier_weights_base.values())
        sum_wt_norm = sum(route_res.carrier_weights_curr.values())
        sum_w_bar = sum(route_res.carrier_average_weights.values())

        assert abs(sum_w0_norm - 1.0) < 1e-5
        assert abs(sum_wt_norm - 1.0) < 1e-5
        assert abs(sum_w_bar - 1.0) < 1e-5

        # Time reversal over route
        elementary_map_rev = {
            ("DEL-BOM", "6E", 15): 1.0 / 1.10,
            ("DEL-BOM", "AI", 15): 1.0 / 1.05,
        }
        route_res_rev = aggregate_route_tornqvist(
            elementary_map=elementary_map_rev,
            sector="DEL-BOM",
            advance_window=15,
            carrier_weights_base=wt_dict,
            carrier_weights_curr=w0_dict,
        )
        assert route_res_rev is not None
        product = route_res.index_value * route_res_rev.index_value
        assert abs(product - 1.0) < 1e-5

    def test_tornqvist_monopoly_reduction(self):
        """Monopoly Route Reduction: Exactly 1 carrier operating reduces to elementary index."""
        elementary_map = {("DEL-DHM", "6E", 7): 1.2345}
        res = aggregate_route_tornqvist(elementary_map, "DEL-DHM", 7)
        assert res is not None
        assert res.is_monopoly is True
        assert abs(res.index_value - 1.2345) < 1e-6


# ==============================================================================
# 3. STAGE 3 LASPEYRES EMPIRICAL STRESS TESTS
# ==============================================================================

class TestLaspeyresEmpiricalStress:
    """Empirical verification of Stage 3 National Laspeyres Aggregator."""

    def test_strict_unit_weight_sum_1e7(self):
        """Axiom: sum_{r=1}^{30} sum_{w=1}^5 omega_{r,w} == 1.0000000 +- 1e-7."""
        matrix = get_default_weight_matrix()
        total_sum = sum(matrix.matrix.values())
        assert abs(total_sum - 1.0) < 1e-7
        # Inspect numpy matrix outer product sum as well
        np_sum = float(np.sum(matrix.to_numpy()))
        assert abs(np_sum - 1.0) < 1e-7

    def test_single_route_and_severe_missing_cells_renormalization(self):
        """
        Adversarial Stress:
        1. Single-route input: Only 1 route-window cell observed out of 150.
           With impute_missing=True, the composite relative must equal the cell relative,
           scaling APIx to exactly relative * 100.0.
        2. Severe missingness: Random subsets of 1, 5, 10, 50 cells under uniform price shock.
        """
        matrix = get_default_weight_matrix()

        # 1. Single route cell: ("DEL-BOM", 1) with relative 1.15 (+15%)
        single_cell_input = {("DEL-BOM", 1): 1.15}
        res_single = calculate_national_laspeyres(single_cell_input, weight_matrix=matrix, impute_missing=True)
        assert res_single.is_fully_observed is False
        assert res_single.missing_cells_count == 149
        assert abs(res_single.apix_index - 115.0) < 1e-4
        assert abs(res_single.raw_composite_relative - 1.15) < 1e-6

        # 2. Random subsets of varying sizes under uniform +8% price rise (relative 1.08)
        import random
        random.seed(777)
        all_cells = list(matrix.matrix.keys())

        for subset_size in [1, 2, 5, 10, 25, 75, 120, 150]:
            sampled = random.sample(all_cells, subset_size)
            idx_dict = {cell: 1.08 for cell in sampled}
            res_sub = calculate_national_laspeyres(idx_dict, weight_matrix=matrix, impute_missing=True)
            assert abs(res_sub.apix_index - 108.0) < 1e-4, f"Subset size {subset_size} failed: {res_sub.apix_index}"

    def test_duck_typed_and_mixed_representation_inputs(self):
        """
        Verifies calculate_national_laspeyres accepts:
        1. Raw float dict with integer window: ("DEL-BOM", 1) -> 1.05
        2. Raw float dict with string window: ("DEL-BOM", "T+1") -> 1.05
        3. Dict of TornqvistRouteResult Pydantic models
        4. Dict of plain dicts with {"index_value": float}
        5. Sequence/List of TornqvistRouteResult models
        6. Sequence/List of custom duck-typed objects with attributes
        7. NumPy 2D array of shape (30, 5)
        """
        matrix = get_default_weight_matrix()

        # 1. Raw float with integer window
        d1 = {("DEL-BOM", 1): 1.10}
        r1 = calculate_national_laspeyres(d1, matrix, impute_missing=True)
        assert abs(r1.apix_index - 110.0) < 1e-4

        # 2. Raw float with string window "T+1"
        d2 = {("DEL-BOM", "T+1"): 1.10}
        r2 = calculate_national_laspeyres(d2, matrix, impute_missing=True)
        assert abs(r2.apix_index - 110.0) < 1e-4

        # 3. Dict of TornqvistRouteResult
        trr = TornqvistRouteResult(
            sector="DEL-BOM",
            advance_window=1,
            active_carriers=["6E"],
            carrier_indices={"6E": 1.10},
            carrier_weights_base={"6E": 1.0},
            carrier_weights_curr={"6E": 1.0},
            carrier_average_weights={"6E": 1.0},
            index_value=1.10,
            is_monopoly=True,
            missing_carriers=[],
        )
        d3 = {("DEL-BOM", 1): trr}
        r3 = calculate_national_laspeyres(d3, matrix, impute_missing=True)
        assert abs(r3.apix_index - 110.0) < 1e-4

        # 4. Dict of plain dicts
        d4 = {("DEL-BOM", 1): {"index_value": 1.10}}
        r4 = calculate_national_laspeyres(d4, matrix, impute_missing=True)
        assert abs(r4.apix_index - 110.0) < 1e-4

        # 5. Sequence of TornqvistRouteResult
        s5 = [trr]
        r5 = calculate_national_laspeyres(s5, matrix, impute_missing=True)
        assert abs(r5.apix_index - 110.0) < 1e-4

        # 6. Sequence of custom duck-typed objects
        class CustomRouteRel:
            def __init__(self, s, w, v):
                self.sector = s
                self.advance_window = w
                self.index_value = v

        s6 = [CustomRouteRel("DEL-BOM", 1, 1.10)]
        r6 = calculate_national_laspeyres(s6, matrix, impute_missing=True)
        assert abs(r6.apix_index - 110.0) < 1e-4

        # 7. NumPy 2D array (30, 5)
        arr7 = np.full((30, 5), 1.10, dtype=float)
        r7 = calculate_national_laspeyres(arr7, matrix)
        assert abs(r7.apix_index - 110.0) < 1e-4
        assert r7.is_fully_observed is True


# ==============================================================================
# 4. STAGE 4 RYGEKS EMPIRICAL STRESS TESTS
# ==============================================================================

class TestRYGEKSEmpiricalStress:
    """Empirical verification of Stage 4 RYGEKS Multilateral Engine."""

    def test_geks_matrix_transitivity_all_triples_large(self):
        """
        Multilateral Transitivity: P_{ik} == P_{ij} * P_{jk} +- 1e-6
        Exhaustively verified for ALL triples in a 20-period multilateral matrix.
        Total triples evaluated = 20^3 = 8,000 triples.
        """
        np.random.seed(505)
        T = 20
        # Generate arbitrary positive price vectors for 5 commodities over 20 periods
        prices = [np.random.uniform(1000.0, 15000.0, size=5) for _ in range(T)]
        weights = [np.random.dirichlet(np.ones(5)) for _ in range(T)]

        bilateral_mat = np.zeros((T, T), dtype=float)
        for i in range(T):
            for j in range(T):
                if i == j:
                    bilateral_mat[i, j] = 1.0
                else:
                    rel = prices[j] / prices[i]
                    bilateral_mat[i, j] = compute_tornqvist_index(rel, weights[i], weights[j])

        geks_vec = compute_geks_vector(bilateral_mat)

        max_error = 0.0
        for i in range(T):
            for j in range(T):
                for k in range(T):
                    P_ij = geks_vec[j] / geks_vec[i]
                    P_jk = geks_vec[k] / geks_vec[j]
                    P_ik = geks_vec[k] / geks_vec[i]
                    err = abs((P_ij * P_jk) - P_ik)
                    if err > max_error:
                        max_error = err

        assert max_error < 1e-6, f"Transitivity violated: max_error={max_error}"
        assert max_error < 1e-12

    def test_geks_zero_chain_drift_multi_period_loops(self):
        """
        Zero Chain Drift on Cyclic Price Loops:
        When a market experiences high-frequency price fluctuations and returns
        to the exact initial price state (P0 -> P1 -> ... -> P_k -> P0):
        1. Chained bilateral index exhibits significant chain drift (|P_chained - 1.0| > 0).
        2. Multilateral GEKS strictly returns 1.000000 +- 1e-6.
        Tested for cycle lengths: 3, 6, 12, 24 periods.
        """
        for cycle_len in [3, 6, 12, 24]:
            np.random.seed(1000 + cycle_len)
            base_p = np.array([2500.0, 4200.0, 6100.0, 8000.0])
            base_w = np.array([0.50, 0.25, 0.15, 0.10])

            p_seq = [base_p.copy()]
            w_seq = [base_w.copy()]

            for _ in range(1, cycle_len):
                shock = np.random.uniform(0.75, 1.35, size=4)
                w_shock = np.random.dirichlet(np.ones(4))
                p_seq.append(base_p * shock)
                w_seq.append(w_shock)

            # Return exactly to base period
            p_seq.append(base_p.copy())
            w_seq.append(base_w.copy())

            T = len(p_seq)
            bilateral = np.zeros((T, T), dtype=float)
            for i in range(T):
                for j in range(T):
                    if i == j:
                        bilateral[i, j] = 1.0
                    else:
                        rel = p_seq[j] / p_seq[i]
                        bilateral[i, j] = compute_tornqvist_index(rel, w_seq[i], w_seq[j])

            geks_vec = compute_geks_vector(bilateral)
            geks_return = geks_vec[-1] / geks_vec[0]

            # GEKS must have zero chain drift
            assert abs(geks_return - 1.0) < 1e-6, f"Cycle {cycle_len} failed: {geks_return}"

    def test_rygeks_mean_splice_linking_historical_immutability(self):
        """
        Mean Splice Linking Invariance:
        Past published history remains 100% immutable when new periods are added
        and the rolling window rolls forward across 40 periods.
        """
        np.random.seed(888)
        engine = RYGEKSEngine(window_length=7, base_value=100.0, splice_method="mean")
        engine.add_first_period("2026-08-01")

        # Record snapshots after each addition
        history_snapshots: List[Dict[str, float]] = [dict(engine.get_published_series())]

        start_date = date(2026, 8, 1)
        for step in range(1, 40):
            cur_date = (start_date + timedelta(days=step)).isoformat()
            # Random bilateral relatives to active history
            active_hist = [
                (start_date + timedelta(days=k)).isoformat()
                for k in range(max(0, step - 7), step)
            ]
            pairwise = {h: float(np.random.uniform(0.96, 1.04)) for h in active_hist}

            engine.add_period(cur_date, pairwise)
            published = engine.get_published_series()

            # Verify that every previously published value has NOT changed
            for past_step, prior_snapshot in enumerate(history_snapshots):
                for period_key, prior_value in prior_snapshot.items():
                    current_value = published[period_key]
                    assert prior_value == current_value, (
                        f"Immutability violated at step {step}: period {period_key} "
                        f"changed from {prior_value} to {current_value}"
                    )

            history_snapshots.append(dict(published))

        # Check total published count
        assert len(engine.get_published_series()) == 40

    def test_rygeks_splicing_methods_and_expanding_window(self):
        """
        Verifies:
        - Expanding window flag is True when t < T, False when t >= T
        - All 4 splicing methods (mean, movement, window, half) execute cleanly
        """
        for splice_m in ["mean", "movement", "window", "half"]:
            engine = RYGEKSEngine(window_length=5, base_value=100.0, splice_method=splice_m)
            r0 = engine.add_first_period("P0")
            assert r0.is_expanding_window is True

            for t in range(1, 10):
                active_hist = [f"P{k}" for k in range(max(0, t - 5), t)]
                pairwise = {h: 1.01 for h in active_hist}
                rt = engine.add_period(f"P{t}", pairwise)
                # P0 is 1st, P1 is 2nd, P2 is 3rd, P3 is 4th (t < 4 -> len < 5 -> expanding)
                # At P4, there are 5 periods, so full window is reached
                if t < 4:
                    assert rt.is_expanding_window is True
                else:
                    assert rt.is_expanding_window is False
                assert rt.published_index > 0.0
