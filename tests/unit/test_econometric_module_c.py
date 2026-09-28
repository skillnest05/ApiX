"""
Unit & Integration Test Suite for Module C: Econometric Index Engine.
Verifies all submodules in apix/econometric:
- jevons.py (Stage 1 elementary Jevons index)
- tornqvist.py (Stage 2 superlative Törnqvist index)
- laspeyres.py (Stage 3 modified Laspeyres national index & NationalWeightMatrix)
- rygeks.py (Multilateral Rolling Year GEKS engine & Mean Splice linking)
- publication.py (Daily, Weekly, Monthly index publication series)
- backtest.py (30-day retrospective DGCA backtesting engine)
- __init__.py (Module C public interface contracts)
"""

from datetime import date, datetime, timedelta
import math
import numpy as np
import pytest

from apix.econometric import (
    calculate_daily_index,
    calculate_rygeks_series,
    run_30day_backtest,
    # Jevons
    JevonsCohortResult,
    compute_geometric_mean,
    compute_jevons_index,
    calculate_cohort_jevons,
    compute_all_elementary_indices,
    JevonsEngine,
    # Törnqvist
    TornqvistRouteResult,
    DEFAULT_DGCA_CARRIER_SHARES,
    load_carrier_market_shares,
    compute_tornqvist_index,
    aggregate_route_tornqvist,
    compute_all_route_indices,
    TornqvistEngine,
    # Laspeyres
    Stage3Result,
    NationalWeightMatrix,
    get_default_weight_matrix,
    load_national_weights,
    calculate_national_laspeyres,
    LaspeyresEngine,
    # RYGEKS
    RYGEKSResult,
    compute_geks_vector,
    compute_geks_pairwise,
    splice_movement,
    splice_window,
    splice_half,
    splice_mean,
    RYGEKSEngine,
    # Publication
    DailyIndexResult,
    WeeklyIndexResult,
    MonthlyIndexResult,
    PublicationEngine,
    # Backtest
    BacktestMetrics,
    BacktestReport,
    BacktestEngine,
    simulate_30day_series,
    compute_r_squared,
    compute_mape,
    compute_directional_concordance,
    compute_rmse,
)


# ==============================================================================
# 1. JEVONS ELEMENTARY INDEX TESTS
# ==============================================================================

class TestJevonsModule:
    """Detailed unit tests for jevons.py."""

    def test_geometric_mean_basic(self):
        prices = [100.0, 200.0, 400.0, 800.0]
        # (100 * 200 * 400 * 800)^(1/4) = (6.4e9)^(0.25) = 282.8427
        gm = compute_geometric_mean(prices)
        assert abs(gm - 282.8427) < 1e-3

    def test_geometric_mean_invalid(self):
        with pytest.raises(ValueError):
            compute_geometric_mean([])
        with pytest.raises(ValueError):
            compute_geometric_mean([100.0, -50.0])
        with pytest.raises(ValueError):
            compute_geometric_mean([100.0, 0.0])

    def test_jevons_index_different_lengths(self):
        """When cohort sizes differ, evaluates ratio of geometric means."""
        p0 = [4000.0, 5000.0]
        pt = [4400.0, 5500.0, 6600.0]
        idx = compute_jevons_index(p0, pt)
        gm_0 = compute_geometric_mean(p0)
        gm_t = compute_geometric_mean(pt)
        assert abs(idx - (gm_t / gm_0)) < 1e-6

        # Time-reversal holds even with unequal sizes
        idx_rev = compute_jevons_index(pt, p0)
        assert abs(idx * idx_rev - 1.0) < 1e-6

    def test_calculate_cohort_jevons_with_mock_quotes(self):
        """Tests cohort filtering and calculation from quote objects."""
        base_quotes = [
            {"sector": "DEL-BOM", "carrier_code": "6E", "advance_days": 7, "flight_number": "6E-101", "departure_time": "06:00", "total_fare": 4500.0},
            {"sector": "DEL-BOM", "carrier_code": "6E", "advance_days": 7, "flight_number": "6E-102", "departure_time": "12:00", "total_fare": 5000.0},
        ]
        curr_quotes = [
            {"sector": "DEL-BOM", "carrier_code": "6E", "advance_days": 7, "flight_number": "6E-101", "departure_time": "06:00", "total_fare": 4950.0},
            {"sector": "DEL-BOM", "carrier_code": "6E", "advance_days": 7, "flight_number": "6E-102", "departure_time": "12:00", "total_fare": 5500.0},
        ]
        res = calculate_cohort_jevons(
            base_quotes=base_quotes,
            current_quotes=curr_quotes,
            sector="DEL-BOM",
            carrier_code="6E",
            advance_window=7,
        )
        assert res is not None
        assert res.sector == "DEL-BOM"
        assert res.carrier_code == "6E"
        assert res.advance_window == 7
        assert res.base_quote_count == 2
        assert res.current_quote_count == 2
        assert res.matched_quote_count == 2
        # Fares both increased by +10%
        assert abs(res.index_value - 1.10) < 1e-5

    def test_jevons_engine_wrapper(self):
        p0 = [3000.0, 4000.0]
        pt = [3300.0, 4400.0]
        assert abs(JevonsEngine.compute_index(p0, pt) - 1.10) < 1e-6


# ==============================================================================
# 2. TÖRNQVIST SUPERLATIVE INDEX TESTS
# ==============================================================================

class TestTornqvistModule:
    """Detailed unit tests for tornqvist.py."""

    def test_carrier_market_shares_loading(self):
        shares = load_carrier_market_shares()
        assert "6E" in shares
        assert "AI" in shares
        assert shares["6E"] >= 0.50

    def test_compute_tornqvist_index_validation(self):
        with pytest.raises(ValueError):
            compute_tornqvist_index([], [], [])
        with pytest.raises(ValueError):
            compute_tornqvist_index([1.0], [0.5, 0.5], [0.5, 0.5])
        with pytest.raises(ValueError):
            compute_tornqvist_index([-1.0], [1.0], [1.0])
        with pytest.raises(ValueError):
            compute_tornqvist_index([1.0], [0.0], [1.0])

    def test_aggregate_route_tornqvist_monopoly(self):
        elem_map = {
            ("DEL-BOM", "6E", 15): JevonsCohortResult(
                sector="DEL-BOM", carrier_code="6E", advance_window=15,
                base_quote_count=5, current_quote_count=5, matched_quote_count=5,
                base_geometric_mean=4000.0, current_geometric_mean=4400.0, index_value=1.10,
            )
        }
        res = aggregate_route_tornqvist(elem_map, "DEL-BOM", 15)
        assert res is not None
        assert res.is_monopoly is True
        assert res.active_carriers == ["6E"]
        assert abs(res.index_value - 1.10) < 1e-6

    def test_tornqvist_engine_wrapper(self):
        I_rel = [1.05, 1.15]
        w0 = [0.7, 0.3]
        wt = [0.65, 0.35]
        pt = TornqvistEngine.compute_index(I_rel, w0, wt)
        assert 1.05 < pt < 1.15


# ==============================================================================
# 3. LASPEYRES NATIONAL INDEX & WEIGHT MATRIX TESTS
# ==============================================================================

class TestLaspeyresModule:
    """Detailed unit tests for laspeyres.py."""

    def test_national_weight_matrix_dimensions(self):
        nwm = get_default_weight_matrix()
        assert len(nwm.sectors) == 30
        assert len(nwm.DEFAULT_WINDOWS) == 5
        mat = nwm.to_numpy()
        assert mat.shape == (30, 5)
        assert abs(np.sum(mat) - 1.0) < 1e-7

    def test_laspeyres_partial_observation_renormalization(self):
        """Missing cells are renormalized so total weight sums to 1.0 without downward bias."""
        nwm = get_default_weight_matrix()
        # Provide only 10 cells out of 150, all with price relative 1.05 (+5%)
        sample_indices = {}
        for s in nwm.sectors[:2]:
            for w in nwm.DEFAULT_WINDOWS:
                sample_indices[(s, w)] = 1.05

        res = calculate_national_laspeyres(sample_indices, weight_matrix=nwm, impute_missing=True)
        assert res.is_fully_observed is False
        assert res.missing_cells_count == 140
        # Renormalized index should be exactly 105.00 (+5%)
        assert abs(res.apix_index - 105.00) < 1e-3

    def test_laspeyres_engine_wrapper(self):
        nwm = get_default_weight_matrix()
        engine = LaspeyresEngine(weight_matrix=nwm)
        ones_mat = np.ones((30, 5))
        res = engine.compute_national_index(ones_mat)
        assert abs(res.apix_index - 100.0) < 1e-6


# ==============================================================================
# 4. MULTILATERAL RYGEKS ENGINE TESTS
# ==============================================================================

class TestRYGEKSModule:
    """Detailed unit tests for rygeks.py."""

    def test_geks_vector_identity(self):
        # 3x3 identity (no price changes)
        P_T = np.ones((3, 3))
        g = compute_geks_vector(P_T)
        assert len(g) == 3
        assert np.allclose(g, 1.0)

    def test_geks_transitivity_direct(self):
        P_T = np.array([
            [1.0, 1.08, 1.15],
            [1.0 / 1.08, 1.0, 1.0648],
            [1.0 / 1.15, 1.0 / 1.0648, 1.0]
        ])
        g = compute_geks_vector(P_T)
        p_01 = g[1] / g[0]
        p_12 = g[2] / g[1]
        p_02 = g[2] / g[0]
        assert abs((p_01 * p_12) - p_02) < 1e-6

    def test_rygeks_engine_stateful_lifecycle(self):
        """Tests advancing rolling window and non-revisability."""
        engine = RYGEKSEngine(window_length=4, base_value=100.0, splice_method="mean")
        r0 = engine.add_first_period("2026-01")
        assert r0.published_index == 100.0

        r1 = engine.add_period("2026-02", {"2026-01": 1.02})
        assert r1.published_index == 102.0

        r2 = engine.add_period("2026-03", {"2026-01": 1.05, "2026-02": 1.03})
        assert r2.published_index > 102.0

        # Non-revisability check: earlier periods remain unchanged
        pub = engine.get_published_series()
        assert pub["2026-01"] == 100.0
        assert pub["2026-02"] == 102.0

    def test_calculate_rygeks_series_interface(self):
        res = calculate_rygeks_series("2026-09-01", "2026-09-10", window_length=7)
        assert len(res) == 10
        assert res[0].published_index == 100.0
        for r in res:
            assert isinstance(r, RYGEKSResult)


# ==============================================================================
# 5. PUBLICATION ENGINE TESTS
# ==============================================================================

class TestPublicationModule:
    """Detailed unit tests for publication.py."""

    def test_calculate_daily_index(self):
        res = calculate_daily_index("2026-09-27")
        assert isinstance(res, DailyIndexResult)
        assert res.computation_date == "2026-09-27"
        assert res.apix_value > 100.0
        assert res.tier_summary.trunk > 100.0
        assert res.advance_window_summary.T_1 > res.advance_window_summary.T_45

    def test_calculate_weekly_index(self):
        res = PublicationEngine().calculate_weekly_index("2026-09-27")
        assert isinstance(res, WeeklyIndexResult)
        assert res.apix_value > 100.0
        assert res.window_start_date == "2026-09-21"
        assert res.window_end_date == "2026-09-27"

    def test_calculate_monthly_index(self):
        res = PublicationEngine().calculate_monthly_index("2026-09")
        assert isinstance(res, MonthlyIndexResult)
        assert res.computation_month == "2026-09"
        assert res.apix_value > 100.0

    def test_get_history(self):
        points = PublicationEngine().get_history("2026-09-01", "2026-09-10")
        assert len(points) == 10
        for p in points:
            assert p.lower_confidence_95 < p.apix_composite < p.upper_confidence_95


# ==============================================================================
# 6. BACKTEST ENGINE & CLI CONTRACT TESTS
# ==============================================================================

class TestBacktestModule:
    """Detailed unit tests for backtest.py."""

    def test_simulate_30day_series(self):
        dgca, apix, cpi = simulate_30day_series(seed=42, window_days=30)
        assert len(dgca) == 30
        assert len(apix) == 30
        assert len(cpi) == 30
        assert np.all(dgca > 4000.0)

    def test_backtest_engine_run(self):
        report = run_30day_backtest(window_days=30, seed=42)
        assert isinstance(report, BacktestReport)
        assert report.overall_validation_verdict == "PASSED"
        assert report.metrics.r_squared > 0.85
        assert report.metrics.mape_pct < 15.0
        assert report.metrics.directional_concordance_pct >= 80.0
        assert report.metrics.data_fill_rate_pct >= 95.0
        assert len(report.time_series_comparison) == 30

    def test_rmse_metric(self):
        act = [100.0, 110.0]
        pred = [102.0, 108.0]
        # (4 + 4)/2 = 4 -> sqrt(4) = 2.0
        assert abs(compute_rmse(act, pred) - 2.0) < 1e-6
