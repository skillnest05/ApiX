"""
Module C: Econometric Index Construction & Backtesting Engine for APIx.

Implements the three-stage aggregation hierarchy and multilateral rolling index:
1. Stage 1: Elementary Jevons Index (log-space geometric mean of price relatives per cohort)
2. Stage 2: Superlative Törnqvist Index (DGCA carrier market shares, active set normalization)
3. Stage 3: Modified Laspeyres National Composite (30 sectors x 5 windows, unit-sum normalization)
4. Multilateral RYGEKS with Mean Splice (CCDI formulation, transitive, zero chain drift, non-revisable)
5. Multi-frequency Publication Series (Daily, Weekly, Monthly)
6. 30-Day Retrospective DGCA Backtesting Engine (R^2 > 0.85, MAPE < 15%, Concordance >= 80%)

Public Interface Contracts:
- calculate_daily_index(target_date) -> DailyIndexResult
- calculate_rygeks_series(start_date, end_date) -> List[RYGEKSResult]
- run_30day_backtest(window_days=30, seed=42, start_date=None) -> BacktestReport
"""

from apix.econometric.jevons import (
    JevonsCohortResult,
    compute_geometric_mean,
    compute_jevons_index,
    calculate_cohort_jevons,
    compute_all_elementary_indices,
    JevonsEngine,
)

from apix.econometric.tornqvist import (
    TornqvistRouteResult,
    DEFAULT_DGCA_CARRIER_SHARES,
    load_carrier_market_shares,
    compute_tornqvist_index,
    aggregate_route_tornqvist,
    compute_all_route_indices,
    TornqvistEngine,
)

from apix.econometric.laspeyres import (
    Stage3Result,
    NationalWeightMatrix,
    get_default_weight_matrix,
    load_national_weights,
    calculate_national_laspeyres,
    LaspeyresEngine,
)

from apix.econometric.rygeks import (
    RYGEKSResult,
    compute_geks_vector,
    compute_geks_pairwise,
    splice_movement,
    splice_window,
    splice_half,
    splice_mean,
    RYGEKSEngine,
    calculate_rygeks_series,
)

from apix.econometric.publication import (
    TierSummary,
    AdvanceWindowSummary,
    DailyIndexResult,
    WeeklyIndexResult,
    MonthlyIndexResult,
    IndexHistoryPoint,
    PublicationEngine,
    get_default_publication_engine,
    calculate_daily_index,
    calculate_weekly_index,
    calculate_monthly_index,
    get_index_history,
)

from apix.econometric.backtest import (
    BacktestMetrics,
    BacktestDataPoint,
    BacktestReport,
    BacktestEngine,
    simulate_30day_series,
    compute_r_squared,
    compute_mape,
    compute_directional_concordance,
    compute_rmse,
    run_30day_backtest,
)

__all__ = [
    # Primary Module C Public Contracts
    "calculate_daily_index",
    "calculate_rygeks_series",
    "run_30day_backtest",
    # Stage 1 Jevons
    "JevonsCohortResult",
    "compute_geometric_mean",
    "compute_jevons_index",
    "calculate_cohort_jevons",
    "compute_all_elementary_indices",
    "JevonsEngine",
    # Stage 2 Törnqvist
    "TornqvistRouteResult",
    "DEFAULT_DGCA_CARRIER_SHARES",
    "load_carrier_market_shares",
    "compute_tornqvist_index",
    "aggregate_route_tornqvist",
    "compute_all_route_indices",
    "TornqvistEngine",
    # Stage 3 Laspeyres
    "Stage3Result",
    "NationalWeightMatrix",
    "get_default_weight_matrix",
    "load_national_weights",
    "calculate_national_laspeyres",
    "LaspeyresEngine",
    # Multilateral RYGEKS
    "RYGEKSResult",
    "compute_geks_vector",
    "compute_geks_pairwise",
    "splice_movement",
    "splice_window",
    "splice_half",
    "splice_mean",
    "RYGEKSEngine",
    # Publication
    "TierSummary",
    "AdvanceWindowSummary",
    "DailyIndexResult",
    "WeeklyIndexResult",
    "MonthlyIndexResult",
    "IndexHistoryPoint",
    "PublicationEngine",
    "get_default_publication_engine",
    "calculate_weekly_index",
    "calculate_monthly_index",
    "get_index_history",
    # Backtest
    "BacktestMetrics",
    "BacktestDataPoint",
    "BacktestReport",
    "BacktestEngine",
    "simulate_30day_series",
    "compute_r_squared",
    "compute_mape",
    "compute_directional_concordance",
    "compute_rmse",
]
