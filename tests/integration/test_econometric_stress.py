"""
Econometric Stress Test Suite: Index Construction & 30-Day Backtesting Robustness.

Tests empirical robustness:
1. Multi-seed 30-day backtesting stress test across 20 random seeds.
2. End-to-end integration: SeedPlaybackEngine -> StorageEngine -> calculate_daily_index -> calculate_rygeks_series.
"""

from datetime import date, timedelta
import os
import tempfile
import pytest

from apix.econometric.backtest import BacktestEngine, simulate_30day_series, compute_directional_concordance
from apix.econometric.publication import PublicationEngine, DailyIndexResult
from apix.econometric.rygeks import calculate_rygeks_series, RYGEKSResult
from apix.econometric.laspeyres import calculate_national_laspeyres, get_default_weight_matrix
from apix.econometric.tornqvist import TornqvistRouteResult
from apix.ingestion.seed_engine import SeedPlaybackEngine
from apix.pipeline.storage import StorageEngine


class TestEconometricStress:
    """Empirical Econometric Robustness and Backtest Stress Tests."""

    def test_multiseed_backtesting_stress_20_seeds_empirical(self):
        """
        Adversarial evaluation of 30-day backtest across 21 distinct seeds:
        [1, 2, 7, 13, 21, 30, 42, 55, 89, 99, 100, 123, 256, 314, 500, 777, 888, 999, 1337, 2024, 2026].
        Verifies that all seeds (including seeds 30 and 55) satisfy all 4 acceptance hurdles with 100% pass rate.
        """
        seeds = [1, 2, 7, 13, 21, 30, 42, 55, 89, 99, 100, 123, 256, 314, 500, 777, 888, 999, 1337, 2024, 2026]
        failures = []

        for s in seeds:
            report = BacktestEngine.run_backtest(window_days=30, seed=s)
            m = report.metrics
            if report.overall_validation_verdict != "PASSED":
                failures.append({
                    "seed": s,
                    "r2": m.r_squared,
                    "mape": m.mape_pct,
                    "concordance": m.directional_concordance_pct,
                    "fill_rate": m.data_fill_rate_pct,
                    "verdict": report.overall_validation_verdict,
                })

        # Remediated: 0 failures across all seeds
        assert len(failures) == 0, f"Failures detected across test seeds: {failures}"

    def test_e2e_storage_to_daily_index_typeerror_bug(self):
        """
        Verifies that passing real quotes through StorageEngine to calculate_daily_index
        seamlessly executes the live econometric pipeline without TypeError.
        """
        temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        temp_db_path = temp_db.name
        temp_db.close()
        db_url = f"sqlite:///{temp_db_path}"

        storage = StorageEngine(db_url=db_url)
        seed_engine = SeedPlaybackEngine(seed=42)

        try:
            d_base = date(2026, 9, 10)
            d_target = date(2026, 9, 11)

            # Generate quotes for both dates across multiple advance windows
            quotes = []
            for adv in [1, 7, 15, 30, 45]:
                for sec in ["DEL-BOM", "BOM-DEL", "DEL-BLR"]:
                    orig, dest = sec.split("-")
                    quotes.extend(seed_engine.generate_quotes_for_sector(orig, dest, travel_date=d_base, advance_days=adv))
                    quotes.extend(seed_engine.generate_quotes_for_sector(orig, dest, travel_date=d_target, advance_days=adv))

            storage.insert_quotes(quotes)
            pub = PublicationEngine(storage_engine=storage)

            # Must execute without raising TypeError and return valid DailyIndexResult
            res = pub.calculate_daily_index(target_date=d_target, base_date=d_base)
            assert isinstance(res, DailyIndexResult)
            assert res.apix_value > 0.0
            assert res.total_quotes_aggregated >= 10
            assert isinstance(res.change_dod_pct, float)
            assert isinstance(res.change_waw_pct, float)
            assert isinstance(res.change_mom_pct, float)
        finally:
            storage.engine.dispose()
            try:
                os.remove(temp_db_path)
            except Exception:
                pass

    def test_laspeyres_dict_of_objects_typeerror_direct(self):
        """
        Direct minimal unit verification of calculate_national_laspeyres
        when route_window_indices is a dict of TornqvistRouteResult.
        """
        nwm = get_default_weight_matrix()
        trr = TornqvistRouteResult(
            sector="DEL-BOM",
            advance_window=1,
            active_carriers=["6E"],
            carrier_indices={"6E": 1.05},
            carrier_weights_base={"6E": 1.0},
            carrier_weights_curr={"6E": 1.0},
            carrier_average_weights={"6E": 1.0},
            index_value=1.05,
            is_monopoly=True,
            missing_carriers=[],
        )
        route_map = {("DEL-BOM", 1): trr}

        res = calculate_national_laspeyres(route_map, weight_matrix=nwm, impute_missing=True)
        assert res.apix_index > 0.0
        assert res.is_fully_observed is False

    def test_e2e_storage_to_rygeks_series_pydantic_validation(self):
        """
        Tests end-to-end calculation of RYGEKS series via calculate_rygeks_series
        and verifies compliance with Pydantic schema RYGEKSResult.
        """
        results = calculate_rygeks_series(start_date="2026-09-01", end_date="2026-09-14", window_length=7)
        assert len(results) == 14
        for r in results:
            assert isinstance(r, RYGEKSResult)
            assert hasattr(r, "period_id")
            assert hasattr(r, "published_index")
            assert hasattr(r, "window_length")
            assert hasattr(r, "splice_method")
            assert hasattr(r, "level_adjustment_factor")
            assert hasattr(r, "is_expanding_window")
            assert r.published_index > 0.0
            assert r.window_length >= 1
            assert r.splice_method in ["base", "mean", "movement", "window", "half"]
