"""
Multi-Seed Robustness Sweep & End-to-End Integration Verification Harness.
Evaluates econometric index consistency across varied stochastic seeds.
"""

import os
import subprocess
import sys
import tempfile
from datetime import date, timedelta
from typing import List, Dict, Any

from apix.econometric.backtest import BacktestEngine
from apix.econometric.publication import PublicationEngine, DailyIndexResult
from apix.ingestion.seed_engine import SeedPlaybackEngine
from apix.pipeline.storage import StorageEngine


def run_seed_sweep(seeds: List[int]) -> List[Dict[str, Any]]:
    results = []
    print(f"{'Seed':<6} | {'R^2':<8} | {'MAPE (%)':<10} | {'Concordance':<12} | {'Fill Rate':<10} | {'RMSE (INR)':<11} | {'CLI --strict':<14} | {'Verdict':<8}")
    print("-" * 95)

    for s in seeds:
        # 1. BacktestEngine programmatic run
        rep = BacktestEngine.run_backtest(window_days=30, seed=s)
        m = rep.metrics

        # 2. CLI runner invocation with --strict flag
        cmd = [
            sys.executable,
            "run_backtest.py",
            "--seed", str(s),
            "--days", "30",
            "--strict",
            "--format", "json",
            "--quiet",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        strict_exit_code = proc.returncode

        # Verification hurdles
        r2_ok = m.r_squared > 0.85
        mape_ok = m.mape_pct < 15.0
        conc_ok = m.directional_concordance_pct >= 80.0
        fill_ok = m.data_fill_rate_pct >= 95.0
        exit_ok = (strict_exit_code == 0)
        verdict_ok = (rep.overall_validation_verdict == "PASSED")

        all_ok = r2_ok and mape_ok and conc_ok and fill_ok and exit_ok and verdict_ok

        row = {
            "seed": s,
            "r2": m.r_squared,
            "mape": m.mape_pct,
            "concordance": m.directional_concordance_pct,
            "fill_rate": m.data_fill_rate_pct,
            "rmse": m.rmse_inr,
            "strict_exit_code": strict_exit_code,
            "verdict": rep.overall_validation_verdict,
            "all_hurdles_passed": all_ok,
        }
        results.append(row)

        print(
            f"{s:<6} | {m.r_squared:<8.4f} | {m.mape_pct:<10.2f} | "
            f"{m.directional_concordance_pct:<11.1f}% | {m.data_fill_rate_pct:<9.1f}% | "
            f"{m.rmse_inr:<11.2f} | {strict_exit_code:<14} | {rep.overall_validation_verdict:<8}"
        )

    return results


def verify_e2e_pipeline() -> bool:
    print("\n" + "=" * 80)
    print("END-TO-END INTEGRATION TEST: SeedPlaybackEngine -> StorageEngine -> calculate_daily_index")
    print("=" * 80)

    temp_db = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    temp_db_path = temp_db.name
    temp_db.close()
    db_url = f"sqlite:///{temp_db_path}"

    try:
        storage = StorageEngine(db_url=db_url)
        seed_engine = SeedPlaybackEngine(seed=42)

        base_date = date(2026, 9, 1)
        target_date = date(2026, 9, 15)

        print(f"Generating full basket using SeedPlaybackEngine...")
        # Generate full basket for base date
        basket_base = seed_engine.generate_full_basket(base_scrape_date=base_date)
        print(f"Synthesized base basket: {len(basket_base)} quotes")

        # Generate full basket for target date
        basket_target = seed_engine.generate_full_basket(base_scrape_date=target_date)
        print(f"Synthesized target basket: {len(basket_target)} quotes")

        # Insert quotes into StorageEngine
        inserted_base = storage.insert_quotes(basket_base)
        inserted_target = storage.insert_quotes(basket_target)
        print(f"Inserted into StorageEngine: {inserted_base} base quotes, {inserted_target} target quotes")

        # Instantiate PublicationEngine with storage
        pub = PublicationEngine(storage_engine=storage)

        # Call calculate_daily_index
        res = pub.calculate_daily_index(target_date=target_date, base_date=base_date)

        print("\nDailyIndexResult Inspection:")
        print(f"  computation_date: {res.computation_date}")
        print(f"  frequency: {res.frequency}")
        print(f"  apix_value: {res.apix_value}")
        print(f"  base_period: {res.base_period}")
        print(f"  change_dod_pct: {res.change_dod_pct}%")
        print(f"  change_waw_pct: {res.change_waw_pct}%")
        print(f"  change_mom_pct: {res.change_mom_pct}%")
        print(f"  total_routes_evaluated: {res.total_routes_evaluated}")
        print(f"  total_quotes_aggregated: {res.total_quotes_aggregated}")
        print(f"  tier_summary: {res.tier_summary}")
        print(f"  advance_window_summary: {res.advance_window_summary}")
        print(f"  sector_indices sample: {dict(list(res.sector_indices.items())[:3]) if res.sector_indices else None}")

        # Assertions
        assert isinstance(res, DailyIndexResult), "Result must be DailyIndexResult instance"
        assert res.apix_value > 0.0, f"APIx index value must be > 0, got {res.apix_value}"
        assert res.total_quotes_aggregated >= 10, f"Total quotes aggregated must be >= 10, got {res.total_quotes_aggregated}"
        assert res.total_routes_evaluated == 30, f"Total routes evaluated must be 30, got {res.total_routes_evaluated}"
        assert res.tier_summary.trunk > 0.0
        assert res.tier_summary.high_density > 0.0
        assert res.tier_summary.regional > 0.0
        assert res.tier_summary.seasonal > 0.0
        assert res.advance_window_summary.T_1 > 0.0
        assert res.advance_window_summary.T_7 > 0.0
        assert res.advance_window_summary.T_15 > 0.0
        assert res.advance_window_summary.T_30 > 0.0
        assert res.advance_window_summary.T_45 > 0.0
        assert isinstance(res.change_dod_pct, float)
        assert isinstance(res.change_waw_pct, float)
        assert isinstance(res.change_mom_pct, float)

        print("\nAll End-to-End assertions PASSED successfully!")
        return True

    finally:
        storage.engine.dispose()
        try:
            os.remove(temp_db_path)
        except Exception:
            pass


def main():
    # Seeds to evaluate: specifically 30, 55, and wide sweep of random seeds
    seeds_to_test = [
        1, 2, 7, 10, 13, 21, 30, 42, 55, 77, 88, 99, 100, 123,
        256, 314, 500, 777, 888, 999, 1337, 2024, 2026, 7777, 9999
    ]

    print("=" * 95)
    print(f"ADVERSARIAL STRESS TEST: 30-Day DGCA Backtesting Across {len(seeds_to_test)} Seeds under --strict Mode")
    print("=" * 95)

    results = run_seed_sweep(seeds_to_test)
    failed_seeds = [r for r in results if not r["all_hurdles_passed"]]

    print("-" * 95)
    print(f"Total Seeds Evaluated: {len(results)}")
    print(f"Total Seeds Passed   : {len(results) - len(failed_seeds)}")
    print(f"Total Seeds Failed   : {len(failed_seeds)}")

    if failed_seeds:
        print(f"FAILED SEEDS DETAILS: {failed_seeds}")
        sys.exit(1)
    else:
        print("ALL SEEDS PASSED 100% OF HURDLES UNDER --strict MODE!")

    e2e_ok = verify_e2e_pipeline()
    if not e2e_ok:
        print("E2E PIPELINE FAILED!")
        sys.exit(1)

    print("\nOVERALL VERDICT: ALL SEED STRESS TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    main()
