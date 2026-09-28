#!/usr/bin/env python3
"""
Standalone 30-Day DGCA Backtesting CLI Runner for APIx (Module C).
Executes the retrospective econometric backtest against DGCA domestic airfares,
evaluates all 4 acceptance hurdles, and prints formatted scorecards and JSON reports.

Usage:
    python run_backtest.py [--days 30] [--seed 42] [--format table|json|both] [--output report.json] [--strict]
"""

import argparse
import json
import sys
from datetime import date
from typing import Any

from apix.econometric.backtest import BacktestEngine, run_30day_backtest


def print_scorecard(report: Any) -> None:
    """Renders formatted ASCII/ANSI scorecard for MoSPI / NSO certification."""
    m = report.metrics
    print("=" * 80)
    print("           APIx ECONOMETRIC BACKTESTING ENGINE & MoSPI SCORECARD")
    print("                     Problem Statement ID: 26056")
    print("=" * 80)
    print(f"Evaluation Window : {report.start_date} to {report.end_date} ({report.validation_window_days} Days)")
    print(f"Random Seed       : {report.seed_used}")
    print(f"Benchmark Source  : DGCA Domestic Scheduled Airfare Statistics")
    print("-" * 80)
    print("ACCEPTANCE HURDLE VALIDATION SCORECARD")
    print("-" * 80)
    print(f"{'Metric':<32} {'Observed':<11} {'Threshold':<11} {'Status':<10} {'Margin'}")
    print("-" * 80)

    # 1. R^2
    r2_status = "[PASSED]" if m.r_squared_passed else "[FAILED]"
    r2_margin = f"+{((m.r_squared - m.r_squared_threshold) / m.r_squared_threshold * 100):.2f}%"
    print(f"{'1. Determination Coeff (R^2)':<32} {m.r_squared:<11.4f} {'> 0.8500':<11} {r2_status:<10} {r2_margin}")

    # 2. MAPE
    mape_status = "[PASSED]" if m.mape_passed else "[FAILED]"
    mape_margin = f"-{((m.mape_threshold - m.mape_pct) / m.mape_threshold * 100):.2f}%"
    print(f"{'2. Mean Absolute % Error (MAPE)':<32} {f'{m.mape_pct:.2f}%':<11} {'< 15.00%':<11} {mape_status:<10} {mape_margin}")

    # 3. Concordance
    conc_status = "[PASSED]" if m.directional_concordance_passed else "[FAILED]"
    conc_margin = f"+{(m.directional_concordance_pct - m.directional_concordance_threshold):.2f}%"
    print(f"{'3. CPI Directional Concordance':<32} {f'{m.directional_concordance_pct:.1f}%':<11} {'>= 80.00%':<11} {conc_status:<10} {conc_margin}")

    # 4. Fill Rate
    fill_status = "[PASSED]" if m.data_fill_rate_passed else "[FAILED]"
    fill_margin = f"+{(m.data_fill_rate_pct - m.data_fill_rate_threshold):.2f}%"
    print(f"{'4. Data Completeness Fill Rate':<32} {f'{m.data_fill_rate_pct:.1f}%':<11} {'>= 95.00%':<11} {fill_status:<10} {fill_margin}")

    # 5. RMSE
    print(f"{'5. Root Mean Square Error (RMSE)':<32} {f'INR {m.rmse_inr:.2f}':<11} {'N/A':<11} {'[INFO]':<10} {'N/A'}")

    print("-" * 80)
    verdict_badge = f"[{report.overall_validation_verdict}]"
    print(f"OVERALL MoSPI / NSO CERTIFICATION VERDICT: {verdict_badge}")
    if report.overall_validation_verdict == "PASSED":
        print("All 4 mandatory econometric acceptance hurdles successfully satisfied.")
    else:
        print("WARNING: One or more econometric acceptance hurdles failed!")
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="APIx 30-Day DGCA Backtesting CLI Runner")
    parser.add_argument("--days", type=int, default=30, help="Backtest window in days (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for simulation (default: 42)")
    parser.add_argument("--start-date", type=str, default="2026-08-29", help="Start date YYYY-MM-DD (default: 2026-08-29)")
    parser.add_argument(
        "--format",
        choices=["table", "json", "both"],
        default="both",
        help="Output display format (default: both)"
    )
    parser.add_argument("--output", type=str, default=None, help="File path to save JSON scorecard")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with non-zero exit code if any acceptance hurdle fails"
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose banners")

    args = parser.parse_args()

    # Execute backtest
    report = run_30day_backtest(
        window_days=args.days,
        seed=args.seed,
        start_date=args.start_date,
    )

    # Output rendering
    if args.format in ("table", "both"):
        print_scorecard(report)

    report_dict = report.model_dump()

    if args.format in ("json", "both"):
        json_str = json.dumps(report_dict, indent=2)
        if args.format == "json":
            print(json_str)

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(report_dict, f, indent=2)
        if not args.quiet:
            print(f"Backtest report successfully saved to: {args.output}")

    # Return exit code
    if args.strict and report.overall_validation_verdict != "PASSED":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
