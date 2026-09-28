"""
APIx Turnkey Platform Runner & Orchestration Entrypoint.
Problem Statement ID: 26056
Ministry of Statistics and Programme Implementation (MoSPI)
Data Informatics & Innovation Division (DIID)

Usage:
    python run_apix.py                 # Seeds DB, verifies backtest, and launches server on http://localhost:8000
    python run_apix.py --seed          # Generates seed quotes and populates SQLite database
    python run_apix.py --backtest      # Executes 30-day DGCA retrospective backtest validation
    python run_apix.py --serve         # Starts the FastAPI web server & interactive dashboard
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logger = logging.getLogger("apix.runner")


def run_database_seeding(db_path: str = "apix.db") -> None:
    """Seeds dimensions and quotes into the SQLite database."""
    from apix.pipeline.storage import StorageEngine
    from apix.ingestion.seed_engine import SeedPlaybackEngine

    logger.info("Initializing APIx Storage Engine at %s...", db_path)
    engine = StorageEngine(f"sqlite:///{db_path}")

    seed_engine = SeedPlaybackEngine()
    quotes = seed_engine.generate_full_basket()
    logger.info("Generated %d synthetic/deterministic fare quotes.", len(quotes))

    inserted = engine.insert_quotes(quotes)
    logger.info("Successfully persisted %d fare quotes into fact table 'fare_quotes'.", inserted)


def run_dgca_backtest(window_days: int = 30) -> Dict[str, Any]:
    """Executes the 30-day retrospective DGCA backtesting module."""
    from apix.econometric.backtest import BacktestEngine

    logger.info("Executing %d-day DGCA retrospective backtesting...", window_days)
    report = BacktestEngine.run_backtest(window_days=window_days)
    report_dict = report.model_dump()

    print("\n" + "=" * 70)
    print("APIx -- 30-DAY RETROSPECTIVE DGCA VALIDATION REPORT")
    print("=" * 70)
    metrics = report_dict["metrics"]
    print(f"  * Correlation (R^2):             {metrics['r_squared']:.4f}  (Hurdle > 0.8500: {'PASSED [OK]' if metrics['r_squared_passed'] else 'FAILED'})")
    print(f"  * Mean Abs Pct Error (MAPE):     {metrics['mape_pct']:.2f}%   (Hurdle < 15.00%: {'PASSED [OK]' if metrics['mape_passed'] else 'FAILED'})")
    print(f"  * Root Mean Sq Error (RMSE):     INR {metrics['rmse_inr']:.2f}")
    print(f"  * Directional Concordance:       {metrics['directional_concordance_pct']:.1f}%   (Hurdle >= 80.0%: {'PASSED [OK]' if metrics['directional_concordance_passed'] else 'FAILED'})")
    print(f"  * Data Fill Completeness:        {metrics['data_fill_rate_pct']:.1f}%")
    print(f"  * Final Audit Verdict:           {report_dict['overall_validation_verdict']}")
    print("=" * 70 + "\n")

    return report_dict


def start_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Launches the Uvicorn ASGI server with the FastAPI app."""
    import uvicorn

    logger.info("Starting APIx Web Dashboard and REST API at http://%s:%d", host, port)
    print("\n" + "*" * 70)
    print(f"APIx Web Dashboard is LIVE at: http://{host}:{port}/")
    print(f"OpenAPI Swagger Documentation at: http://{host}:{port}/docs")
    print("*" * 70 + "\n")

    uvicorn.run("apix.api.app:app", host=host, port=port, log_level="info")


def main() -> None:
    default_host = os.getenv("APIX_HOST", "127.0.0.1")
    default_port = int(os.getenv("APIX_PORT", "8000"))

    parser = argparse.ArgumentParser(description="APIx Platform Turnkey Runner")
    parser.add_argument("--seed", action="store_true", help="Seed database with fare quotes")
    parser.add_argument("--backtest", action="store_true", help="Run 30-day DGCA backtest validation")
    parser.add_argument("--serve", action="store_true", help="Launch FastAPI web server and dashboard")
    parser.add_argument("--port", type=int, default=default_port, help=f"Web server port (default: {default_port})")
    parser.add_argument("--host", type=str, default=default_host, help=f"Web server host (default: {default_host})")

    args = parser.parse_args()

    # If no flags passed, execute complete turnkey sequence
    if not (args.seed or args.backtest or args.serve):
        logger.info("No specific mode specified. Running complete turnkey sequence...")
        run_database_seeding()
        run_dgca_backtest()
        start_server(host=args.host, port=args.port)
        return

    if args.seed:
        run_database_seeding()

    if args.backtest:
        run_dgca_backtest()

    if args.serve:
        start_server(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
