"""
Automated Scheduling Orchestrator for APIx (Module A)
Manages high-frequency and regular airfare price collection schedules using APScheduler:
1. Full Basket Scrape: 3x daily at 06:00, 12:00, 18:00 IST (all 30 sectors, 5 windows, 9 sources)
2. Trunk Volatility Monitor: Hourly (07:00-23:00 IST) for top 5 trunk routes (10 sectors x T+1, T+7)
"""

from datetime import date, datetime, timedelta
import logging
import os
from typing import Any, Callable, Dict, List, Optional
import zoneinfo

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from apix.ingestion.base import (
    RawFareQuote,
    APPROVED_SECTORS
)
from apix.ingestion.seed_engine import SeedPlaybackEngine
from apix.ingestion.airlines import AIRLINE_SCRAPERS, get_airline_scraper
from apix.ingestion.otas import OTA_SCRAPERS, get_ota_scraper

logger = logging.getLogger("apix.ingestion.scheduler")

# Standard Indian Standard Time (IST = UTC+05:30)
DEFAULT_TIMEZONE_STR = "Asia/Kolkata"

# Top 5 DGCA trunk pairs (10 bidirectional sectors)
TOP_TRUNK_SECTORS = [
    "DEL-BOM", "BOM-DEL",
    "DEL-BLR", "BLR-DEL",
    "BOM-BLR", "BLR-BOM",
    "DEL-HYD", "HYD-DEL",
    "DEL-CCU", "CCU-DEL"
]


class APIxIngestionScheduler:
    """
    Automated ingestion job manager orchestrating scheduled airfare extractions.
    Uses APScheduler with IST cron triggers.
    """
    def __init__(
        self,
        timezone_str: str = DEFAULT_TIMEZONE_STR,
        on_quotes_collected: Optional[Callable[[List[RawFareQuote], str], None]] = None,
        auto_start: bool = False
    ):
        self.timezone_str = timezone_str
        try:
            self.tz = zoneinfo.ZoneInfo(timezone_str)
        except Exception as e:
            logger.warning(f"Could not load zoneinfo for {timezone_str}: {e}. Falling back to UTC.")
            self.tz = zoneinfo.ZoneInfo("UTC")

        self.on_quotes_collected = on_quotes_collected
        self.scheduler = BackgroundScheduler(timezone=self.tz)
        self.seed_engine = SeedPlaybackEngine()
        self._is_running = False

        self._register_default_jobs()

        if auto_start:
            self.start()

    def _register_default_jobs(self) -> None:
        """Register the 3x daily full run and hourly trunk monitor jobs."""
        # 1. Full Basket Scrape: 06:00, 12:00, 18:00 IST
        self.scheduler.add_job(
            func=self.run_full_basket_scrape,
            trigger=CronTrigger(hour="6,12,18", minute=0, timezone=self.tz),
            id="full_basket_scrape",
            name="APIx 3x Daily Full Basket Scrape (06:00, 12:00, 18:00 IST)",
            replace_existing=True
        )

        # 2. Hourly Trunk Volatility Monitor: 07:00 - 23:00 IST
        self.scheduler.add_job(
            func=self.run_trunk_volatility_monitor,
            trigger=CronTrigger(hour="7-23", minute=0, timezone=self.tz),
            id="trunk_volatility_monitor",
            name="APIx Hourly Trunk Route Volatility Monitor (07:00-23:00 IST)",
            replace_existing=True
        )

        logger.info(f"Registered default APIx ingestion jobs with timezone {self.timezone_str}")

    def start(self) -> None:
        """Start the background scheduler."""
        if not self.scheduler.running:
            self.scheduler.start()
            self._is_running = True
            logger.info("APIx Ingestion Scheduler started.")

    def shutdown(self, wait: bool = False) -> None:
        """Shutdown the background scheduler."""
        if self.scheduler.running:
            self.scheduler.shutdown(wait=wait)
            self._is_running = False
            logger.info("APIx Ingestion Scheduler stopped.")

    def pause(self) -> None:
        """Pause all scheduled jobs."""
        self.scheduler.pause()
        logger.info("APIx Ingestion Scheduler paused.")

    def resume(self) -> None:
        """Resume all scheduled jobs."""
        self.scheduler.resume()
        logger.info("APIx Ingestion Scheduler resumed.")

    @property
    def is_running(self) -> bool:
        """Return True if scheduler is actively running."""
        return self.scheduler.running

    def get_jobs_status(self) -> List[Dict[str, Any]]:
        """Return summary of all registered jobs and their next fire times."""
        jobs_info = []
        for job in self.scheduler.get_jobs():
            next_run = getattr(job, "next_run_time", None)
            jobs_info.append({
                "job_id": job.id,
                "name": job.name,
                "next_run_time": next_run.isoformat() if next_run else None,
                "trigger": str(job.trigger)
            })
        return jobs_info

    def run_full_basket_scrape(
        self,
        base_date: Optional[date] = None,
        quotes_per_cohort: int = 1
    ) -> List[RawFareQuote]:
        """
        Execute full basket extraction covering all 30 sectors and 5 advance windows.
        """
        if base_date is None:
            base_date = datetime.now(self.tz).date()

        logger.info(f"Executing Full Basket Scrape for base date {base_date}...")
        quotes = self.seed_engine.generate_full_basket(
            base_scrape_date=base_date,
            quotes_per_cohort=quotes_per_cohort
        )

        if self.on_quotes_collected:
            try:
                self.on_quotes_collected(quotes, "full_basket_scrape")
            except Exception as e:
                logger.error(f"Callback error in run_full_basket_scrape: {e}")

        logger.info(f"Full Basket Scrape completed with {len(quotes)} quotes.")
        return quotes

    def run_trunk_volatility_monitor(
        self,
        base_date: Optional[date] = None
    ) -> List[RawFareQuote]:
        """
        Execute hourly trunk route monitoring for the top 5 city-pairs (10 sectors)
        across urgent windows (T+1, T+7).
        """
        if base_date is None:
            base_date = datetime.now(self.tz).date()

        logger.info(f"Executing Trunk Volatility Monitor for base date {base_date}...")
        trunk_quotes: List[RawFareQuote] = []

        # Urgent windows: T+1 and T+7
        urgent_windows = [1, 7]
        carriers = ["6E", "AI", "IX"]
        otas = ["makemytrip", "easemytrip"]

        for sector in TOP_TRUNK_SECTORS:
            orig, dest = sector.split("-")
            for adv in urgent_windows:
                t_date = base_date + timedelta(days=adv)
                # Airlines
                for c_code in carriers:
                    carrier_src = {"6E": "indigo", "AI": "airindia", "IX": "airindiaexpress"}[c_code]
                    q = self.seed_engine.synthesize_quote(
                        origin=orig,
                        destination=dest,
                        travel_date=t_date,
                        advance_days=adv,
                        carrier_code=c_code,
                        source=carrier_src
                    )
                    trunk_quotes.append(q)
                # OTAs
                for ota in otas:
                    q = self.seed_engine.synthesize_quote(
                        origin=orig,
                        destination=dest,
                        travel_date=t_date,
                        advance_days=adv,
                        carrier_code="6E",
                        source=ota
                    )
                    trunk_quotes.append(q)

        if self.on_quotes_collected:
            try:
                self.on_quotes_collected(trunk_quotes, "trunk_volatility_monitor")
            except Exception as e:
                logger.error(f"Callback error in run_trunk_volatility_monitor: {e}")

        logger.info(f"Trunk Volatility Monitor completed with {len(trunk_quotes)} quotes.")
        return trunk_quotes

    def trigger_job_now(self, job_id: str) -> List[RawFareQuote]:
        """Manually trigger a job immediately and return generated quotes."""
        if job_id == "full_basket_scrape":
            return self.run_full_basket_scrape()
        elif job_id == "trunk_volatility_monitor":
            return self.run_trunk_volatility_monitor()
        else:
            job = self.scheduler.get_job(job_id)
            if not job:
                raise ValueError(f"Job ID '{job_id}' not found in scheduler.")
            job.modify(next_run_time=datetime.now(self.tz))
            return []
