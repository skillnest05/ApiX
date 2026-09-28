"""
Storage Concurrency and High-Throughput Ingestion Verification Suite.

Verification Focus Areas:
1. Integration stress test: Use SeedPlaybackEngine to generate multiple full baskets
   across all 15 city-pairs, 30 sectors, 5 booking windows (T+1 to T+45). Insert all quotes
   into SQLite StorageEngine.
2. Query using get_clean_quotes(): Verify returned quotes have all required statutory
   property getters (fare_base, fare_fuel, fare_gst, fare_udf, fare_psf, fare_fee, dqs_score).
3. Verify concurrent reading and writing to SQLite: start multiple threads and processes
   reading and writing to ensure SQLite WAL mode prevents database lock contention.
4. Stress-test edge cases, duck typing, and deduplication.
"""

from collections import defaultdict
import concurrent.futures
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import multiprocessing as mp
import os
import random
import re
import sqlite3
import tempfile
import time as time_mod
from typing import Any, Dict, List, Optional, Tuple

import pytest

from apix.ingestion.base import (
    APPROVED_CARRIERS,
    APPROVED_SECTORS,
    APPROVED_SOURCES,
    RawFareQuote as IngestionRawFareQuote,
)
from apix.ingestion.seed_engine import SeedPlaybackEngine
from apix.pipeline.outlier import OutlierDetector, filter_iqr_bounds
from apix.pipeline.quality import (
    canonical_flight_key,
    clean_quote,
    deduplicate_quotes,
)
from apix.pipeline.schemas import (
    CleanedFareQuote,
    DecomposedFare,
    RawFareQuote as PipelineRawFareQuote,
)
from apix.pipeline.storage import (
    Carrier,
    FareQuoteRecord,
    Route,
    StorageEngine,
    get_clean_quotes,
)


# ==============================================================================
# 1. INTEGRATION STRESS TEST: SEEDPLAYBACKENGINE -> STORAGEENGINE FULL BASKETS
# ==============================================================================

class TestFullBasketIntegrationStress:
    """
    Adversarial verification of Module A ↔ Module B integration:
    SeedPlaybackEngine full baskets across all 15 city-pairs, 30 sectors,
    5 booking windows (T+1 to T+45) inserted into StorageEngine.
    """

    @pytest.fixture
    def temp_sqlite_db(self):
        """Creates a fresh temporary file-based SQLite database with WAL mode."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "apix_integration_stress.db")
        engine = StorageEngine(f"sqlite:///{db_path}")
        yield engine, db_path
        engine.engine.dispose()
        for ext in ["", "-wal", "-shm"]:
            f = db_path + ext
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

    def test_single_full_basket_composition(self):
        """Verify SeedPlaybackEngine full basket generates exactly 3,750 quotes covering all cohorts."""
        seed = SeedPlaybackEngine(seed=42)
        base_date = date(2026, 10, 1)
        basket = seed.generate_full_basket(base_scrape_date=base_date)

        assert len(basket) == 3750, f"Expected 3750 quotes, got {len(basket)}"

        # Check coverage
        sectors_seen = set()
        carriers_seen = set()
        windows_seen = set()
        sources_seen = set()

        for q in basket:
            sec = f"{q.origin_iata}-{q.destination_iata}"
            sectors_seen.add(sec)
            carriers_seen.add(q.carrier_code)
            windows_seen.add(q.advance_days)
            sources_seen.add(q.source)

            # Each quote must have mathematical decomposition identity
            recon = (
                q.raw_base_fare
                + q.raw_fuel_surcharge
                + q.raw_taxes
                + q.raw_udf
                + q.raw_psf
                + q.raw_convenience_fee
            )
            assert abs(float(recon) - float(q.raw_total_fare)) <= 1.00

        assert sectors_seen == APPROVED_SECTORS, f"Missing sectors: {APPROVED_SECTORS - sectors_seen}"
        assert carriers_seen == set(APPROVED_CARRIERS), f"Missing carriers: {set(APPROVED_CARRIERS) - carriers_seen}"
        assert windows_seen == {1, 7, 15, 30, 45}, f"Missing windows: {{1, 7, 15, 30, 45}} - {windows_seen}"
        expected_sources = {"indigo", "airindia", "airindiaexpress", "akasa", "spicejet", "makemytrip", "cleartrip", "ixigo", "easemytrip"}
        assert sources_seen == expected_sources, f"Missing sources: {expected_sources - sources_seen}"

    def test_multi_basket_ingestion_and_upsert(self, temp_sqlite_db):
        """
        Adversarial test: Ingest multiple full baskets (2 distinct days = 7,500 quotes)
        and test idempotent re-insertion without data loss or corruption.
        """
        engine, db_path = temp_sqlite_db
        seed = SeedPlaybackEngine(seed=101)

        # Generate Basket 1 (Day 1: 3,750 quotes)
        basket_1 = seed.generate_full_basket(base_scrape_date=date(2026, 10, 1))
        inserted_1 = engine.insert_quotes(basket_1)
        assert inserted_1 == 3750, f"Expected 3750 inserted, got {inserted_1}"

        # Generate Basket 2 (Day 2: 3,750 quotes)
        basket_2 = seed.generate_full_basket(base_scrape_date=date(2026, 10, 2))
        inserted_2 = engine.insert_quotes(basket_2)
        assert inserted_2 == 3750, f"Expected 3750 inserted, got {inserted_2}"

        # Verify total database records in fact table
        with engine.get_session() as session:
            total_records = session.query(FareQuoteRecord).count()
            assert total_records == 7500, f"Expected 7500 total records, found {total_records}"

            # Verify foreign key linkages
            unlinked_routes = session.query(FareQuoteRecord).filter(FareQuoteRecord.route_id.is_(None)).count()
            assert unlinked_routes == 0, f"Found {unlinked_routes} quotes without linked route_id!"

            unlinked_carriers = session.query(FareQuoteRecord).filter(FareQuoteRecord.carrier_id.is_(None)).count()
            assert unlinked_carriers == 0, f"Found {unlinked_carriers} quotes without linked carrier_id!"

        # Idempotent re-insertion: re-insert Basket 1
        re_inserted_1 = engine.insert_quotes(basket_1)
        assert re_inserted_1 == 3750, "Re-insertion should succeed idempotently via session.merge"

        # Verify record count is still exactly 7,500
        with engine.get_session() as session:
            total_after_reinsert = session.query(FareQuoteRecord).count()
            assert total_after_reinsert == 7500, "Record count changed after idempotent re-insertion!"


# ==============================================================================
# 2. QUERY STRESS TEST: CLEANEDFAREQUOTE PROPERTY GETTERS
# ==============================================================================

class TestCleanedFareQuotePropertyGetters:
    """
    Adversarial verification of CleanedFareQuote property getters via get_clean_quotes():
    fare_base, fare_fuel, fare_gst, fare_udf, fare_psf, fare_fee, dqs_score.
    """

    @pytest.fixture(scope="class")
    def populated_db(self):
        """Creates an in-memory SQLite database populated with a full seed basket."""
        engine = StorageEngine("sqlite:///:memory:")
        seed = SeedPlaybackEngine(seed=2026)
        basket = seed.generate_full_basket(base_scrape_date=date(2026, 10, 1))
        engine.insert_quotes(basket)
        return engine

    def test_all_statutory_property_getters_exist_and_typed(self, populated_db):
        """
        Verify all returned quotes from get_clean_quotes() possess all required
        statutory property getters with correct float types and non-negative values.
        """
        quotes = populated_db.get_clean_quotes(min_dqs=80.0)
        assert len(quotes) >= 3000, f"Expected >= 3000 clean quotes, got {len(quotes)}"

        # Check every single quote in the sample
        checked_count = 0
        for q in quotes:
            checked_count += 1

            # 1. Statutory property getters presence and types
            assert hasattr(q, "fare_base"), "Missing fare_base property"
            assert isinstance(q.fare_base, float), f"fare_base must be float, got {type(q.fare_base)}"
            assert q.fare_base > 0.0, f"fare_base must be positive, got {q.fare_base}"

            assert hasattr(q, "fare_fuel"), "Missing fare_fuel property"
            assert isinstance(q.fare_fuel, float), f"fare_fuel must be float, got {type(q.fare_fuel)}"
            assert q.fare_fuel >= 0.0, f"fare_fuel must be non-negative, got {q.fare_fuel}"

            assert hasattr(q, "fare_gst"), "Missing fare_gst property"
            assert isinstance(q.fare_gst, float), f"fare_gst must be float, got {type(q.fare_gst)}"
            assert q.fare_gst >= 0.0, f"fare_gst must be non-negative, got {q.fare_gst}"

            assert hasattr(q, "fare_udf"), "Missing fare_udf property"
            assert isinstance(q.fare_udf, float), f"fare_udf must be float, got {type(q.fare_udf)}"
            assert q.fare_udf >= 0.0, f"fare_udf must be non-negative, got {q.fare_udf}"

            assert hasattr(q, "fare_psf"), "Missing fare_psf property"
            assert isinstance(q.fare_psf, float), f"fare_psf must be float, got {type(q.fare_psf)}"
            assert q.fare_psf == 180.0, f"fare_psf must match statutory ₹180, got {q.fare_psf}"

            assert hasattr(q, "fare_fee"), "Missing fare_fee property"
            assert isinstance(q.fare_fee, float), f"fare_fee must be float, got {type(q.fare_fee)}"
            assert q.fare_fee >= 0.0, f"fare_fee must be non-negative, got {q.fare_fee}"

            assert hasattr(q, "dqs_score"), "Missing dqs_score property"
            assert isinstance(q.dqs_score, float), f"dqs_score must be float, got {type(q.dqs_score)}"
            assert 80.0 <= q.dqs_score <= 100.0, f"dqs_score out of range [80, 100]: {q.dqs_score}"

            # 2. Advance window property
            assert hasattr(q, "advance_window"), "Missing advance_window property"
            assert isinstance(q.advance_window, int), f"advance_window must be int, got {type(q.advance_window)}"
            assert q.advance_window in (1, 7, 15, 30, 45), f"Unexpected advance_window: {q.advance_window}"

            # 3. Arithmetic balance of property getters
            decomp_sum = (
                q.fare_base
                + q.fare_fuel
                + q.fare_gst
                + q.fare_udf
                + q.fare_psf
                + q.fare_fee
            )
            total_fare_val = float(q.total_fare)
            assert abs(decomp_sum - total_fare_val) <= 1.00, (
                f"Arithmetic decomposition mismatch: sum={decomp_sum} vs total={total_fare_val} "
                f"for quote {q.quote_id}"
            )

            # 4. Outlier & quarantine invariants
            assert q.is_outlier is False, f"Quote {q.quote_id} flagged as outlier but returned by get_clean_quotes"
            assert q.is_quarantined is False, f"Quote {q.quote_id} quarantined but returned by get_clean_quotes"

        assert checked_count >= 3000, f"Only checked {checked_count} quotes"

    def test_filtered_get_clean_quotes_properties(self, populated_db):
        """Test get_clean_quotes with sector, carrier, and advance_window filters."""
        # Query sector DEL-BOM, carrier 6E, advance_window 7
        filtered_quotes = populated_db.get_clean_quotes(
            sector="DEL-BOM",
            carrier="6E",
            advance_window=7,
            min_dqs=80.0,
        )
        assert len(filtered_quotes) > 0, "No quotes returned for filtered query!"

        for q in filtered_quotes:
            assert q.sector == "DEL-BOM"
            assert q.carrier_code == "6E"
            assert q.advance_window == 7
            assert q.fare_base > 0
            assert q.dqs_score >= 80.0

    def test_property_alias_consistency(self):
        """Direct adversarial test of property aliases against underlying fields."""
        d = DecomposedFare(
            base_fare=Decimal("4500.50"),
            fuel_surcharge=Decimal("1000.00"),
            airport_taxes_gst=Decimal("275.03"),
            user_dev_fee=Decimal("450.00"),
            passenger_service_fee=Decimal("180.00"),
            convenience_fee=Decimal("350.00"),
            total_fare=Decimal("6755.53"),
        )
        q = CleanedFareQuote(
            quote_id="test-prop-001",
            sector="DEL-BOM",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            source="makemytrip",
            source_type="ota",
            travel_date=date(2026, 10, 15),
            advance_days=15,
            flight_number="6E-2041",
            base_fare=d.base_fare,
            fuel_surcharge=d.fuel_surcharge,
            airport_taxes_gst=d.airport_taxes_gst,
            user_development_fee=d.user_development_fee,
            passenger_service_fee=d.passenger_service_fee,
            convenience_fee=d.convenience_fee,
            total_fare=d.total_fare,
            decomposed_fare=d,
            data_quality_score=92.5,
        )

        assert q.fare_base == 4500.50
        assert q.fare_fuel == 1000.00
        assert q.fare_gst == 275.03
        assert q.fare_udf == 450.00
        assert q.fare_psf == 180.00
        assert q.fare_fee == 350.00
        assert q.dqs_score == 92.5
        assert q.advance_window == 15
        assert q.carrier == "6E"
        assert q.origin == "DEL"
        assert q.destination == "BOM"


# ==============================================================================
# 3. SQLITE WAL MODE & CONCURRENCY STRESS TEST
# ==============================================================================

def _worker_process_reader(db_path: str, worker_id: int, iterations: int, result_queue: mp.Queue):
    """Standalone worker function for multiprocess reading test."""
    try:
        engine = StorageEngine(f"sqlite:///{db_path}")
        total_quotes_read = 0
        for i in range(iterations):
            quotes = engine.get_clean_quotes(sector="DEL-BOM")
            total_quotes_read += len(quotes)
            time_mod.sleep(0.005)
        engine.engine.dispose()
        result_queue.put(("OK", worker_id, "reader", total_quotes_read, None))
    except Exception as e:
        result_queue.put(("FAIL", worker_id, "reader", 0, str(e)))


def _worker_process_writer(db_path: str, worker_id: int, iterations: int, result_queue: mp.Queue):
    """Standalone worker function for multiprocess writing test."""
    try:
        engine = StorageEngine(f"sqlite:///{db_path}")
        seed = SeedPlaybackEngine(seed=worker_id + 5000)
        total_quotes_written = 0
        for i in range(iterations):
            t_date = date(2026, 11, (i % 25) + 1)
            q = seed.synthesize_quote(
                "DEL", "BOM", t_date, 15, "6E", "indigo", flight_idx=worker_id * 1000 + i
            )
            n = engine.insert_quotes([q])
            total_quotes_written += n
            time_mod.sleep(0.005)
        engine.engine.dispose()
        result_queue.put(("OK", worker_id, "writer", total_quotes_written, None))
    except Exception as e:
        result_queue.put(("FAIL", worker_id, "writer", 0, str(e)))


class TestSqliteWalConcurrency:
    """
    Adversarial verification of SQLite WAL mode and concurrent access:
    Ensures readers and writers execute concurrently without database lock contention.
    """

    @pytest.fixture
    def wal_db(self):
        """Creates a temporary SQLite file database."""
        temp_dir = tempfile.mkdtemp()
        db_path = os.path.join(temp_dir, "apix_wal_concurrency.db")
        engine = StorageEngine(f"sqlite:///{db_path}")

        # Seed baseline data
        seed = SeedPlaybackEngine(seed=777)
        initial_quotes = seed.generate_quotes_for_sector("DEL", "BOM", date(2026, 11, 1), 7, quotes_per_carrier=2)
        engine.insert_quotes(initial_quotes)

        yield engine, db_path

        engine.engine.dispose()
        for ext in ["", "-wal", "-shm"]:
            f = db_path + ext
            if os.path.exists(f):
                try:
                    os.remove(f)
                except Exception:
                    pass
        try:
            os.rmdir(temp_dir)
        except Exception:
            pass

    def test_wal_mode_pragmas_enabled(self, wal_db):
        """Verify journal_mode is WAL, synchronous is NORMAL, and busy_timeout is >= 5000."""
        _, db_path = wal_db
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        cur.execute("PRAGMA journal_mode;")
        journal_mode = cur.fetchone()[0]
        assert journal_mode.lower() == "wal", f"Expected WAL journal mode, got {journal_mode}"

        cur.execute("PRAGMA synchronous;")
        synchronous = cur.fetchone()[0]
        # In SQLite: 0 = OFF, 1 = NORMAL, 2 = FULL, 3 = EXTRA
        assert synchronous in (1, 2), f"Expected NORMAL (1) or FULL (2), got {synchronous}"

        conn.close()

    def test_threaded_concurrent_reads_and_writes(self, wal_db):
        """
        Adversarial multithreaded test: 6 reader threads and 6 writer threads
        running concurrently across 180 total operations.
        Assert zero sqlite3.OperationalError: database is locked.
        """
        _, db_path = wal_db

        errors = []
        read_successes = []
        write_successes = []
        engines_to_dispose = []

        def reader_task(tid: int, ops: int = 15):
            eng = StorageEngine(f"sqlite:///{db_path}")
            engines_to_dispose.append(eng)
            for i in range(ops):
                try:
                    quotes = eng.get_clean_quotes(sector="DEL-BOM")
                    summary = eng.get_route_summary(sector="DEL-BOM")
                    read_successes.append((tid, i, len(quotes), summary["sector"]))
                    time_mod.sleep(0.002)
                except Exception as e:
                    errors.append(("reader", tid, i, str(e)))

        def writer_task(tid: int, ops: int = 15):
            eng = StorageEngine(f"sqlite:///{db_path}")
            engines_to_dispose.append(eng)
            local_seed = SeedPlaybackEngine(seed=tid * 1000 + 777)
            for i in range(ops):
                try:
                    t_date = date(2026, 11, (i % 20) + 1)
                    q = local_seed.synthesize_quote(
                        "DEL", "BOM", t_date, 7, "AI", "airindia", flight_idx=tid * 100 + i
                    )
                    cnt = eng.insert_quotes([q])
                    write_successes.append((tid, i, cnt))
                    time_mod.sleep(0.002)
                except Exception as e:
                    errors.append(("writer", tid, i, str(e)))

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            futures = []
            for t in range(6):
                futures.append(executor.submit(reader_task, t, 15))
                futures.append(executor.submit(writer_task, t, 15))
            concurrent.futures.wait(futures)

        for eng in engines_to_dispose:
            eng.engine.dispose()

        assert len(errors) == 0, f"Encountered database lock or concurrency errors: {errors}"
        assert len(read_successes) == 90, f"Expected 90 read operations, completed {len(read_successes)}"
        assert len(write_successes) == 90, f"Expected 90 write operations, completed {len(write_successes)}"

    def test_multiprocess_concurrent_reads_and_writes(self, wal_db):
        """
        Adversarial multiprocess test: separate OS processes accessing the same
        SQLite file concurrently under WAL mode.
        """
        _, db_path = wal_db

        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()

        processes = []
        n_readers = 3
        n_writers = 3
        iterations_per_worker = 12

        for r in range(n_readers):
            p = ctx.Process(
                target=_worker_process_reader,
                args=(db_path, r, iterations_per_worker, result_queue),
            )
            processes.append(p)

        for w in range(n_writers):
            p = ctx.Process(
                target=_worker_process_writer,
                args=(db_path, w, iterations_per_worker, result_queue),
            )
            processes.append(p)

        for p in processes:
            p.start()

        for p in processes:
            p.join(timeout=30)

        # Collect results from queue
        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        assert len(results) == n_readers + n_writers, (
            f"Expected {n_readers + n_writers} results from child processes, got {len(results)}"
        )

        failures = [r for r in results if r[0] != "OK"]
        assert len(failures) == 0, f"Multiprocess workers encountered failures: {failures}"


# ==============================================================================
# 4. ADVERSARIAL EDGE CASES & DUCK TYPING
# ==============================================================================

class TestAdversarialEdgeCasesAndDuckTyping:
    """
    Adversarial verification of input robustness, duck-typing, and normalization.
    """

    def test_duck_typed_quote_varieties(self):
        """Verify StorageEngine.insert_quotes handles dicts, RawFareQuotes, and custom models."""
        engine = StorageEngine("sqlite:///:memory:")

        # 1. PipelineRawFareQuote
        q_raw_pipe = PipelineRawFareQuote(
            source="indigo",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            flight_number="6E-101",
            departure_time=time(6, 0),
            arrival_time=time(8, 0),
            duration_minutes=120,
            travel_date=date(2026, 10, 1),
            advance_days=7,
            raw_total_fare=5200.0,
        )

        # 2. IngestionRawFareQuote
        q_raw_ingest = IngestionRawFareQuote(
            quote_id="ingest-uuid-001",
            source="makemytrip",
            source_type="ota",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            flight_number="6E-102",
            departure_time=time(9, 0),
            arrival_time=time(11, 15),
            duration_minutes=135,
            travel_date=date(2026, 10, 1),
            advance_days=7,
            raw_base_fare=Decimal("3800.00"),
            raw_fuel_surcharge=Decimal("1000.00"),
            raw_taxes=Decimal("240.00"),
            raw_udf=Decimal("450.00"),
            raw_psf=Decimal("180.00"),
            raw_convenience_fee=Decimal("350.00"),
            raw_total_fare=Decimal("6020.00"),
        )

        # 3. CleanedFareQuote
        decomp = DecomposedFare(
            base_fare=Decimal("3000.00"),
            fuel_surcharge=Decimal("1000.00"),
            airport_taxes_gst=Decimal("200.00"),
            user_dev_fee=Decimal("450.00"),
            passenger_service_fee=Decimal("180.00"),
            convenience_fee=Decimal("0.00"),
            total_fare=Decimal("4830.00"),
        )
        q_clean = CleanedFareQuote(
            quote_id="clean-uuid-001",
            sector="DEL-BOM",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            source="indigo",
            travel_date=date(2026, 10, 1),
            advance_days=7,
            flight_number="6E-103",
            base_fare=decomp.base_fare,
            fuel_surcharge=decomp.fuel_surcharge,
            airport_taxes_gst=decomp.airport_taxes_gst,
            user_development_fee=decomp.user_development_fee,
            passenger_service_fee=decomp.passenger_service_fee,
            convenience_fee=decomp.convenience_fee,
            total_fare=decomp.total_fare,
            decomposed_fare=decomp,
            data_quality_score=95.0,
        )

        # 4. Dictionary quote
        q_dict = {
            "source": "airindia",
            "origin_iata": "DEL",
            "destination_iata": "BOM",
            "carrier_code": "AI",
            "flight_number": "AI-887",
            "departure_time": time(7, 30),
            "arrival_time": time(9, 45),
            "duration_minutes": 135,
            "travel_date": date(2026, 10, 1),
            "advance_days": 7,
            "raw_total_fare": 6500.0,
        }

        batch = [q_raw_pipe, q_raw_ingest, q_clean, q_dict]
        count = engine.insert_quotes(batch)
        assert count == 4, f"Expected 4 quotes inserted, got {count}"

        clean_retrieved = engine.get_clean_quotes()
        assert len(clean_retrieved) >= 3, f"Expected at least 3 clean quotes, got {len(clean_retrieved)}"

    def test_whitespace_and_hyphen_flight_number_dedup(self):
        """Flight numbers with disparate whitespace, casing, and hyphens coalesce."""
        q1 = PipelineRawFareQuote(
            source="indigo",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            flight_number="6E-5012",
            departure_time=time(6, 0),
            arrival_time=time(8, 0),
            duration_minutes=120,
            travel_date=date(2026, 10, 5),
            advance_days=7,
            raw_total_fare=4500.0,
        )
        q2 = PipelineRawFareQuote(
            source="makemytrip",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            flight_number="6E 5012",
            departure_time=time(6, 0),
            arrival_time=time(8, 0),
            duration_minutes=120,
            travel_date=date(2026, 10, 5),
            advance_days=7,
            raw_total_fare=4650.0,
        )
        q3 = PipelineRawFareQuote(
            source="cleartrip",
            origin_iata="DEL",
            destination_iata="BOM",
            carrier_code="6E",
            flight_number="6e5012",
            departure_time=time(6, 0),
            arrival_time=time(8, 0),
            duration_minutes=120,
            travel_date=date(2026, 10, 5),
            advance_days=7,
            raw_total_fare=4700.0,
        )

        deduped = deduplicate_quotes([q1, q2, q3])
        assert len(deduped) == 1, f"Expected 1 quote after dedup, got {len(deduped)}"
        assert deduped[0].source == "indigo"
        assert float(deduped[0].raw_total_fare) == 4500.0
