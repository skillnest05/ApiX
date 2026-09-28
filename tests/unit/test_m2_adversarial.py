"""
Robustness Test Suite for Fare Decomposition & Concurrency.

Tests:
1. Duplicate quote resolution across carrier and multiple OTAs (prioritization and coalescing).
2. SQLite concurrency under rapid batch inserts and transaction rollbacks on corrupt batches.
3. Decomposition arithmetic identity under arbitrary float amounts: |Total - sum(Components)| <= 1.00 INR.
"""

from collections import defaultdict
import concurrent.futures
from datetime import date, datetime, time, timedelta
from decimal import Decimal
import itertools
import os
import random
import tempfile
from typing import List, Tuple

import pytest

from apix.pipeline.decomposition import (
    AERA_UDF_TARIFFS,
    CONVENIENCE_FEES,
    DEFAULT_FALLBACK_UDF,
    STATUTORY_PSF,
    decompose_fare,
    decompose_fare_dict,
    decompose_quote,
    verify_decomposition_identity,
)
from apix.pipeline.quality import (
    DIRECT_AIRLINE_SOURCES,
    canonical_flight_key,
    deduplicate_quotes,
)
from apix.pipeline.schemas import (
    CleanedFareQuote,
    DecomposedFare,
    RawFareQuote,
)
from apix.pipeline.storage import (
    Carrier,
    FareQuoteRecord,
    Route,
    StorageEngine,
    get_clean_quotes,
    insert_quotes,
)


# ==============================================================================
# 1. DUPLICATE QUOTE RESOLUTION ACROSS CARRIER AND MULTIPLE OTAS
# ==============================================================================

class TestDuplicateQuoteResolution:
    """
    Adversarial stress-testing of canonical deduplication:
    - Identical flight number, date, sector from 1 carrier and 3 OTAs.
    - Direct carrier quote prioritized over OTA quotes even when OTAs quote lower fares.
    - Coalescence of duplicate quotes into a single canonical quote.
    - Permutation order invariance across input quote sequences.
    """

    @pytest.fixture
    def base_flight_params(self):
        return {
            "origin": "DEL",
            "destination": "BOM",
            "carrier": "6E",
            "flight_number": "6E-2041",
            "departure_time": time(6, 0),
            "arrival_time": time(8, 15),
            "travel_date": date(2026, 10, 5),
            "advance_days": 7,
            "duration_minutes": 135,
        }

    def test_carrier_prioritized_over_three_otas(self, base_flight_params):
        """
        Verify that when 1 carrier and 3 OTAs provide quotes for the identical flight,
        the carrier quote is selected even when OTAs advertise cheaper prices (e.g. promotions).
        Asserts that the 4 quotes coalesce into exactly 1 quote.
        """
        # Carrier fare is higher (₹5,500) than OTAs (₹4,950 to ₹5,300)
        q_carrier = RawFareQuote(
            source="indigo",
            total_fare=5500.0,
            **base_flight_params,
        )
        q_mmt = RawFareQuote(
            source="makemytrip",
            total_fare=5100.0,
            **base_flight_params,
        )
        q_ct = RawFareQuote(
            source="cleartrip",
            total_fare=5250.0,
            **base_flight_params,
        )
        q_emt = RawFareQuote(
            source="easemytrip",
            total_fare=4950.0,
            **base_flight_params,
        )

        input_batch = [q_mmt, q_ct, q_carrier, q_emt]
        deduped = deduplicate_quotes(input_batch)

        assert len(deduped) == 1, f"Expected 1 coalesced quote, found {len(deduped)}"
        selected = deduped[0]
        assert selected.source == "indigo", f"Expected carrier 'indigo', got '{selected.source}'"
        assert selected.carrier_code == "6E"
        assert selected.total_fare == 5500.0

    def test_ota_fallback_lowest_fare_when_no_carrier(self, base_flight_params):
        """
        Verify that when no direct carrier quote is present, deduplication chooses
        the lowest fare among the 3 OTAs.
        """
        q_mmt = RawFareQuote(source="makemytrip", total_fare=5100.0, **base_flight_params)
        q_ct = RawFareQuote(source="cleartrip", total_fare=5250.0, **base_flight_params)
        q_emt = RawFareQuote(source="easemytrip", total_fare=4950.0, **base_flight_params)

        input_batch = [q_mmt, q_ct, q_emt]
        deduped = deduplicate_quotes(input_batch)

        assert len(deduped) == 1
        selected = deduped[0]
        assert selected.source == "easemytrip", f"Expected lowest OTA 'easemytrip', got '{selected.source}'"
        assert selected.total_fare == 4950.0

    def test_order_invariance_carrier_resolution(self, base_flight_params):
        """
        Adversarial test: verify that quote arrival order does not affect resolution.
        Tests all 4! = 24 permutations of [carrier, mmt, ct, emt].
        In all 24 permutations, the carrier quote MUST be selected.
        """
        q_carrier = RawFareQuote(source="indigo", total_fare=5600.0, **base_flight_params)
        q_mmt = RawFareQuote(source="makemytrip", total_fare=5100.0, **base_flight_params)
        q_ct = RawFareQuote(source="cleartrip", total_fare=5200.0, **base_flight_params)
        q_emt = RawFareQuote(source="easemytrip", total_fare=4900.0, **base_flight_params)

        items = [q_carrier, q_mmt, q_ct, q_emt]
        for idx, perm in enumerate(itertools.permutations(items)):
            deduped = deduplicate_quotes(list(perm))
            assert len(deduped) == 1, f"Permutation {idx} failed: got {len(deduped)} quotes"
            assert deduped[0].source == "indigo", (
                f"Permutation {idx} selected '{deduped[0].source}' instead of carrier 'indigo'"
            )

    def test_multi_flight_coalescence_across_carriers_and_sectors(self):
        """
        Stress test: 10 distinct flight instances across 5 carriers and 10 sectors.
        Each flight is quoted by 1 direct carrier and 3 OTAs (40 total quotes).
        Asserts that the 40 quotes coalesce into exactly 10 quotes, all from carriers.
        """
        carriers_and_sectors = [
            ("6E", "indigo", "DEL", "BOM", "6E-2041"),
            ("AI", "airindia", "BOM", "DEL", "AI-805"),
            ("QP", "akasa", "DEL", "BLR", "QP-1102"),
            ("SG", "spicejet", "BLR", "DEL", "SG-8169"),
            ("IX", "airindiaexpress", "BOM", "BLR", "IX-1142"),
            ("6E", "indigo", "DEL", "HYD", "6E-5012"),
            ("AI", "airindia", "HYD", "DEL", "AI-543"),
            ("QP", "akasa", "DEL", "CCU", "QP-1304"),
            ("SG", "spicejet", "CCU", "DEL", "SG-278"),
            ("6E", "indigo", "BOM", "GOI", "6E-442"),
        ]

        quotes: List[RawFareQuote] = []
        for c_code, c_source, orig, dest, f_num in carriers_and_sectors:
            base_kw = {
                "origin": orig,
                "destination": dest,
                "carrier": c_code,
                "flight_number": f_num,
                "departure_time": time(7, 0),
                "arrival_time": time(9, 15),
                "travel_date": date(2026, 10, 5),
                "advance_days": 7,
                "duration_minutes": 135,
            }
            # 1 Carrier quote
            quotes.append(RawFareQuote(source=c_source, total_fare=5500.0, **base_kw))
            # 3 OTA quotes with lower promotional fares
            quotes.append(RawFareQuote(source="makemytrip", total_fare=5100.0, **base_kw))
            quotes.append(RawFareQuote(source="cleartrip", total_fare=5200.0, **base_kw))
            quotes.append(RawFareQuote(source="easemytrip", total_fare=4900.0, **base_kw))

        assert len(quotes) == 40
        random.seed(999)
        random.shuffle(quotes)

        deduped = deduplicate_quotes(quotes)
        assert len(deduped) == 10, f"Expected 10 coalesced quotes, got {len(deduped)}"

        expected_sources = {item[1] for item in carriers_and_sectors}
        actual_sources = {q.source for q in deduped}
        assert actual_sources == expected_sources, f"Mismatch in carrier sources: {actual_sources} != {expected_sources}"

    def test_date_and_sector_differentiation(self):
        """
        Verify that quotes for the same flight number on DIFFERENT dates or DIFFERENT sectors
        are NOT coalesced together (each forms a distinct canonical flight instance).
        """
        base = {
            "origin": "DEL",
            "destination": "BOM",
            "carrier": "6E",
            "flight_number": "6E-2041",
            "departure_time": time(6, 0),
            "arrival_time": time(8, 15),
            "advance_days": 7,
            "duration_minutes": 135,
            "total_fare": 5000.0,
        }
        # Same flight, two different dates (1 carrier + 3 OTAs each = 8 quotes)
        quotes: List[RawFareQuote] = []
        for d in [date(2026, 10, 5), date(2026, 10, 12)]:
            quotes.append(RawFareQuote(source="indigo", travel_date=d, **base))
            quotes.append(RawFareQuote(source="makemytrip", travel_date=d, **base))
            quotes.append(RawFareQuote(source="cleartrip", travel_date=d, **base))
            quotes.append(RawFareQuote(source="easemytrip", travel_date=d, **base))

        deduped = deduplicate_quotes(quotes)
        assert len(deduped) == 2, f"Expected 2 quotes for 2 travel dates, got {len(deduped)}"
        dates_in_output = {q.travel_date for q in deduped}
        assert dates_in_output == {date(2026, 10, 5), date(2026, 10, 12)}

    def test_source_casing_and_whitespace_robustness(self, base_flight_params):
        """
        Verify that carrier source matching handles casing and whitespace:
        e.g. 'INDIGO', '  indigo  ', '6E', 'AirIndia'.
        """
        variations = ["INDIGO", "  indigo  ", "6E", " 6e "]
        for var in variations:
            q_carrier = RawFareQuote(source=var, total_fare=5800.0, **base_flight_params)
            q_ota = RawFareQuote(source="makemytrip", total_fare=5000.0, **base_flight_params)
            deduped = deduplicate_quotes([q_ota, q_carrier])
            assert len(deduped) == 1
            assert deduped[0].source == var, f"Failed for variation '{var}'"


# ==============================================================================
# 2. SQLITE CONCURRENCY UNDER RAPID BATCH INSERTS & ROLLBACKS
# ==============================================================================

class TestSqliteConcurrencyAndRollbacks:
    """
    Empirical stress-testing of SQLite storage engine:
    - Multi-threaded concurrent batch inserts.
    - Concurrent read and write access without locking errors.
    - Transaction atomicity and rollback on corrupt batches.
    - Sequential batch isolation (corrupt batch failure does not corrupt prior/subsequent data).
    """

    @pytest.fixture
    def temp_sqlite_engine(self):
        """Creates a temporary SQLite database for testing, disposing connection on cleanup."""
        tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        tmp.close()
        db_path = tmp.name
        engine = StorageEngine(f"sqlite:///{db_path}")
        yield engine
        engine.engine.dispose()
        if os.path.exists(db_path):
            try:
                os.remove(db_path)
            except OSError:
                pass

    def test_rapid_concurrent_batch_inserts(self, temp_sqlite_engine):
        """
        Stress test: 10 worker threads concurrently inserting 20 quotes each (200 quotes total)
        into a SQLite database file.
        Asserts that all 200 quotes are stored without operational lock failures or deadlocks.
        """
        def insert_worker(tid: int) -> int:
            batch = []
            for i in range(20):
                q = RawFareQuote(
                    source="indigo",
                    origin="DEL",
                    destination="BOM",
                    carrier="6E",
                    flight_number=f"6E-{tid:02d}{i:03d}",
                    departure_time=time(6, 0),
                    arrival_time=time(8, 15),
                    travel_date=date(2026, 10, 5),
                    advance_days=7,
                    duration_minutes=135,
                    total_fare=5000.0 + tid * 10 + i,
                )
                batch.append(q)
            return temp_sqlite_engine.insert_quotes(batch)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(insert_worker, tid) for tid in range(10)]
            results = [f.result() for f in futures]

        assert sum(results) == 200, f"Expected 200 inserted quotes, got {sum(results)}"
        clean_quotes = temp_sqlite_engine.get_clean_quotes()
        assert len(clean_quotes) == 200, f"Expected 200 quotes in DB, found {len(clean_quotes)}"

    def test_concurrent_read_write_stress(self, temp_sqlite_engine):
        """
        Stress test: 8 writer threads inserting batches simultaneously with 4 reader threads
        querying get_clean_quotes and get_route_summary.
        Asserts zero SQLite lock exceptions and consistent eventual count.
        """
        def writer_task(wid: int) -> int:
            batch = [
                RawFareQuote(
                    source="airindia",
                    origin="BOM",
                    destination="BLR",
                    carrier="AI",
                    flight_number=f"AI-{wid:02d}{i:03d}",
                    departure_time=time(8, 0),
                    arrival_time=time(9, 45),
                    travel_date=date(2026, 10, 5),
                    advance_days=7,
                    duration_minutes=105,
                    total_fare=4800.0 + wid * 5 + i,
                )
                for i in range(15)
            ]
            return temp_sqlite_engine.insert_quotes(batch)

        def reader_task(rid: int) -> Tuple[int, str]:
            quotes = temp_sqlite_engine.get_clean_quotes(sector="BOM-BLR")
            summary = temp_sqlite_engine.get_route_summary("BOM-BLR")
            return len(quotes), summary["sector"]

        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as executor:
            writer_futures = [executor.submit(writer_task, i) for i in range(8)]
            reader_futures = [executor.submit(reader_task, j) for j in range(4)]

            writer_results = [f.result() for f in writer_futures]
            reader_results = [f.result() for f in reader_futures]

        assert sum(writer_results) == 120
        final_quotes = temp_sqlite_engine.get_clean_quotes(sector="BOM-BLR")
        assert len(final_quotes) == 120

    def test_transaction_rollback_on_corrupt_batch(self, temp_sqlite_engine):
        """
        Verify transaction atomicity and rollback:
        When a batch containing valid quotes is corrupted by an invalid quote
        (e.g. identical origin and destination 'DEL-DEL'),
        the entire batch must fail and 0 quotes must be committed.
        """
        valid_q1 = {
            "source": "indigo", "origin": "DEL", "destination": "BOM", "carrier": "6E",
            "flight_number": "6E-101", "departure_time": time(6, 0), "arrival_time": time(8, 15),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 135,
            "total_fare": 5000.0,
        }
        valid_q2 = {
            "source": "indigo", "origin": "DEL", "destination": "BOM", "carrier": "6E",
            "flight_number": "6E-102", "departure_time": time(7, 0), "arrival_time": time(9, 15),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 135,
            "total_fare": 5200.0,
        }
        corrupt_q = {
            "source": "indigo", "origin": "DEL", "destination": "DEL", # Corrupt!
            "carrier": "6E", "flight_number": "6E-999", "departure_time": time(8, 0),
            "arrival_time": time(10, 0), "travel_date": date(2026, 10, 5), "advance_days": 7,
            "duration_minutes": 120, "total_fare": 5000.0,
        }

        batch = [valid_q1, valid_q2, corrupt_q]

        with pytest.raises(Exception):
            temp_sqlite_engine.insert_quotes(batch)

        # Database must remain empty
        quotes_in_db = temp_sqlite_engine.get_clean_quotes()
        assert len(quotes_in_db) == 0, f"Expected 0 quotes in DB after rollback, found {len(quotes_in_db)}"

    def test_sequential_batch_rollback_isolation(self, temp_sqlite_engine):
        """
        Verify that a failed corrupt batch does not contaminate existing records
        or prevent subsequent valid batches from committing:
        Batch 1 (2 valid quotes) -> commits (DB has 2)
        Batch 2 (1 valid, 1 corrupt) -> rolls back (DB still has 2)
        Batch 3 (2 valid quotes) -> commits (DB has 4)
        """
        # Batch 1: 2 valid quotes
        b1_q1 = {
            "source": "indigo", "origin": "DEL", "destination": "BOM", "carrier": "6E",
            "flight_number": "6E-101", "departure_time": time(6, 0), "arrival_time": time(8, 15),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 135,
            "total_fare": 5000.0,
        }
        b1_q2 = {
            "source": "indigo", "origin": "DEL", "destination": "BOM", "carrier": "6E",
            "flight_number": "6E-102", "departure_time": time(7, 0), "arrival_time": time(9, 15),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 135,
            "total_fare": 5200.0,
        }
        res1 = temp_sqlite_engine.insert_quotes([b1_q1, b1_q2])
        assert res1 == 2
        assert len(temp_sqlite_engine.get_clean_quotes()) == 2

        # Batch 2: corrupt batch
        b2_q1 = {
            "source": "indigo", "origin": "DEL", "destination": "BOM", "carrier": "6E",
            "flight_number": "6E-103", "departure_time": time(8, 0), "arrival_time": time(10, 15),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 135,
            "total_fare": 5400.0,
        }
        b2_corrupt = {
            "source": "indigo", "origin": "DEL", "destination": "DEL",
            "carrier": "6E", "flight_number": "6E-999", "departure_time": time(9, 0),
            "arrival_time": time(11, 0), "travel_date": date(2026, 10, 5), "advance_days": 7,
            "duration_minutes": 120, "total_fare": 5000.0,
        }
        with pytest.raises(Exception):
            temp_sqlite_engine.insert_quotes([b2_q1, b2_corrupt])

        # DB must still have exactly the 2 quotes from Batch 1
        assert len(temp_sqlite_engine.get_clean_quotes()) == 2

        # Batch 3: 2 valid quotes
        b3_q1 = {
            "source": "airindia", "origin": "DEL", "destination": "BLR", "carrier": "AI",
            "flight_number": "AI-201", "departure_time": time(9, 0), "arrival_time": time(11, 45),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 165,
            "total_fare": 6100.0,
        }
        b3_q2 = {
            "source": "akasa", "origin": "BOM", "destination": "BLR", "carrier": "QP",
            "flight_number": "QP-301", "departure_time": time(10, 0), "arrival_time": time(11, 45),
            "travel_date": date(2026, 10, 5), "advance_days": 7, "duration_minutes": 105,
            "total_fare": 4300.0,
        }
        res3 = temp_sqlite_engine.insert_quotes([b3_q1, b3_q2])
        assert res3 == 2
        assert len(temp_sqlite_engine.get_clean_quotes()) == 4


# ==============================================================================
# 3. DECOMPOSITION ARITHMETIC IDENTITY UNDER ARBITRARY FLOAT AMOUNTS
# ==============================================================================

class TestDecompositionArithmeticIdentity:
    """
    Empirical stress-testing of 6-component statutory fare decomposition:
    Total = Base Fare + Fuel Surcharge + GST (5%) + UDF + PSF + Convenience Fee
    Asserts:
    1. |Total - sum(Components)| <= 1.00 INR for arbitrary floating point fares.
    2. Precision stress (e.g. ₹4,321.49, ₹12,789.93, multi-decimal values).
    3. Combinatorial testing across all 13 AERA airport hubs and OTAs.
    4. Exact balance validation via verify_decomposition_identity.
    """

    ALL_AIRPORTS = [
        "DEL", "BOM", "BLR", "HYD", "CCU", "PNQ", "AMD",
        "MAA", "JAI", "GOI", "LKO", "SXR", "GAU", "IXC",
    ]
    ALL_SOURCES = ["indigo", "airindia", "makemytrip", "cleartrip", "easemytrip", "ixigo"]

    @pytest.mark.parametrize("amount", [4321.49, 12789.93])
    def test_mandated_benchmark_floats(self, amount):
        """
        Verify the two specific benchmark amounts mandated in DISPATCH.md:
        ₹4,321.49 and ₹12,789.93 across all 13 AERA hubs and multiple sources.
        Asserts: |Total - sum(Components)| <= 1.00 INR.
        """
        for airport in self.ALL_AIRPORTS:
            for src in self.ALL_SOURCES:
                d = decompose_fare(amount, origin_iata=airport, source=src)
                diff = float(d.reconciliation_difference)
                assert diff <= 1.00, (
                    f"Breached for amount {amount} at airport {airport} from {src}: diff={diff}"
                )
                assert d.is_balanced(tolerance=1.00)
                # Verify exact components sum to total
                assert d.reconstructed_total == d.total_fare

    @pytest.mark.parametrize("amount", [
        999.00, 999.49, 1000.00, 24999.99, 25000.00,
        5432.11, 8888.88, 15674.33, 21999.50,
    ])
    def test_boundary_and_intermediate_floats(self, amount):
        """
        Verify domain boundaries (floor ₹999.00, ceiling ₹25,000.00) and intermediate amounts.
        Asserts |Total - sum(Components)| <= 1.00 INR.
        """
        for airport in self.ALL_AIRPORTS:
            d = decompose_fare(amount, origin_iata=airport, source="indigo")
            diff = float(d.reconciliation_difference)
            assert diff <= 1.00
            assert d.is_balanced(tolerance=1.00)

    @pytest.mark.parametrize("stress_float", [
        4321.489999999999,
        12789.930000000001,
        4321.494999999,
        12789.925000001,
        999.00000001,
        999.4999999,
        24999.999999,
        25000.000000,
        1500.123456789,
        8765.987654321,
    ])
    def test_extended_precision_floats(self, stress_float):
        """
        Adversarial test: Floating point numbers with extended decimal digits.
        Verifies that precision drift does not cause reconciliation residual > 1.00 INR.
        """
        for airport in ["DEL", "BOM", "BLR", "SXR"]:
            d = decompose_fare(stress_float, origin_iata=airport, source="makemytrip")
            diff = float(d.reconciliation_difference)
            assert diff <= 1.00, f"Precision failure on {stress_float}: diff={diff}"
            assert d.is_balanced(tolerance=1.00)

    def test_large_scale_random_float_sweep(self):
        """
        Comprehensive stress test: 5,000 arbitrary float amounts randomly sampled
        from the continuous uniform distribution [999.0, 25000.0].
        Tested against random combinations of 14 airports and 6 sources.
        Asserts 100% adherence to |Total - sum(Components)| <= 1.00 INR.
        """
        random.seed(4242)
        sample_size = 5000
        max_observed_diff = 0.0

        for _ in range(sample_size):
            amt = random.uniform(999.0, 25000.0)
            apt = random.choice(self.ALL_AIRPORTS)
            src = random.choice(self.ALL_SOURCES)

            d = decompose_fare(amt, origin_iata=apt, source=src)
            diff = float(d.reconciliation_difference)
            if diff > max_observed_diff:
                max_observed_diff = diff

            assert diff <= 1.00, f"Decomposition failure on {amt}: diff={diff}"

        # In fact, rounding absorption guarantees exact zero residual
        assert max_observed_diff <= 0.01, f"Max difference unexpectedly high: {max_observed_diff}"

    def test_decomposed_fare_model_tolerance_boundary(self):
        """
        Unit validation of DecomposedFare.is_balanced(tolerance=1.00)
        at exact tolerance boundaries.
        """
        # Exactly 0.50 difference -> balanced
        d_balanced = DecomposedFare(
            base_fare=Decimal("4000.00"),
            fuel_surcharge=Decimal("1000.00"),
            airport_taxes_gst=Decimal("250.00"),
            user_dev_fee=Decimal("450.00"),
            passenger_service_fee=Decimal("180.00"),
            convenience_fee=Decimal("0.00"),
            total_fare=Decimal("5880.50"), # 5880.00 + 0.50
        )
        assert d_balanced.is_balanced(tolerance=1.00) is True
        assert verify_decomposition_identity(d_balanced, tolerance=1.00) is True

        # Exactly 1.00 difference -> balanced
        d_edge = DecomposedFare(
            base_fare=Decimal("4000.00"),
            fuel_surcharge=Decimal("1000.00"),
            airport_taxes_gst=Decimal("250.00"),
            user_dev_fee=Decimal("450.00"),
            passenger_service_fee=Decimal("180.00"),
            convenience_fee=Decimal("0.00"),
            total_fare=Decimal("5881.00"), # 5880.00 + 1.00
        )
        assert d_edge.is_balanced(tolerance=1.00) is True

        # 1.01 difference -> unbalanced
        d_unbalanced = DecomposedFare(
            base_fare=Decimal("4000.00"),
            fuel_surcharge=Decimal("1000.00"),
            airport_taxes_gst=Decimal("250.00"),
            user_dev_fee=Decimal("450.00"),
            passenger_service_fee=Decimal("180.00"),
            convenience_fee=Decimal("0.00"),
            total_fare=Decimal("5881.01"), # 5880.00 + 1.01
        )
        assert d_unbalanced.is_balanced(tolerance=1.00) is False
        assert verify_decomposition_identity(d_unbalanced, tolerance=1.00) is False
