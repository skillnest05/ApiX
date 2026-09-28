"""
Unit tests for official MoSPI eSankhyiki Ingestion Client & SQLite Storage (Problem Statement 26056).
"""

import json
import os
import pytest

from apix.ingestion.esankhyiki import ESankhyikiClient, ESankhyikiRecord
from apix.pipeline.storage import StorageEngine, MospiCpiRecord


class TestESankhyikiIngestion:
    """Tests authentic MoSPI eSankhyiki CPI data ingestion and processing."""

    def test_record_parsing_from_dict(self):
        sample = {
            "base_year": "2024",
            "series": "Current",
            "year": "2026",
            "month": "August",
            "state": "All India",
            "sector": "Combined",
            "division": "Transport",
            "group": "Passenger transport services",
            "class": "Passenger transport by air",
            "sub_class": "Passenger transport by air, domestic",
            "item": "Airfare",
            "code": "07.3.3.1.2.01",
            "index": "135.49",
            "inflation": "20.85",
            "imputation": "N",
        }
        rec = ESankhyikiRecord.from_api_dict(sample)
        assert rec.base_year == "2024"
        assert rec.code == "07.3.3.1.2.01"
        assert rec.index == 135.49
        assert rec.inflation == 20.85
        assert rec.item == "Airfare"

    def test_client_loads_verified_cache(self, tmp_path):
        client = ESankhyikiClient()
        data = client.load_cached_data()
        assert "metadata" in data
        assert data["metadata"]["item_code"] == 294
        assert "current_base_2024" in data
        airfare = data["current_base_2024"]["airfare_domestic"]
        assert len(airfare) > 0

    def test_national_airfare_benchmark(self):
        client = ESankhyikiClient()
        bench = client.get_national_airfare_benchmark()
        assert bench["code"] == "07.3.3.1.2.01"
        assert bench["index"] > 100.0
        assert bench["inflation_pct"] > 0

    def test_national_transport_benchmark(self):
        client = ESankhyikiClient()
        trans = client.get_national_transport_benchmark()
        assert trans["code"] == "07"
        assert trans["index"] > 100.0

    def test_storage_engine_esankhyiki_persistence(self, tmp_path):
        db_path = f"sqlite:///{tmp_path}/test_esankhyiki.db"
        engine = StorageEngine(db_url=db_path)

        sample_records = [
            {
                "base_year": "2024",
                "series": "Current",
                "year": "2026",
                "month": "August",
                "state": "All India",
                "sector": "Combined",
                "division": "Transport",
                "code": "07.3.3.1.2.01",
                "item": "Airfare",
                "index": 135.49,
                "inflation": 20.85,
            },
            {
                "base_year": "2024",
                "series": "Current",
                "year": "2026",
                "month": "August",
                "state": "All India",
                "sector": "Combined",
                "division": "Transport",
                "code": "07",
                "item": None,
                "index": 105.90,
                "inflation": 4.60,
            }
        ]

        inserted = engine.store_esankhyiki_records(sample_records)
        assert inserted == 2

        benchmarks = engine.get_esankhyiki_cpi_benchmarks("2024")
        assert benchmarks["airfare"]["index"] == 135.49
        assert benchmarks["transport"]["index"] == 105.90
        assert benchmarks["cpi_weight_pct"] == 9.43
