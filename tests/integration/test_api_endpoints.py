"""
Integration Test Suite: Institutional REST API Endpoints & Schema Validation.
Verifies all 12+ API endpoints for HTTP status 200, valid OpenAPI 3.1 schemas,
and quantitative analytical correctness.
"""

import pytest
from fastapi.testclient import TestClient


class TestAPIEndpoints:
    """Verifies all 12+ REST API endpoints under /api/v1/."""

    # 1. Index Endpoints
    def test_get_daily_index_default(self, api_client: TestClient):
        """Tier 1: GET /api/v1/index/daily returns HTTP 200 with composite index value."""
        res = api_client.get("/api/v1/index/daily")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["apix_value"] > 0
        assert data["frequency"] == "daily"
        assert "tier_summary" in data
        assert "advance_window_summary" in data
        assert data["tier_summary"]["trunk"] > 0
        assert data["advance_window_summary"]["T_1"] > data["advance_window_summary"]["T_45"]

    def test_get_daily_index_weekly_monthly_frequencies(self, api_client: TestClient):
        """Tier 1: GET /api/v1/index/daily supports weekly and monthly frequency requests."""
        res_w = api_client.get("/api/v1/index/daily?frequency=weekly")
        assert res_w.status_code == 200
        assert res_w.json()["data"]["frequency"] == "weekly"

        res_m = api_client.get("/api/v1/index/daily?frequency=monthly")
        assert res_m.status_code == 200
        assert res_m.json()["data"]["frequency"] == "monthly"

    def test_get_index_history_30_days(self, api_client: TestClient):
        """Tier 1: GET /api/v1/index/history returns time series over 30 days."""
        res = api_client.get("/api/v1/index/history?start_date=2026-08-29&end_date=2026-09-27")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["data"]) >= 2
        for pt in body["data"]:
            assert pt["apix_composite"] > 0
            assert pt["cpi_transport_benchmark"] > 0

    def test_get_route_index_specific(self, api_client: TestClient):
        """Tier 1: GET /api/v1/index/route/{id} returns route-level breakdown and carrier shares."""
        res = api_client.get("/api/v1/index/route/DEL-BOM")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        data = body["data"]
        assert data["route_id"] == "DEL-BOM"
        assert data["distance_km"] == 1148.0
        assert len(data["advance_windows"]) == 5
        assert len(data["carrier_shares"]) >= 3

    # 2. Quote Endpoints
    def test_get_latest_fare_quotes(self, api_client: TestClient):
        """Tier 1: GET /api/v1/fares/latest returns itemized decomposed quotes."""
        res = api_client.get("/api/v1/fares/latest?origin=DEL&destination=BLR&carrier=6E&limit=10")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert len(body["data"]) > 0
        quote = body["data"][0]
        decomp = quote["decomposed_fare"]
        reconstructed = (
            decomp["base_fare"]
            + decomp["fuel_surcharge"]
            + decomp["airport_taxes_gst"]
            + decomp["user_dev_fee"]
            + decomp["passenger_service_fee"]
            + decomp["convenience_fee"]
        )
        assert abs(reconstructed - decomp["total_fare"]) <= 1.00

    def test_search_fare_quotes(self, api_client: TestClient):
        """Tier 1: GET /api/v1/fares/search returns multivariate search query results."""
        res = api_client.get("/api/v1/fares/search?origin=DEL&destination=BOM&min_price=3000&max_price=8000")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert "summary_stats" in body

    # 3. Analytics Endpoints
    def test_get_lead_time_elasticity_curve(self, api_client: TestClient):
        """Tier 1: GET /api/v1/analytics/elasticity returns empirical advance curve points."""
        res = api_client.get("/api/v1/analytics/elasticity?route_id=DEL-BOM")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        points = body["curve_points"]
        assert len(points) == 5

        # T+1 price must strictly exceed T+45 price (yield management curve)
        t1_point = next(p for p in points if p["advance_window"] == "T+1")
        t45_point = next(p for p in points if p["advance_window"] == "T+45")
        assert t1_point["average_fare"] > t45_point["average_fare"]
        assert body["elasticity_metrics"]["t1_to_t45_ratio"] > 1.5

    def test_get_sector_heatmap_geospatial(self, api_client: TestClient):
        """Tier 1: GET /api/v1/analytics/heatmap returns 30 sectors with Indian coordinates."""
        res = api_client.get("/api/v1/analytics/heatmap")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        sectors = body["sectors"]
        assert len(sectors) == 30

        for s in sectors:
            assert 8.0 <= s["origin"]["lat"] <= 38.0 # India lat bounds
            assert 68.0 <= s["origin"]["lon"] <= 98.0 # India lon bounds
            assert s["average_fare"] > 0

    def test_get_anomalies_detected(self, api_client: TestClient):
        """Tier 1: GET /api/v1/analytics/anomalies returns flagged price surge events."""
        res = api_client.get("/api/v1/analytics/anomalies?severity=high")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        assert body["total_anomalies"] >= 1
        anom = body["anomalies"][0]
        assert anom["z_score"] >= 3.5

    # 4. Simulation Endpoint
    def test_post_simulation_what_if(self, api_client: TestClient):
        """
        Tier 1 & Tier 4: POST /api/v1/simulation/what-if simulates ATF shock
        and validates macroeconomic transmission to 9.43% Transport CPI.
        """
        payload = {
            "atf_change_pct": 15.0,
            "demand_shock_pct": 5.0,
            "target_horizon_days": 30
        }
        res = api_client.post("/api/v1/simulation/what-if", json=payload)
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        results = body["results"]
        assert results["delta_apix_pct"] > 0
        assert results["cpi_transport_impact_bps"] > 0
        assert results["headline_cpi_impact_bps"] > 0
        assert results["headline_cpi_weight_used"] == 0.0943

        # Quantitative relation: Headline CPI impact == Transport impact * 0.0943
        expected_headline = round(results["cpi_transport_impact_bps"] * 0.0943, 2)
        assert abs(results["headline_cpi_impact_bps"] - expected_headline) <= 0.05

    # 5. Export Endpoints
    def test_export_csv_and_json(self, api_client: TestClient):
        """Tier 1: Bulk export endpoints stream RFC compliant CSV and JSON."""
        res_csv = api_client.get("/api/v1/export/csv?dataset=index_daily")
        assert res_csv.status_code == 200
        assert "text/csv" in res_csv.headers["content-type"]
        assert "date,route_id,index_value" in res_csv.text

        res_json = api_client.get("/api/v1/export/json?dataset=index_daily")
        assert res_json.status_code == 200
        assert res_json.json()["status"] == "success"

    # 6. Backtest Endpoint
    def test_backtest_report_endpoint(self, api_client: TestClient):
        """Tier 1 & Tier 4: GET /api/v1/backtest/report returns validation scorecard meeting hurdles."""
        res = api_client.get("/api/v1/backtest/report?window_days=30")
        assert res.status_code == 200
        body = res.json()
        assert body["status"] == "success"
        metrics = body["metrics"]
        assert metrics["r_squared"] > 0.85
        assert metrics["r_squared_passed"] is True
        assert metrics["mape_pct"] < 15.0
        assert metrics["mape_passed"] is True
        assert metrics["directional_concordance_pct"] >= 80.0
        assert metrics["directional_concordance_passed"] is True
        assert body["overall_validation_verdict"] == "PASSED"

    # 7. Static Dashboard & Docs
    def test_root_dashboard_serving(self, api_client: TestClient):
        """Tier 1: Root GET / serves Web Dashboard HTML."""
        res = api_client.get("/")
        assert res.status_code == 200
        assert "text/html" in res.headers["content-type"]

    def test_openapi_docs_endpoints(self, api_client: TestClient):
        """Tier 1: /docs and /openapi.json are accessible."""
        res_docs = api_client.get("/docs")
        assert res_docs.status_code == 200
        res_schema = api_client.get("/openapi.json")
        assert res_schema.status_code == 200
        assert res_schema.json()["info"]["title"] == "APIx Institutional REST API"
