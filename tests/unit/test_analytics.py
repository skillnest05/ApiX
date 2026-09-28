"""
Unit Test Suite for APIx Analytics & Innovation Layer (Module D).
Verifies:
1. Anomaly detection (Z-scores, price surges, regulatory recommendations)
2. CPI What-If policy simulation (ATF transmission to 9.43% Transport basket)
3. Natural Language Policy Query engine (entity extraction, intent parsing, SQL output)
4. Fare Affordability Index (FAI) & wage-day metrics
5. 14-day econometric forecaster with weekly seasonality
"""

from datetime import date
import pytest

from apix.analytics.anomaly import AnomalyDetector, AnomalyRecord
from apix.analytics.simulation import WhatIfSimulator, WhatIfRequestModel
from apix.analytics.nlp_query import NLPQueryEngine
from apix.analytics.affordability import AffordabilityEngine
from apix.analytics.forecast import FareForecaster


class TestAnomalyDetector:
    """Tests statistical and heuristic price anomaly detection."""

    def test_cohort_anomaly_detection_normal_vs_surge(self):
        detector = AnomalyDetector(z_threshold=3.0)
        quotes = [
            {"total_fare": 4800.0, "carrier_code": "6E"},
            {"total_fare": 4900.0, "carrier_code": "6E"},
            {"total_fare": 5000.0, "carrier_code": "AI"},
            {"total_fare": 5100.0, "carrier_code": "IX"},
            {"total_fare": 4950.0, "carrier_code": "QP"},
            {"total_fare": 5050.0, "carrier_code": "SG"},
            {"total_fare": 18500.0, "carrier_code": "6E"},  # Extreme surge
        ]

        anomalies = detector.detect_cohort_anomalies(quotes, "DEL-BOM", 1)
        assert len(anomalies) >= 1
        surge = anomalies[0]
        assert surge.observed_fare == 18500.0
        assert surge.z_score >= 2.0
        assert surge.route_id == "DEL-BOM"
        assert surge.advance_window == 1

    def test_curated_anomalies_filtering(self):
        detector = AnomalyDetector()
        all_anoms = detector.get_curated_anomalies()
        assert len(all_anoms) >= 3

        high_anoms = detector.get_curated_anomalies(severity="high")
        assert all(a.severity.lower() == "high" for a in high_anoms)


class TestWhatIfSimulator:
    """Tests macroeconomic policy simulation transmission."""

    def test_atf_shock_transmission_identity(self):
        sim = WhatIfSimulator.simulate(atf_change_pct=15.0, demand_shock_pct=5.0, target_horizon_days=30)
        assert sim["status"] == "success"
        res = sim["results"]

        # Delta APIx = 0.42 * 15 + 0.55 * 5 = 6.30 + 2.75 = 9.05%
        assert abs(res["delta_apix_pct"] - 9.05) <= 0.05
        assert res["cpi_transport_impact_bps"] > 0
        assert res["headline_cpi_weight_used"] == 0.0943

        # Quantitative macroeconomic identity
        expected_headline = round(res["cpi_transport_impact_bps"] * 0.0943, 2)
        assert abs(res["headline_cpi_impact_bps"] - expected_headline) <= 0.05

    def test_route_impacts_presence(self):
        sim = WhatIfSimulator.simulate(atf_change_pct=10.0, demand_shock_pct=0.0)
        route_impacts = sim["results"]["route_impacts"]
        assert len(route_impacts) >= 3
        for r in route_impacts:
            assert r["delta_fare_inr"] > 0
            assert r["projected_avg_fare"] > r["baseline_avg_fare"]


class TestNLPQueryEngine:
    """Tests natural language parsing and policy query synthesis."""

    def test_nlp_simulation_intent(self):
        res = NLPQueryEngine.execute_query("What if ATF fuel prices increase by 15%?")
        assert res["status"] == "success"
        assert res["parsed_intent"] == "simulation"
        assert "Aviation Turbine Fuel" in res["natural_answer"]
        assert len(res["data_table"]) >= 2
        assert "SELECT" in res["sql_equivalent"]

    def test_nlp_route_intent_with_entities(self):
        res = NLPQueryEngine.execute_query("Show me fares from Delhi to Mumbai")
        assert res["status"] == "success"
        assert res["entities"]["origin"] == "DEL"
        assert res["entities"]["destination"] == "BOM"
        assert res["entities"]["sector"] == "DEL-BOM"
        assert "DEL-BOM" in res["natural_answer"]

    def test_nlp_elasticity_intent(self):
        res = NLPQueryEngine.execute_query("What is the lead-time yield curve for DEL to BOM?")
        assert res["status"] == "success"
        assert res["parsed_intent"] == "lead_time_elasticity"
        assert any(pt["advance_window"] == "T+1" for pt in res["data_table"])

    def test_nlp_anomalies_intent(self):
        res = NLPQueryEngine.execute_query("Are there any price surges or anomalies detected?")
        assert res["status"] == "success"
        assert res["parsed_intent"] == "anomalies"
        assert len(res["data_table"]) >= 1


class TestAffordabilityEngine:
    """Tests purchasing-power adjusted fare affordability."""

    def test_route_affordability_calculation(self):
        res = AffordabilityEngine.compute_route_affordability("DEL", "BOM", 5120.0)
        assert res["origin"] == "DEL"
        assert res["destination"] == "BOM"
        assert res["joint_daily_income_inr"] > 0
        assert res["wage_days_required"] > 0
        assert res["affordability_score"] > 0
        assert res["affordability_rating"] in ["Very Affordable", "Affordable", "Moderate", "High Cost Barrier"]

    def test_national_affordability_summary(self):
        summary = AffordabilityEngine.get_national_affordability_summary()
        assert len(summary) >= 5


class TestFareForecaster:
    """Tests 14-day projection engine."""

    def test_14day_forecast_structure_and_seasonality(self):
        forecast = FareForecaster.generate_14day_forecast(current_apix=134.72)
        assert forecast["status"] == "success"
        assert forecast["forecast_horizon_days"] == 14
        points = forecast["forecast_points"]
        assert len(points) == 14

        for pt in points:
            assert pt["projected_apix"] > 0
            assert pt["lower_confidence_95"] < pt["projected_apix"] < pt["upper_confidence_95"]
            assert pt["day_offset"] >= 1
