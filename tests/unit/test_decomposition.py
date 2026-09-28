"""
Unit Test Suite: 6-Component Statutory Fare Decomposition Identity.
Tests the statutory decomposition formula:
Total Fare = Base Fare + Fuel Surcharge + GST (5%) + UDF + PSF + Convenience Fee
Enforces arithmetic identity reconciliation (|Total - sum(Components)| <= 1.00 INR).
"""

from decimal import Decimal
import pytest

from tests.conftest import (
    oracle_decompose_fare,
    DecomposedFareModel,
    AERA_CHARGES,
    AIRPORT_COORDS,
)


class TestFareDecomposition:
    """Tests for 6-Component Fare Decomposition across Tiers 1, 2, and 3."""

    # --------------------------------------------------------------------------
    # Tier 1: Feature Coverage (Primary Happy Paths across Origin Airports)
    # --------------------------------------------------------------------------

    @pytest.mark.parametrize("origin", list(AERA_CHARGES.keys()))
    def test_arithmetic_identity_all_airports(self, origin):
        """
        Tier 1: Verify exact identity Total == Base + Fuel + GST + UDF + PSF + Fee
        for standard domestic economy fare across all 13 origin airports.
        """
        total_fare = 5850.0
        decomp = oracle_decompose_fare(total_fare=total_fare, origin_iata=origin, ota_fee=0.0)

        model = DecomposedFareModel(**decomp)
        reconstructed = (
            model.base_fare
            + model.fuel_surcharge
            + model.airport_taxes_gst
            + model.user_dev_fee
            + model.passenger_service_fee
            + model.convenience_fee
        )

        assert abs(reconstructed - total_fare) <= 1.00, (
            f"Arithmetic identity breached at {origin}: "
            f"Expected {total_fare}, got {reconstructed} (diff: {abs(reconstructed - total_fare)})"
        )
        assert model.user_dev_fee == AERA_CHARGES[origin]["udf"]
        assert model.passenger_service_fee == AERA_CHARGES[origin]["psf"]

    def test_statutory_gst_rate_5_pct(self):
        """Tier 1: Statutory Goods and Services Tax (GST) is exactly 5.0% on (Base + Fuel)."""
        decomp = oracle_decompose_fare(total_fare=6000.0, origin_iata="DEL", ota_fee=0.0)
        taxable_base = decomp["base_fare"] + decomp["fuel_surcharge"]
        expected_gst = taxable_base * 0.05
        assert abs(decomp["airport_taxes_gst"] - expected_gst) <= 1.00

    def test_direct_carrier_zero_convenience_fee(self):
        """Tier 1: Direct carrier quotes (e.g. IndiGo, Air India) have 0 convenience fee."""
        decomp = oracle_decompose_fare(total_fare=4200.0, origin_iata="BOM", ota_fee=0.0)
        assert decomp["convenience_fee"] == 0.0

    def test_ota_convenience_fee_incorporation(self):
        """Tier 1: OTA quotes with explicit convenience fee (e.g. MMT ₹350) reconcile perfectly."""
        decomp = oracle_decompose_fare(total_fare=4550.0, origin_iata="BOM", ota_fee=350.0)
        assert decomp["convenience_fee"] == 350.0
        reconstructed = sum([
            decomp["base_fare"],
            decomp["fuel_surcharge"],
            decomp["airport_taxes_gst"],
            decomp["user_dev_fee"],
            decomp["passenger_service_fee"],
            decomp["convenience_fee"],
        ])
        assert abs(reconstructed - 4550.0) <= 1.00

    def test_pydantic_schema_validation(self):
        """Tier 1: Decomposed output validates against Pydantic DecomposedFareModel."""
        decomp = oracle_decompose_fare(7200.0, origin_iata="BLR")
        validated = DecomposedFareModel.model_validate(decomp)
        assert validated.total_fare == 7200.0
        assert validated.base_fare > 0
        assert validated.fuel_surcharge > 0

    # --------------------------------------------------------------------------
    # Tier 2: Boundary & Corner Cases (Extreme Fares & Tolerances)
    # --------------------------------------------------------------------------

    def test_boundary_minimum_fare_999(self):
        """Tier 2 Boundary: Minimum domain floor fare (₹999) decomposes without negative components."""
        decomp = oracle_decompose_fare(total_fare=999.0, origin_iata="SXR")
        assert decomp["base_fare"] >= 0.0
        assert decomp["fuel_surcharge"] >= 0.0
        assert decomp["total_fare"] == 999.0
        reconstructed = sum([
            decomp["base_fare"],
            decomp["fuel_surcharge"],
            decomp["airport_taxes_gst"],
            decomp["user_dev_fee"],
            decomp["passenger_service_fee"],
            decomp["convenience_fee"],
        ])
        assert abs(reconstructed - 999.0) <= 1.00

    def test_boundary_maximum_economy_fare_25000(self):
        """Tier 2 Boundary: Maximum allowable economy ceiling fare (₹25,000) decomposes accurately."""
        decomp = oracle_decompose_fare(total_fare=25000.0, origin_iata="DEL")
        assert decomp["base_fare"] > 15000.0
        assert decomp["total_fare"] == 25000.0
        reconstructed = sum([
            decomp["base_fare"],
            decomp["fuel_surcharge"],
            decomp["airport_taxes_gst"],
            decomp["user_dev_fee"],
            decomp["passenger_service_fee"],
            decomp["convenience_fee"],
        ])
        assert abs(reconstructed - 25000.0) <= 1.00

    def test_boundary_rounding_absorption_within_1_inr(self):
        """Tier 2 Boundary: Rounding discrepancies up to 1.00 INR are absorbed into Base Fare."""
        total = 4999.73
        decomp = oracle_decompose_fare(total, origin_iata="PNQ")
        reconstructed = sum([
            decomp["base_fare"],
            decomp["fuel_surcharge"],
            decomp["airport_taxes_gst"],
            decomp["user_dev_fee"],
            decomp["passenger_service_fee"],
            decomp["convenience_fee"],
        ])
        assert abs(reconstructed - round(total, 2)) <= 0.01

    def test_corner_unknown_origin_airport_fallback(self):
        """Tier 2 Corner: Origin airport not in explicit AERA table uses standard national fallback."""
        decomp = oracle_decompose_fare(total_fare=5000.0, origin_iata="IXC") # Chandigarh
        assert decomp["user_dev_fee"] == 380.0
        assert decomp["passenger_service_fee"] == 180.0
        reconstructed = sum([
            decomp["base_fare"],
            decomp["fuel_surcharge"],
            decomp["airport_taxes_gst"],
            decomp["user_dev_fee"],
            decomp["passenger_service_fee"],
            decomp["convenience_fee"],
        ])
        assert abs(reconstructed - 5000.0) <= 1.00

    # --------------------------------------------------------------------------
    # Tier 3: Combinatorial & Multi-Source Interactions
    # --------------------------------------------------------------------------

    def test_convenience_fee_spread_across_sources(self):
        """
        Tier 3: Comparing decomposition across direct airline (₹0) and 3 OTAs
        (EaseMyTrip ₹0, Ixigo ₹199, MakeMyTrip ₹350) for identical base tariff.
        """
        base_tariff = 4000.0
        sources = [
            ("airline_direct", 0.0),
            ("easemytrip", 0.0),
            ("ixigo", 199.0),
            ("makemytrip", 350.0),
        ]

        decomps = {}
        for src, fee in sources:
            total = base_tariff + fee
            decomps[src] = oracle_decompose_fare(total, origin_iata="DEL", ota_fee=fee)
            assert decomps[src]["convenience_fee"] == fee

        # Difference in total fares equals exactly the convenience fee spread
        diff_mmt_direct = decomps["makemytrip"]["total_fare"] - decomps["airline_direct"]["total_fare"]
        assert abs(diff_mmt_direct - 350.0) <= 1.00
