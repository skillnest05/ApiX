"""
CPI Policy 'What-If' Simulation Engine for APIx (Innovation Layer).
Simulates macro shocks (Aviation Turbine Fuel / ATF price changes, festival demand surges)
and projects the econometric transmission through APIx, the 9.43% CPI Transport sub-group,
and the headline All-India Consumer Price Index.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class WhatIfRequestModel(BaseModel):
    """Parameters for macroeconomic policy simulation."""
    model_config = ConfigDict(frozen=True)

    atf_change_pct: float = Field(default=0.0, description="Hypothetical percentage change in ATF prices (%)")
    demand_shock_pct: float = Field(default=0.0, description="Hypothetical surge in passenger demand (%)")
    target_horizon_days: int = Field(default=30, description="Projection horizon in days (e.g. 15, 30, 60)")


class WhatIfSimulator:
    """
    Econometric policy transmission model for MoSPI and RBI.
    Calibrated against Indian aviation cost structures and COICOP 2018 / CPI 2024=100 weights.
    """

    # Empirical Indian Aviation Elasticities
    ELASTICITY_ATF: float = 0.42     # Fuel is ~40-42% of Indian airline operating costs
    ELASTICITY_DEMAND: float = 0.55  # Dynamic pricing response to passenger demand surges
    AIRFARE_SHARE_IN_TRANSPORT: float = 0.032 # Air travel weight in Transport sub-group
    CPI_TRANSPORT_WEIGHT: float = 0.0943      # 9.43% Transport weight in Headline CPI (2024=100)

    @classmethod
    def simulate(
        cls,
        atf_change_pct: float,
        demand_shock_pct: float,
        target_horizon_days: int = 30,
        baseline_apix: float = 134.72
    ) -> Dict[str, Any]:
        """
        Calculates projected airfare price index shift and macroeconomic transmission.
        """
        # Delta APIx percentage
        delta_apix = (cls.ELASTICITY_ATF * atf_change_pct) + (cls.ELASTICITY_DEMAND * demand_shock_pct)
        projected_apix = baseline_apix * (1.0 + delta_apix / 100.0)

        # Transmission to CPI Transport sub-group (in basis points, 1% = 100 bps)
        cpi_transport_impact_bps = delta_apix * cls.AIRFARE_SHARE_IN_TRANSPORT * 100.0

        # Transmission to All-India Headline CPI (in basis points)
        headline_cpi_impact_bps = round(cpi_transport_impact_bps * cls.CPI_TRANSPORT_WEIGHT, 2)

        # Route-level empirical fare impact simulations
        routes_impact = [
            {
                "route": "DEL-BOM",
                "baseline_avg_fare": 5120.0,
                "delta_fare_inr": round(5120.0 * (delta_apix / 100.0), 1),
                "projected_avg_fare": round(5120.0 * (1.0 + delta_apix / 100.0), 1)
            },
            {
                "route": "DEL-BLR",
                "baseline_avg_fare": 6350.0,
                "delta_fare_inr": round(6350.0 * (delta_apix / 100.0), 1),
                "projected_avg_fare": round(6350.0 * (1.0 + delta_apix / 100.0), 1)
            },
            {
                "route": "BOM-BLR",
                "baseline_avg_fare": 4480.0,
                "delta_fare_inr": round(4480.0 * (delta_apix / 100.0), 1),
                "projected_avg_fare": round(4480.0 * (1.0 + delta_apix / 100.0), 1)
            },
            {
                "route": "DEL-CCU",
                "baseline_avg_fare": 5890.0,
                "delta_fare_inr": round(5890.0 * (delta_apix / 100.0), 1),
                "projected_avg_fare": round(5890.0 * (1.0 + delta_apix / 100.0), 1)
            }
        ]

        return {
            "status": "success",
            "simulation_inputs": {
                "atf_change_pct": round(atf_change_pct, 2),
                "demand_shock_pct": round(demand_shock_pct, 2),
                "target_horizon_days": target_horizon_days
            },
            "results": {
                "baseline_apix": round(baseline_apix, 2),
                "projected_apix": round(projected_apix, 2),
                "delta_apix_pct": round(delta_apix, 2),
                "cpi_transport_impact_bps": round(cpi_transport_impact_bps, 2),
                "headline_cpi_impact_bps": headline_cpi_impact_bps,
                "headline_cpi_weight_used": cls.CPI_TRANSPORT_WEIGHT,
                "airfare_share_in_transport": cls.AIRFARE_SHARE_IN_TRANSPORT,
                "route_impacts": routes_impact
            }
        }
