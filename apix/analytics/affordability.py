"""
Fare Affordability Index (FAI) Engine for APIx (Innovation Layer).
Calculates real purchasing-power adjusted airfare affordability across Indian metros
and Tier-2 regional hubs based on per-capita Net State Domestic Product (NSDP) data.
"""

from typing import Any, Dict, List, Optional


# Per-capita annual net domestic income by metro/state hub (INR approx. 2024-25 MoSPI NSDP)
CITY_PER_CAPITA_ANNUAL_INR: Dict[str, float] = {
    "DEL": 444768.0,  # Delhi NCT
    "BOM": 385000.0,  # Mumbai / Maharashtra
    "BLR": 365000.0,  # Bengaluru / Karnataka
    "HYD": 348000.0,  # Hyderabad / Telangana
    "CCU": 168000.0,  # Kolkata / West Bengal
    "PNQ": 370000.0,  # Pune / Maharashtra
    "AMD": 310000.0,  # Ahmedabad / Gujarat
    "MAA": 320000.0,  # Chennai / Tamil Nadu
    "JAI": 185000.0,  # Jaipur / Rajasthan
    "GOI": 520000.0,  # Goa
    "LKO": 115000.0,  # Lucknow / Uttar Pradesh
    "SXR": 145000.0,  # Srinagar / Jammu & Kashmir
    "GAU": 140000.0,  # Guwahati / Assam
}


class AffordabilityEngine:
    """Computes Fare Affordability Index and wage-days to fly."""

    @classmethod
    def compute_route_affordability(
        cls,
        origin_iata: str,
        destination_iata: str,
        average_fare_inr: float
    ) -> Dict[str, Any]:
        """Calculates affordability metrics for a given city-pair."""
        inc_o = CITY_PER_CAPITA_ANNUAL_INR.get(origin_iata, 250000.0)
        inc_d = CITY_PER_CAPITA_ANNUAL_INR.get(destination_iata, 250000.0)
        avg_annual_income = (inc_o + inc_d) / 2.0
        daily_income_inr = avg_annual_income / 365.0

        # Wage days required to purchase one one-way ticket
        wage_days = average_fare_inr / daily_income_inr if daily_income_inr > 0 else 5.0

        # FAI score: 100 = average affordability benchmark (e.g. 5 days of wage)
        # Higher score = more affordable, lower = less affordable
        fai_score = round((5.0 / wage_days) * 100.0, 1)

        rating = "Very Affordable" if fai_score >= 120 else ("Affordable" if fai_score >= 90 else ("Moderate" if fai_score >= 70 else "High Cost Barrier"))

        return {
            "origin": origin_iata,
            "destination": destination_iata,
            "average_fare_inr": round(average_fare_inr, 2),
            "joint_daily_income_inr": round(daily_income_inr, 2),
            "wage_days_required": round(wage_days, 1),
            "affordability_score": fai_score,
            "affordability_rating": rating
        }

    @classmethod
    def get_national_affordability_summary(cls) -> List[Dict[str, Any]]:
        """Returns baseline affordability ratings across major sectors."""
        sample_sectors = [
            ("DEL", "BOM", 5120.0),
            ("DEL", "BLR", 6350.0),
            ("BOM", "BLR", 4480.0),
            ("DEL", "HYD", 5280.0),
            ("DEL", "CCU", 5890.0),
            ("DEL", "LKO", 3850.0),
            ("BOM", "GOI", 3980.0),
            ("DEL", "SXR", 8200.0),
            ("CCU", "GAU", 4120.0),
        ]
        return [cls.compute_route_affordability(o, d, fare) for o, d, fare in sample_sectors]
