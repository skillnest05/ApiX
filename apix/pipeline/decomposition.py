"""
6-Component Statutory Fare Decomposition Engine for APIx Pipeline (Module B).
Decomposes gross domestic airfare into:
1. Base Fare (net airline commercial tariff)
2. Fuel Surcharge (YQ/YR carrier surcharge)
3. Goods and Services Tax (GST 5% for domestic economy on Base + Fuel)
4. User Development Fee (UDF per AERA airport tariff schedules)
5. Passenger Service Fee (PSF statutory security & facilitation fee, ₹180)
6. Convenience Fee (OTA / booking channel specific charge)

Enforces strict arithmetic identity reconciliation:
|Total Fare - sum(Components)| <= 1.00 INR.
"""

from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, Optional, Union

from apix.pipeline.schemas import DecomposedFare, RawFareQuote


# ==============================================================================
# STATUTORY & REGULATORY TARIFF SCHEDULES
# ==============================================================================

# AERA Approved User Development Fees (UDF in INR) for 13 Major Indian Hubs
AERA_UDF_TARIFFS: Dict[str, float] = {
    "DEL": 650.0,   # Delhi Indira Gandhi International
    "BOM": 450.0,   # Mumbai Chhatrapati Shivaji Maharaj International
    "BLR": 550.0,   # Bengaluru Kempegowda International
    "HYD": 480.0,   # Hyderabad Rajiv Gandhi International
    "CCU": 420.0,   # Kolkata Netaji Subhash Chandra Bose International
    "PNQ": 390.0,   # Pune Lohegaon Airport
    "AMD": 350.0,   # Ahmedabad Sardar Vallabhbhai Patel International
    "MAA": 380.0,   # Chennai International Airport
    "JAI": 320.0,   # Jaipur International Airport
    "GOI": 400.0,   # Goa Dabolim / Mopa International
    "LKO": 350.0,   # Lucknow Chaudhary Charan Singh International
    "SXR": 300.0,   # Srinagar Sheikh ul-Alam International
    "GAU": 280.0,   # Guwahati Lokpriya Gopinath Bordoloi International
}

# Baseline AERA charges (conftest reference baseline)
AERA_UDF_BASELINE: Dict[str, float] = {
    "DEL": 450.0,
    "BOM": 380.0,
    "BLR": 450.0,
    "HYD": 410.0,
    "CCU": 400.0,
    "PNQ": 390.0,
    "AMD": 350.0,
    "MAA": 380.0,
    "JAI": 390.0,
    "GOI": 350.0,
    "LKO": 350.0,
    "SXR": 300.0,
    "GAU": 320.0,
}

# Standard national fallback UDF for unlisted secondary airports
DEFAULT_FALLBACK_UDF: float = 380.0

# Passenger Service Fee (statutory BCAS/MoCA passenger security & facilitation fee in INR)
STATUTORY_PSF: float = 180.0

# Statutory Goods and Services Tax (GST) rate for domestic economy air travel
GST_RATE_ECONOMY: float = 0.05  # 5% (CGST 2.5% + SGST 2.5% or IGST 5.0%)

# Typical Booking Channel Convenience Fees (INR)
CONVENIENCE_FEES: Dict[str, float] = {
    # OTAs
    "easemytrip": 0.0,
    "emt": 0.0,
    "makemytrip": 350.0,
    "mmt": 350.0,
    "cleartrip": 300.0,
    "ct": 300.0,
    "ixigo": 250.0,
    "ixi": 250.0,
    # Direct Airlines (Zero convenience fee on direct web portals)
    "indigo": 0.0,
    "6e": 0.0,
    "airindia": 0.0,
    "ai": 0.0,
    "airindiaexpress": 0.0,
    "ix": 0.0,
    "akasa": 0.0,
    "qp": 0.0,
    "spicejet": 0.0,
    "sg": 0.0,
}


# ==============================================================================
# DECOMPOSITION LOGIC
# ==============================================================================

def decompose_fare(
    total_fare: Union[float, Decimal],
    origin_iata: str,
    carrier_code: str = "6E",
    source: str = "airline",
    ota_fee: Optional[Union[float, Decimal]] = None,
    udf_rates: Optional[Dict[str, float]] = None,
    psf_rates: Optional[Dict[str, float]] = None,
    gst_rate: float = GST_RATE_ECONOMY,
) -> DecomposedFare:
    """
    Decomposes a gross domestic economy airfare into its 6 statutory components:
    Total = Base Fare + Fuel Surcharge + GST (5%) + UDF + PSF + Convenience Fee

    Args:
        total_fare: Gross airfare quoted (INR).
        origin_iata: 3-letter IATA code of origin airport (determines AERA UDF).
        carrier_code: 2-3 letter carrier code (e.g. '6E', 'AI').
        source: Booking source identifier (e.g. 'indigo', 'makemytrip').
        ota_fee: Optional explicit convenience fee override.
        udf_rates: Optional custom dictionary of UDF tariffs by origin airport.
        psf_rates: Optional custom dictionary of PSF rates by origin airport.
        gst_rate: Statutory GST percentage (default 0.05).

    Returns:
        DecomposedFare Pydantic model guaranteeing |Total - sum(Components)| <= 1.00 INR.
    """
    total = float(total_fare)
    origin_clean = origin_iata.upper().strip()

    # 1. Determine User Development Fee (UDF)
    if udf_rates is not None:
        udf = float(udf_rates.get(origin_clean, DEFAULT_FALLBACK_UDF))
    elif origin_clean in AERA_UDF_TARIFFS:
        udf = AERA_UDF_TARIFFS[origin_clean]
    elif origin_clean in AERA_UDF_BASELINE:
        udf = AERA_UDF_BASELINE[origin_clean]
    else:
        udf = DEFAULT_FALLBACK_UDF

    # 2. Determine Passenger Service Fee (PSF)
    if psf_rates is not None:
        psf = float(psf_rates.get(origin_clean, STATUTORY_PSF))
    else:
        psf = STATUTORY_PSF

    # 3. Determine Convenience Fee
    if ota_fee is not None:
        fee = float(ota_fee)
    else:
        src_clean = source.lower().strip()
        fee = float(CONVENIENCE_FEES.get(src_clean, 0.0))

    # 4. Compute Taxable Net Sum: (Base Fare + Fuel Surcharge)
    # Total = (Base + Fuel) * (1 + GST) + UDF + PSF + Fee
    # Taxable Sum = (Total - UDF - PSF - Fee) / (1 + GST)
    taxable_sum = (total - udf - psf - fee) / (1.0 + gst_rate)
    if taxable_sum < 0.0:
        taxable_sum = 0.0

    # 5. Statutory GST on taxable sum
    gst = taxable_sum * gst_rate

    # 6. Split taxable sum between Base Fare and Fuel Surcharge (YQ/YR)
    # Standard domestic yield structure: Fuel surcharge is ~25% of taxable sum, bounded [400, 1200]
    if taxable_sum <= 0.0:
        fuel = 0.0
        base = 0.0
    elif taxable_sum < 400.0:
        fuel = taxable_sum
        base = 0.0
    else:
        fuel = min(1200.0, max(400.0, taxable_sum * 0.25))
        base = taxable_sum - fuel

    # 7. Exact Decimal Reconciliation to prevent precision drift
    total_dec = Decimal(str(round(total, 2)))
    fuel_dec = Decimal(str(round(fuel, 2)))
    gst_dec = Decimal(str(round(gst, 2)))
    udf_dec = Decimal(str(round(udf, 2)))
    psf_dec = Decimal(str(round(psf, 2)))
    fee_dec = Decimal(str(round(fee, 2)))

    # Absorb residual rounding into Base Fare
    base_dec = total_dec - (fuel_dec + gst_dec + udf_dec + psf_dec + fee_dec)

    # In extreme boundary cases (e.g. fare < UDF + PSF), prevent negative base fare if total is legitimate
    if base_dec < Decimal("0.00") and total >= 999.0:
        # Scale fuel surcharge downwards to keep base non-negative
        deficit = abs(base_dec)
        fuel_dec = max(Decimal("0.00"), fuel_dec - deficit)
        base_dec = total_dec - (fuel_dec + gst_dec + udf_dec + psf_dec + fee_dec)

    return DecomposedFare(
        base_fare=base_dec,
        fuel_surcharge=fuel_dec,
        airport_taxes_gst=gst_dec,
        user_development_fee=udf_dec,
        passenger_service_fee=psf_dec,
        convenience_fee=fee_dec,
        total_fare=total_dec,
    )


def decompose_fare_dict(
    total_fare: Union[float, Decimal],
    origin_iata: str,
    carrier_code: str = "6E",
    source: str = "airline",
    ota_fee: Optional[Union[float, Decimal]] = None,
    udf_rates: Optional[Dict[str, float]] = None,
    psf_rates: Optional[Dict[str, float]] = None,
    gst_rate: float = GST_RATE_ECONOMY,
) -> Dict[str, float]:
    """Convenience helper returning the 6 decomposed components as a standard float dictionary."""
    model = decompose_fare(
        total_fare=total_fare,
        origin_iata=origin_iata,
        carrier_code=carrier_code,
        source=source,
        ota_fee=ota_fee,
        udf_rates=udf_rates,
        psf_rates=psf_rates,
        gst_rate=gst_rate,
    )
    return {
        "base_fare": float(model.base_fare),
        "fuel_surcharge": float(model.fuel_surcharge),
        "airport_taxes_gst": float(model.airport_taxes_gst),
        "user_dev_fee": float(model.user_development_fee),
        "passenger_service_fee": float(model.passenger_service_fee),
        "convenience_fee": float(model.convenience_fee),
        "total_fare": float(model.total_fare),
    }


def decompose_quote(
    quote: RawFareQuote,
    udf_rates: Optional[Dict[str, float]] = None,
    psf_rates: Optional[Dict[str, float]] = None,
) -> DecomposedFare:
    """Decompose a RawFareQuote into its statutory 6-component DecomposedFare."""
    return decompose_fare(
        total_fare=quote.raw_total_fare,
        origin_iata=quote.origin_iata,
        carrier_code=quote.carrier_code,
        source=quote.source,
        ota_fee=quote.raw_convenience_fee,
        udf_rates=udf_rates,
        psf_rates=psf_rates,
    )


def verify_decomposition_identity(
    fare: DecomposedFare,
    tolerance: float = 1.00
) -> bool:
    """Validates that sum(components) equals total_fare within tolerance."""
    return fare.is_balanced(tolerance=tolerance)
