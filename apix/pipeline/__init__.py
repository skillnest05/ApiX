"""
APIx Data Pipeline Package (Module B).
Provides data cleaning, normalization, statutory fare decomposition,
outlier filtering, quality scoring, and relational storage.
"""

from apix.pipeline.schemas import (
    CarrierCode,
    CleanedFareQuote,
    DQSBreakdown,
    DecomposedFare,
    FareClass,
    OutlierEvaluation,
    OutlierReason,
    RawFareQuote,
    RouteTier,
    SourceType,
)

from apix.pipeline.decomposition import (
    AERA_UDF_BASELINE,
    AERA_UDF_TARIFFS,
    CONVENIENCE_FEES,
    DEFAULT_FALLBACK_UDF,
    GST_RATE_ECONOMY,
    STATUTORY_PSF,
    decompose_fare,
    decompose_fare_dict,
    decompose_quote,
    verify_decomposition_identity,
)

from apix.pipeline.outlier import (
    OutlierDetector,
    compute_kendall_w,
    evaluate_outlier_status,
    filter_iqr_bounds,
    is_cross_source_variance_violation,
    is_domain_ceiling_violation,
    is_domain_floor_violation,
    is_temporal_surge_violation,
)

from apix.pipeline.quality import (
    clean_quote,
    clean_quotes_batch,
    compute_dqs_breakdown,
    compute_dqs_score,
    deduplicate_quotes,
    handle_sold_out,
)

from apix.pipeline.storage import (
    AtfPriceRecord,
    Carrier,
    CpiReferenceRecord,
    DailyIndexRecord,
    FareQuoteRecord,
    Route,
    StorageEngine,
    get_clean_quotes,
    get_storage_engine,
    insert_quotes,
)

__all__ = [
    # Schemas
    "CarrierCode",
    "CleanedFareQuote",
    "DQSBreakdown",
    "DecomposedFare",
    "FareClass",
    "OutlierEvaluation",
    "OutlierReason",
    "RawFareQuote",
    "RouteTier",
    "SourceType",
    # Decomposition
    "AERA_UDF_BASELINE",
    "AERA_UDF_TARIFFS",
    "CONVENIENCE_FEES",
    "DEFAULT_FALLBACK_UDF",
    "GST_RATE_ECONOMY",
    "STATUTORY_PSF",
    "decompose_fare",
    "decompose_fare_dict",
    "decompose_quote",
    "verify_decomposition_identity",
    # Outliers
    "OutlierDetector",
    "compute_kendall_w",
    "evaluate_outlier_status",
    "filter_iqr_bounds",
    "is_cross_source_variance_violation",
    "is_domain_ceiling_violation",
    "is_domain_floor_violation",
    "is_temporal_surge_violation",
    # Quality & Deduplication
    "clean_quote",
    "clean_quotes_batch",
    "compute_dqs_breakdown",
    "compute_dqs_score",
    "deduplicate_quotes",
    "handle_sold_out",
    # Storage
    "AtfPriceRecord",
    "Carrier",
    "CpiReferenceRecord",
    "DailyIndexRecord",
    "FareQuoteRecord",
    "Route",
    "StorageEngine",
    "get_clean_quotes",
    "get_storage_engine",
    "insert_quotes",
]
