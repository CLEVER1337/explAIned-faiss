"""Prometheus series owned by this service.

HTTP-level series (`http_request_duration_seconds` and friends) come from
prometheus-fastapi-instrumentator; everything here is about the index itself.
"""

from prometheus_client import Counter, Gauge, Histogram

INDEX_VECTORS = Gauge("faiss_index_vectors", "Vectors in the currently loaded index")

INDEX_AGE_SECONDS = Gauge(
    "faiss_index_age_seconds",
    "Seconds since the loaded index was built — the freshness signal for the whole content loop",
)

SEARCH_SECONDS = Histogram(
    "faiss_search_seconds",
    "Time spent inside /search, Redis read included",
    buckets=(0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.12, 0.25),
)

SEARCH_TOTAL = Counter(
    "faiss_search_total",
    "Searches by outcome",
    labelnames=("result",),  # hit | no_embedding | no_index
)

RELOAD_TOTAL = Counter(
    "faiss_index_reload_total",
    "Index reloads by outcome",
    labelnames=("result",),  # loaded | unchanged | missing | failed
)
