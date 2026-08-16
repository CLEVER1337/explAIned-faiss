"""HTTP surface. The shape of /search is fixed by explAInedRecommendationService/CONTRACT.md —
`{user_id, top_k}` in, `{ids, scores}` out. Do not reinvent it here.

Status codes the orchestrator distinguishes:

* 200 — hits (possibly an empty list once the index is warm but the user matches nothing)
* 204 — no `rec:user_embedding:{uid}`. The FAISS source reports `Disabled`; not degradation.
* 503 — no index loaded. The orchestrator sees `Faulted`; that *is* an error.
"""

import time

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from redis.exceptions import RedisError

from ..metrics import SEARCH_SECONDS, SEARCH_TOTAL
from .state import IndexState

router = APIRouter()


class SearchRequest(BaseModel):
    user_id: str = Field(min_length=1)
    top_k: int = Field(default=200, ge=1)


class SearchResponse(BaseModel):
    ids: list[str]
    scores: list[float]


@router.post("/search", response_model=SearchResponse, responses={204: {}, 503: {}})
async def search(payload: SearchRequest, request: Request) -> Response:
    settings = request.app.state.settings
    state: IndexState = request.app.state.index
    store = request.app.state.store

    started = time.perf_counter()
    try:
        bundle = state.bundle
        if bundle is None:
            SEARCH_TOTAL.labels(result="no_index").inc()
            return JSONResponse({"detail": "index not loaded"}, status_code=503)

        try:
            embedding = await store.get_user_embedding(payload.user_id)
        except RedisError as exc:
            SEARCH_TOTAL.labels(result="no_index").inc()
            return JSONResponse({"detail": f"redis unavailable: {exc}"}, status_code=503)

        if embedding is None:
            SEARCH_TOTAL.labels(result="no_embedding").inc()
            return Response(status_code=204)

        top_k = min(payload.top_k, settings.default_top_k)
        ids, scores = bundle.search(embedding, top_k)

        SEARCH_TOTAL.labels(result="hit").inc()
        return JSONResponse({"ids": ids, "scores": scores})
    finally:
        SEARCH_SECONDS.observe(time.perf_counter() - started)


@router.post("/reload")
async def reload(request: Request) -> Response:
    state: IndexState = request.app.state.index
    outcome = await state.reload(force=True)

    body = {"result": outcome.result, "error": outcome.error, **state.describe()}
    status = 200 if outcome.result in {"loaded", "unchanged"} else 503

    return JSONResponse(body, status_code=status)


@router.get("/health")
async def health(request: Request) -> dict:
    """Always 200 — liveness, not readiness. `index_loaded` carries the real state."""
    state: IndexState = request.app.state.index
    return {"status": "ok", **state.describe()}
