"""FastAPI application factory.

Run it with exactly one uvicorn worker: the index lives in process memory, so every extra worker
is another full copy of it loaded from the same artifact.

    uvicorn explained_faiss.service.app:app --port 8001 --workers 1
"""

import asyncio
import logging
from contextlib import asynccontextmanager, suppress

from fastapi import FastAPI
from prometheus_fastapi_instrumentator import Instrumentator

from ..config import Settings, get_settings
from ..metrics import INDEX_AGE_SECONDS
from ..store import EmbeddingStore
from .routes import router
from .state import IndexState, watch_manifest

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, store: EmbeddingStore | None = None) -> FastAPI:
    settings = settings or get_settings()
    index_state = IndexState(settings.index_dir)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = settings
        app.state.index = index_state
        app.state.store = store or EmbeddingStore(settings.redis_url, settings.embedding_dim)

        # A missing index is a normal cold start: /search answers 503 until the builder has run.
        outcome = await index_state.reload(force=True)
        if outcome.result != "loaded":
            logger.warning("starting without an index (%s: %s)", outcome.result, outcome.error)

        watcher = asyncio.create_task(watch_manifest(index_state, settings.reload_watch_seconds))
        try:
            yield
        finally:
            watcher.cancel()
            with suppress(asyncio.CancelledError):
                await watcher
            if store is None:
                await app.state.store.close()

    app = FastAPI(title="explAIned FAISS", version="0.1.0", lifespan=lifespan)
    app.include_router(router)

    INDEX_AGE_SECONDS.set_function(
        lambda: index_state.bundle.manifest.age_seconds if index_state.bundle else 0.0
    )

    Instrumentator().instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)

    return app


app = create_app()
