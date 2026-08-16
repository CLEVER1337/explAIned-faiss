"""Holds the index the service is currently serving.

The bundle is immutable, so a reload is a reference swap: in-flight searches keep the old object
alive until they finish and never see a half-loaded index. The lock serialises loaders, not readers.
"""

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path

from ..index import IndexBundle, IndexLoadError, load_current
from ..manifest import Manifest, read_manifest
from ..metrics import INDEX_VECTORS, RELOAD_TOTAL

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReloadResult:
    result: str  # loaded | unchanged | missing | failed
    manifest: Manifest | None = None
    error: str | None = None


class IndexState:
    def __init__(self, index_dir: Path) -> None:
        self._index_dir = index_dir
        self._bundle: IndexBundle | None = None
        self._lock = asyncio.Lock()

    @property
    def bundle(self) -> IndexBundle | None:
        return self._bundle

    @property
    def loaded_version(self) -> str | None:
        return self._bundle.manifest.version if self._bundle else None

    async def reload(self, force: bool = False) -> ReloadResult:
        """`force=False` is the watcher's path: it reloads only when the manifest names a new version."""
        async with self._lock:
            manifest = await asyncio.to_thread(read_manifest, self._index_dir)
            if manifest is None:
                RELOAD_TOTAL.labels(result="missing").inc()
                return ReloadResult("missing", error=f"no manifest in {self._index_dir}")

            if not force and manifest.version == self.loaded_version:
                RELOAD_TOTAL.labels(result="unchanged").inc()
                return ReloadResult("unchanged", manifest)

            try:
                bundle = await asyncio.to_thread(load_current, self._index_dir)
            except IndexLoadError as exc:
                # Keep serving the previous index: a broken artifact is better than no answers.
                logger.error("index reload failed: %s", exc)
                RELOAD_TOTAL.labels(result="failed").inc()
                return ReloadResult("failed", error=str(exc))

            if bundle is None:
                RELOAD_TOTAL.labels(result="missing").inc()
                return ReloadResult("missing", error=f"no manifest in {self._index_dir}")

            self._bundle = bundle
            INDEX_VECTORS.set(bundle.size)
            RELOAD_TOTAL.labels(result="loaded").inc()
            logger.info("loaded index %s: %d vectors", bundle.manifest.version, bundle.size)
            return ReloadResult("loaded", bundle.manifest)

    def describe(self) -> dict:
        bundle = self._bundle
        if bundle is None:
            return {"index_loaded": False, "vectors": 0, "built_at": None, "dim": None, "version": None}

        return {
            "index_loaded": True,
            "vectors": bundle.size,
            "built_at": bundle.manifest.built_at,
            "dim": bundle.dim,
            "version": bundle.manifest.version,
        }


async def watch_manifest(state: IndexState, interval_seconds: float) -> None:
    """Background reload loop. `interval_seconds <= 0` means the index only moves on POST /reload."""
    if interval_seconds <= 0:
        return

    while True:
        await asyncio.sleep(interval_seconds)
        try:
            await state.reload()
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - the watcher must outlive any single failure
            logger.exception("manifest watcher iteration failed")
