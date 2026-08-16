"""Builds the FAISS artifact from `rec:article_embedding:*`.

Lives next to the service on purpose: the server loads a binary format this job writes, so the
faiss version, the index type and its parameters have to move together (todo.md §2).

    python -m explained_faiss.jobs.build_index

Exit codes: 0 built, 1 nothing to index (the existing manifest is left alone), 2 Redis failed.
"""

import argparse
import asyncio
import logging
import shutil
import sys
from pathlib import Path

import numpy as np
from redis.exceptions import RedisError

from ..config import Settings, get_settings
from ..index import IndexBundle
from ..manifest import MANIFEST_NAME, read_manifest, write_manifest
from ..store import EmbeddingStore

logger = logging.getLogger("explained_faiss.build_index")


async def build(settings: Settings, keep: int) -> int:
    store = EmbeddingStore(settings.redis_url, settings.embedding_dim)

    ids: list[str] = []
    vectors: list[np.ndarray] = []
    try:
        async for article_id, vector in store.iter_article_embeddings():
            ids.append(article_id)
            vectors.append(vector)
    except RedisError as exc:
        logger.error("redis failed while reading embeddings: %s", exc)
        return 2
    finally:
        await store.close()

    if not ids:
        current = read_manifest(settings.index_dir)
        logger.error(
            "no embeddings under rec:article_embedding:* — keeping the current manifest (%s)",
            current.version if current else "none",
        )
        return 1

    bundle = IndexBundle.build(ids, np.vstack(vectors), settings.embedding_dim)
    version_dir = settings.index_dir / bundle.manifest.version
    bundle.save(version_dir)
    write_manifest(settings.index_dir, bundle.manifest)

    logger.info("built %s: %d vectors, dim %d", bundle.manifest.version, bundle.size, bundle.dim)

    _prune(settings.index_dir, keep=keep, current=bundle.manifest.version)
    return 0


def _prune(index_dir: Path, keep: int, current: str) -> None:
    """Keeps the newest `keep` version directories. Version names sort chronologically."""
    if keep < 1:
        return

    versions = sorted(
        (path for path in index_dir.iterdir() if path.is_dir() and path.name != MANIFEST_NAME),
        key=lambda path: path.name,
        reverse=True,
    )

    for stale in versions[keep:]:
        if stale.name == current:
            continue
        logger.info("pruning %s", stale.name)
        shutil.rmtree(stale, ignore_errors=True)


def cli() -> int:
    parser = argparse.ArgumentParser(description="Build the FAISS index from Redis embeddings")
    parser.add_argument("--index-dir", type=Path, default=None, help="overrides INDEX_DIR")
    parser.add_argument("--keep", type=int, default=None, help="version directories to keep")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(name)s %(message)s")

    settings = get_settings()
    if args.index_dir is not None:
        settings = settings.model_copy(update={"index_dir": args.index_dir})

    return asyncio.run(build(settings, keep=args.keep if args.keep is not None else settings.keep_versions))


if __name__ == "__main__":
    sys.exit(cli())
