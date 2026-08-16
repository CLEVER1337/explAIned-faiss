"""DEV FIXTURE — this is not ML.

Nothing writes `rec:article_embedding:*` yet (`embeddings.py` belongs to explAIned-ml and is not
built), so there is no way to exercise the index builder or /search against a live Redis. This
script fills those keys with *deterministic pseudo-random* vectors derived from the article id:
same id → same vector, so results are reproducible and reviewable by eye.

Delete it as soon as the real Sentence-BERT job exists. Do not let anything depend on it.

    python -m explained_faiss.scripts.seed_dev_embeddings --user <uid from the JWT sub claim>
"""

import argparse
import asyncio
import hashlib
import logging
import sys

import httpx
import numpy as np
from redis.asyncio import Redis

from ..codec import encode_vector
from ..config import get_settings
from ..store import ARTICLE_KEY_PREFIX, USER_KEY_PREFIX

logger = logging.getLogger("explained_faiss.seed")

PAGE_SIZE = 100


def fake_embedding(article_id: str, dim: int) -> np.ndarray:
    seed = int.from_bytes(hashlib.blake2b(article_id.encode("utf-8"), digest_size=8).digest(), "little")
    vector = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return vector / np.linalg.norm(vector)


async def fetch_article_ids(base_url: str, max_articles: int) -> list[str]:
    """`GET /articles/recent` is the only way to enumerate the corpus over HTTP — there is no
    "all ids" endpoint, and it only returns Published + Public articles."""
    ids: list[str] = []

    async with httpx.AsyncClient(base_url=base_url, timeout=10.0) as client:
        offset = 0
        while len(ids) < max_articles:
            response = await client.get(
                "/articles/recent", params={"limit": PAGE_SIZE, "offset": offset}
            )
            response.raise_for_status()
            page = response.json()
            if not page:
                break

            ids.extend(str(item["id"]) for item in page)
            offset += PAGE_SIZE

    return ids[:max_articles]


async def seed(base_url: str, user_ids: list[str], max_articles: int, user_sample: int) -> int:
    settings = get_settings()
    dim = settings.embedding_dim

    article_ids = await fetch_article_ids(base_url, max_articles)
    if not article_ids:
        logger.error("article service returned no articles — is :5036 up and are there Published ones?")
        return 1

    redis = Redis.from_url(settings.redis_url)
    try:
        pipe = redis.pipeline(transaction=False)
        vectors = {}
        for article_id in article_ids:
            vector = fake_embedding(article_id, dim)
            vectors[article_id] = vector
            pipe.set(f"{ARTICLE_KEY_PREFIX}{article_id}", encode_vector(vector, dim))
        await pipe.execute()
        logger.info("wrote %d article embeddings", len(article_ids))

        for user_id in user_ids:
            sample = article_ids[:user_sample]
            centroid = np.mean([vectors[aid] for aid in sample], axis=0).astype(np.float32)
            centroid /= np.linalg.norm(centroid)
            await redis.set(f"{USER_KEY_PREFIX}{user_id}", encode_vector(centroid, dim))
            logger.info("wrote user embedding for %s (centroid of %d articles)", user_id, len(sample))
    finally:
        await redis.aclose()

    return 0


def cli() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--articles-url", default="http://localhost:5036")
    parser.add_argument("--user", action="append", default=[], help="user id to seed (repeatable)")
    parser.add_argument("--max-articles", type=int, default=1000)
    parser.add_argument("--user-sample", type=int, default=5, help="articles averaged into a user vector")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")

    return asyncio.run(seed(args.articles_url, args.user, args.max_articles, args.user_sample))


if __name__ == "__main__":
    sys.exit(cli())
