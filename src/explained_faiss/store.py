"""Redis access. This service is the only reader of the embedding keys.

`rec:` is the recommendation-loop prefix (see explAInedRecommendationService/CONTRACT.md);
`explAIned_` belongs to the .NET `IDistributedCache` path and must not be used here.
"""

import logging
from collections.abc import AsyncIterator

import numpy as np
from redis.asyncio import Redis

from .codec import VectorFormatError, decode_vector

ARTICLE_KEY_PREFIX = "rec:article_embedding:"
USER_KEY_PREFIX = "rec:user_embedding:"

logger = logging.getLogger(__name__)


class EmbeddingStore:
    def __init__(self, redis_url: str, dim: int, client: Redis | None = None) -> None:
        self._dim = dim
        self._redis = client if client is not None else Redis.from_url(redis_url)

    async def close(self) -> None:
        await self._redis.aclose()

    async def get_user_embedding(self, user_id: str) -> np.ndarray | None:
        """None means "this user has no embedding" — a normal state, not a failure.

        A malformed value is treated the same way: nothing useful can be searched with it.
        """
        raw = await self._redis.get(f"{USER_KEY_PREFIX}{user_id}")
        if raw is None:
            return None

        try:
            return decode_vector(raw, self._dim)
        except VectorFormatError:
            logger.warning("user %s has a malformed embedding, treating it as absent", user_id)
            return None

    async def iter_article_embeddings(self, batch_size: int = 500) -> AsyncIterator[tuple[str, np.ndarray]]:
        """SCAN + MGET over `rec:article_embedding:*`. Never KEYS — this runs against the live Redis."""
        keys: list[bytes] = []

        async for key in self._redis.scan_iter(match=f"{ARTICLE_KEY_PREFIX}*", count=batch_size):
            keys.append(key)
            if len(keys) >= batch_size:
                for item in await self._fetch(keys):
                    yield item
                keys = []

        if keys:
            for item in await self._fetch(keys):
                yield item

    async def _fetch(self, keys: list[bytes]) -> list[tuple[str, np.ndarray]]:
        values = await self._redis.mget(keys)

        rows: list[tuple[str, np.ndarray]] = []
        for key, raw in zip(keys, values, strict=True):
            if raw is None:
                continue  # expired between SCAN and MGET

            article_id = _article_id(key)
            try:
                rows.append((article_id, decode_vector(raw, self._dim)))
            except VectorFormatError as exc:
                logger.warning("skipping %s: %s", article_id, exc)

        return rows


def _article_id(key: bytes | str) -> str:
    text = key.decode("utf-8") if isinstance(key, bytes) else key
    return text.removeprefix(ARTICLE_KEY_PREFIX)
