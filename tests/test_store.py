import numpy as np
import pytest

from explained_faiss.codec import encode_vector
from explained_faiss.store import ARTICLE_KEY_PREFIX, USER_KEY_PREFIX, EmbeddingStore

DIM = 8


class FakeRedis:
    """Only the four calls EmbeddingStore makes."""

    def __init__(self, data: dict[str, bytes]) -> None:
        self.data = {key.encode(): value for key, value in data.items()}
        self.scanned_with_keys_command = False

    async def get(self, key):
        return self.data.get(key.encode() if isinstance(key, str) else key)

    async def scan_iter(self, match: str, count: int = 10):
        prefix = match.rstrip("*").encode()
        for key in list(self.data):
            if key.startswith(prefix):
                yield key

    async def mget(self, keys):
        return [self.data.get(key) for key in keys]

    async def aclose(self):
        return None


def store_over(data: dict[str, bytes]) -> EmbeddingStore:
    return EmbeddingStore("redis://unused", DIM, client=FakeRedis(data))


def a_vector(seed: int) -> np.ndarray:
    return np.full(DIM, float(seed), dtype=np.float32)


async def test_user_embedding_is_decoded():
    vector = a_vector(3)
    store = store_over({f"{USER_KEY_PREFIX}u1": encode_vector(vector, DIM)})

    assert np.array_equal(await store.get_user_embedding("u1"), vector)


async def test_absent_user_embedding_is_none():
    assert await store_over({}).get_user_embedding("nobody") is None


async def test_malformed_user_embedding_is_treated_as_absent():
    store = store_over({f"{USER_KEY_PREFIX}u1": b"\x00\x01\x02"})

    assert await store.get_user_embedding("u1") is None


async def test_iterating_article_embeddings_skips_malformed_values():
    store = store_over(
        {
            f"{ARTICLE_KEY_PREFIX}a": encode_vector(a_vector(1), DIM),
            f"{ARTICLE_KEY_PREFIX}b": b"too short",
            f"{ARTICLE_KEY_PREFIX}c": encode_vector(a_vector(2), DIM),
            f"{USER_KEY_PREFIX}u1": encode_vector(a_vector(3), DIM),
        }
    )

    rows = [article_id async for article_id, _ in store.iter_article_embeddings(batch_size=2)]

    assert sorted(rows) == ["a", "c"]


@pytest.mark.parametrize("batch_size", [1, 2, 100])
async def test_batching_does_not_lose_rows(batch_size):
    data = {f"{ARTICLE_KEY_PREFIX}a{i}": encode_vector(a_vector(i), DIM) for i in range(7)}

    rows = [article_id async for article_id, _ in store_over(data).iter_article_embeddings(batch_size)]

    assert len(rows) == 7
