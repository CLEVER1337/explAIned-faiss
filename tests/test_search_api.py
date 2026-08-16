import numpy as np
import pytest
from fastapi.testclient import TestClient
from redis.exceptions import ConnectionError as RedisConnectionError

from explained_faiss.config import Settings
from explained_faiss.index import IndexBundle
from explained_faiss.manifest import write_manifest
from explained_faiss.service.app import create_app


class FakeStore:
    """Stands in for EmbeddingStore: the service reads the user vector itself, so /search never
    receives one over the wire (CONTRACT.md)."""

    def __init__(self, embeddings: dict[str, np.ndarray] | None = None) -> None:
        self.embeddings = embeddings or {}
        self.faulted = False

    async def get_user_embedding(self, user_id: str):
        if self.faulted:
            raise RedisConnectionError("redis is down")
        return self.embeddings.get(user_id)

    async def close(self) -> None:
        return None


@pytest.fixture
def settings(tmp_path, dim) -> Settings:
    return Settings(
        index_dir=tmp_path,
        embedding_dim=dim,
        default_top_k=5,
        reload_watch_seconds=0,
        redis_url="redis://localhost:6379/0",
    )


def write_index(index_dir, ids, vectors, dim) -> None:
    bundle = IndexBundle.build(ids, vectors, dim)
    bundle.save(index_dir / bundle.manifest.version)
    write_manifest(index_dir, bundle.manifest)


def test_search_returns_ids_and_scores(settings, corpus, dim):
    ids, vectors = corpus
    write_index(settings.index_dir, ids, vectors, dim)
    store = FakeStore({"u1": vectors[2]})

    with TestClient(create_app(settings, store)) as client:
        response = client.post("/search", json={"user_id": "u1", "top_k": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["ids"][0] == ids[2]
    assert len(body["ids"]) == len(body["scores"]) == 3


def test_top_k_is_clamped_to_default_top_k(settings, corpus, dim):
    ids, vectors = corpus
    write_index(settings.index_dir, ids, vectors, dim)
    store = FakeStore({"u1": vectors[0]})

    with TestClient(create_app(settings, store)) as client:
        response = client.post("/search", json={"user_id": "u1", "top_k": 1000})

    assert len(response.json()["ids"]) == settings.default_top_k


def test_missing_user_embedding_is_204_not_an_error(settings, corpus, dim):
    ids, vectors = corpus
    write_index(settings.index_dir, ids, vectors, dim)

    with TestClient(create_app(settings, FakeStore())) as client:
        response = client.post("/search", json={"user_id": "cold-start", "top_k": 10})

    assert response.status_code == 204
    assert not response.content


def test_missing_index_is_503(settings, corpus):
    _, vectors = corpus

    with TestClient(create_app(settings, FakeStore({"u1": vectors[0]}))) as client:
        response = client.post("/search", json={"user_id": "u1", "top_k": 10})

    assert response.status_code == 503


def test_redis_failure_is_503(settings, corpus, dim):
    ids, vectors = corpus
    write_index(settings.index_dir, ids, vectors, dim)
    store = FakeStore({"u1": vectors[0]})
    store.faulted = True

    with TestClient(create_app(settings, store)) as client:
        response = client.post("/search", json={"user_id": "u1", "top_k": 10})

    assert response.status_code == 503


def test_invalid_payload_is_422(settings):
    with TestClient(create_app(settings, FakeStore())) as client:
        assert client.post("/search", json={"top_k": 10}).status_code == 422
        assert client.post("/search", json={"user_id": "", "top_k": 10}).status_code == 422
        assert client.post("/search", json={"user_id": "u", "top_k": 0}).status_code == 422


def test_health_is_200_even_without_an_index(settings):
    with TestClient(create_app(settings, FakeStore())) as client:
        body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["index_loaded"] is False
    assert body["vectors"] == 0


def test_reload_picks_up_an_index_built_after_startup(settings, corpus, dim):
    ids, vectors = corpus

    with TestClient(create_app(settings, FakeStore({"u1": vectors[1]}))) as client:
        assert client.post("/search", json={"user_id": "u1", "top_k": 3}).status_code == 503

        write_index(settings.index_dir, ids, vectors, dim)
        reloaded = client.post("/reload")

        assert reloaded.status_code == 200
        assert reloaded.json()["result"] == "loaded"
        assert reloaded.json()["vectors"] == len(ids)
        assert client.post("/search", json={"user_id": "u1", "top_k": 3}).json()["ids"][0] == ids[1]


def test_reload_without_an_artifact_is_503(settings):
    with TestClient(create_app(settings, FakeStore())) as client:
        response = client.post("/reload")

    assert response.status_code == 503
    assert response.json()["result"] == "missing"


def test_metrics_exposes_the_index_gauges(settings, corpus, dim):
    ids, vectors = corpus
    write_index(settings.index_dir, ids, vectors, dim)

    with TestClient(create_app(settings, FakeStore())) as client:
        body = client.get("/metrics").text

    assert "faiss_index_vectors" in body
    assert "faiss_index_age_seconds" in body
