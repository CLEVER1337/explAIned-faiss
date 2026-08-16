import json

import numpy as np
import pytest

from explained_faiss.index import IndexBundle, IndexLoadError, load_current
from explained_faiss.manifest import write_manifest


def save_bundle(index_dir, bundle: IndexBundle) -> IndexBundle:
    bundle.save(index_dir / bundle.manifest.version)
    write_manifest(index_dir, bundle.manifest)
    return bundle


def test_build_records_the_corpus(corpus, dim):
    ids, vectors = corpus

    bundle = IndexBundle.build(ids, vectors, dim)

    assert bundle.size == len(ids)
    assert bundle.dim == dim
    assert bundle.manifest.count == len(ids)
    assert bundle.manifest.metric == "ip"


def test_search_returns_the_query_itself_first(corpus, dim):
    ids, vectors = corpus
    bundle = IndexBundle.build(ids, vectors, dim)

    hits, scores = bundle.search(vectors[3], top_k=3)

    assert hits[0] == ids[3]
    assert scores[0] == pytest.approx(1.0, abs=1e-5)
    assert scores == sorted(scores, reverse=True)


def test_search_is_cosine_so_vector_length_does_not_matter(corpus, dim):
    ids, vectors = corpus
    bundle = IndexBundle.build(ids, vectors, dim)

    small, _ = bundle.search(vectors[2] * 0.001, top_k=1)
    large, _ = bundle.search(vectors[2] * 1000.0, top_k=1)

    assert small == large == [ids[2]]


def test_top_k_larger_than_the_corpus_returns_everything(corpus, dim):
    ids, vectors = corpus
    bundle = IndexBundle.build(ids, vectors, dim)

    hits, scores = bundle.search(vectors[0], top_k=500)

    assert len(hits) == len(ids)
    assert len(scores) == len(ids)


def test_build_does_not_mutate_the_caller_vectors(corpus, dim):
    ids, vectors = corpus
    unnormalised = vectors * 5.0
    original = unnormalised.copy()

    IndexBundle.build(ids, unnormalised, dim)

    assert np.array_equal(unnormalised, original)


def test_save_load_round_trip(tmp_path, corpus, dim):
    ids, vectors = corpus
    save_bundle(tmp_path, IndexBundle.build(ids, vectors, dim))

    loaded = load_current(tmp_path)

    assert loaded is not None
    assert loaded.size == len(ids)
    assert loaded.search(vectors[5], top_k=1)[0] == [ids[5]]


def test_load_current_without_a_manifest_is_none(tmp_path):
    assert load_current(tmp_path) is None


def test_incomplete_artifact_is_rejected(tmp_path, corpus, dim):
    ids, vectors = corpus
    bundle = save_bundle(tmp_path, IndexBundle.build(ids, vectors, dim))
    (tmp_path / bundle.manifest.version / "ids.json").unlink()

    with pytest.raises(IndexLoadError, match="incomplete"):
        load_current(tmp_path)


def test_id_count_mismatch_is_rejected(tmp_path, corpus, dim):
    ids, vectors = corpus
    bundle = save_bundle(tmp_path, IndexBundle.build(ids, vectors, dim))
    ids_path = tmp_path / bundle.manifest.version / "ids.json"
    ids_path.write_text(json.dumps(ids[:-1]), encoding="utf-8")

    with pytest.raises(IndexLoadError, match="ids.json"):
        load_current(tmp_path)


def test_wrong_dimension_is_rejected(dim):
    with pytest.raises(IndexLoadError, match="matrix"):
        IndexBundle.build(["a"], np.ones((1, dim + 1), dtype=np.float32), dim)
