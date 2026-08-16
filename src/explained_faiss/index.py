"""Index construction, persistence and search.

`IndexFlatIP` over L2-normalised vectors, so inner product *is* cosine similarity. Exact search,
nothing to train — the corpus is small, and an approximate index (IVF/HNSW) would only add
parameters to tune. Swapping it later is confined to `build()` and `load()`.

Article ids are GUIDs, so `IndexIDMap2` (int64 ids) buys nothing: position `i` in the index is
position `i` in `ids.json`, and the two files are written and loaded as one unit.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import faiss
import numpy as np

from .manifest import Manifest, new_version, read_manifest

INDEX_FILE = "index.faiss"
IDS_FILE = "ids.json"


class IndexLoadError(RuntimeError):
    """The artifact named by the manifest is missing or inconsistent."""


def _as_matrix(vectors: np.ndarray, dim: int) -> np.ndarray:
    # Always a copy: normalize_L2 works in place, and ascontiguousarray would hand back the
    # caller's own array whenever it is already float32/C-contiguous.
    matrix = np.array(vectors, dtype=np.float32, order="C", copy=True)
    if matrix.ndim != 2 or matrix.shape[1] != dim:
        raise IndexLoadError(f"expected an (n, {dim}) matrix, got {matrix.shape}")

    faiss.normalize_L2(matrix)
    return matrix


class IndexBundle:
    """A loaded index plus the id list and the manifest it came from. Immutable once built."""

    def __init__(self, index: faiss.Index, ids: list[str], manifest: Manifest) -> None:
        if index.ntotal != len(ids):
            raise IndexLoadError(f"index holds {index.ntotal} vectors but ids.json has {len(ids)}")

        self._index = index
        self._ids = ids
        self.manifest = manifest

    @property
    def size(self) -> int:
        return len(self._ids)

    @property
    def dim(self) -> int:
        return self._index.d

    @classmethod
    def build(cls, ids: list[str], vectors: np.ndarray, dim: int) -> "IndexBundle":
        matrix = _as_matrix(vectors, dim)
        if len(ids) != matrix.shape[0]:
            raise IndexLoadError(f"{len(ids)} ids for {matrix.shape[0]} vectors")

        index = faiss.IndexFlatIP(dim)
        index.add(matrix)

        manifest = Manifest(
            version=new_version(),
            built_at=datetime.now(UTC).isoformat(),
            dim=dim,
            count=len(ids),
            metric="ip",
            faiss_version=faiss.__version__,
        )

        return cls(index, list(ids), manifest)

    def save(self, version_dir: Path) -> Path:
        version_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(version_dir / INDEX_FILE))
        (version_dir / IDS_FILE).write_text(json.dumps(self._ids, ensure_ascii=False), encoding="utf-8")
        return version_dir

    @classmethod
    def load(cls, version_dir: Path, manifest: Manifest) -> "IndexBundle":
        index_path = version_dir / INDEX_FILE
        ids_path = version_dir / IDS_FILE
        if not index_path.exists() or not ids_path.exists():
            raise IndexLoadError(f"incomplete artifact in {version_dir}")

        try:
            index = faiss.read_index(str(index_path))
            ids = json.loads(ids_path.read_text(encoding="utf-8"))
        except (OSError, RuntimeError, json.JSONDecodeError) as exc:
            raise IndexLoadError(f"cannot read artifact in {version_dir}: {exc}") from exc

        if not isinstance(ids, list):
            raise IndexLoadError(f"{ids_path} does not hold a list")

        return cls(index, [str(item) for item in ids], manifest)

    def search(self, query: np.ndarray, top_k: int) -> tuple[list[str], list[float]]:
        if self.size == 0:
            return [], []

        vector = _as_matrix(query.reshape(1, -1), self.dim)
        scores, positions = self._index.search(vector, min(max(top_k, 1), self.size))

        ids: list[str] = []
        hits: list[float] = []
        for position, score in zip(positions[0], scores[0], strict=True):
            # faiss pads with -1 when fewer than k neighbours exist.
            if position < 0:
                continue
            ids.append(self._ids[int(position)])
            hits.append(float(score))

        return ids, hits


def load_current(index_dir: Path) -> IndexBundle | None:
    """Loads the version the manifest points at, or None when there is no manifest."""
    manifest = read_manifest(index_dir)
    if manifest is None:
        return None

    return IndexBundle.load(index_dir / manifest.version, manifest)
