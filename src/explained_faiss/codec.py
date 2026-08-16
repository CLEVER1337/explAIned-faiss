"""Redis vector format.

`rec:article_embedding:{aid}` and `rec:user_embedding:{uid}` hold a vector as raw
little-endian float32 — `embedding_dim * 4` bytes, no header, no separators. Fixed here
because the index builder and the future `embeddings.py` (explAIned-ml) both depend on it;
see CONTRACT.md. The C# side never reads these keys.
"""

import numpy as np

DTYPE = np.dtype("<f4")


class VectorFormatError(ValueError):
    """A Redis value does not hold `dim` little-endian float32."""


def expected_bytes(dim: int) -> int:
    return dim * DTYPE.itemsize


def encode_vector(vector: np.ndarray, dim: int) -> bytes:
    if vector.shape != (dim,):
        raise VectorFormatError(f"expected shape ({dim},), got {vector.shape}")

    return np.ascontiguousarray(vector, dtype=DTYPE).tobytes()


def decode_vector(raw: bytes, dim: int) -> np.ndarray:
    want = expected_bytes(dim)
    if len(raw) != want:
        raise VectorFormatError(f"expected {want} bytes ({dim} little-endian float32), got {len(raw)}")

    # frombuffer gives a read-only view over `raw`; faiss needs a writable, owned array.
    return np.frombuffer(raw, dtype=DTYPE).astype(np.float32, copy=True)
