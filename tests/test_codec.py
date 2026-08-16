import struct

import numpy as np
import pytest

from explained_faiss.codec import VectorFormatError, decode_vector, encode_vector, expected_bytes

DIM = 8


def test_round_trip_preserves_values():
    vector = np.arange(DIM, dtype=np.float32) / 3.0

    decoded = decode_vector(encode_vector(vector, DIM), DIM)

    assert np.array_equal(decoded, vector)
    assert decoded.dtype == np.float32


def test_layout_is_little_endian_float32():
    vector = np.array([1.0, -2.5] + [0.0] * (DIM - 2), dtype=np.float32)

    raw = encode_vector(vector, DIM)

    assert len(raw) == expected_bytes(DIM) == DIM * 4
    assert raw[:8] == struct.pack("<ff", 1.0, -2.5)


def test_decoded_vector_is_writable():
    # faiss.normalize_L2 mutates in place; a frombuffer view would raise.
    decoded = decode_vector(encode_vector(np.ones(DIM, dtype=np.float32), DIM), DIM)

    decoded[0] = 42.0

    assert decoded[0] == 42.0


@pytest.mark.parametrize("size", [0, DIM * 4 - 1, DIM * 4 + 1])
def test_wrong_length_is_rejected(size):
    with pytest.raises(VectorFormatError, match="bytes"):
        decode_vector(b"\x00" * size, DIM)


def test_encoding_wrong_shape_is_rejected():
    with pytest.raises(VectorFormatError, match="shape"):
        encode_vector(np.ones(DIM + 1, dtype=np.float32), DIM)
