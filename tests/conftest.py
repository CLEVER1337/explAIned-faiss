import numpy as np
import pytest

DIM = 8


@pytest.fixture
def dim() -> int:
    return DIM


@pytest.fixture
def corpus(dim):
    """Eight orthogonal unit vectors, so the nearest neighbour of a vector is unambiguously itself."""
    ids = [f"article-{i}" for i in range(dim)]
    vectors = np.eye(dim, dtype=np.float32)
    return ids, vectors
