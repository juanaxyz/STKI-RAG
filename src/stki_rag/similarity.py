"""Similarity metrics for vector search."""

import numpy as np
from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Compute cosine similarity between two vectors.

    Args:
        a: First vector.
        b: Second vector.

    Returns:
        Cosine similarity in [-1, 1].
    """
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def softmax(x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert vector to probability distribution via softmax.

    Args:
        x: Input vector.

    Returns:
        Probability distribution (sums to 1).
    """
    x = np.array(x, dtype=np.float64)
    x = x - np.max(x)  # numerical stability
    e_x = np.exp(x)
    return e_x / e_x.sum()


def kl_divergence(p: NDArray[np.float64], q: NDArray[np.float64], eps: float = 1e-10) -> float:
    """Compute KL divergence D_KL(P || Q).

    Args:
        p: Probability distribution P.
        q: Probability distribution Q.
        eps: Small constant for numerical stability.

    Returns:
        KL divergence (non-negative, smaller = more similar).
    """
    p, q = np.array(p) + eps, np.array(q) + eps
    return float(np.sum(p * np.log(p / q)))