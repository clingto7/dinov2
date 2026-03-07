#!/usr/bin/env python3
"""
Diversity metrics for DINOv2 feature embeddings.

Provides functions to compute pairwise similarity/distance and aggregate
diversity scores from a set of feature vectors.
"""

import torch
import torch.nn.functional as F


def pairwise_cosine_matrix(x: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity matrix.

    Args:
        x: Feature matrix of shape [N, D].

    Returns:
        Cosine similarity matrix of shape [N, N], values in [-1, 1].
    """
    x_norm = F.normalize(x, p=2, dim=-1)  # [N, D]
    return x_norm @ x_norm.T  # [N, N]


def upper_triangle_values(matrix: torch.Tensor) -> torch.Tensor:
    """Extract upper-triangle values (excluding diagonal) from a square matrix.

    This avoids duplicate pairs (A-B == B-A) and self-comparisons (A-A).

    Args:
        matrix: Square matrix of shape [N, N].

    Returns:
        1-D tensor of upper-triangle values, length N*(N-1)/2.
    """
    mask = torch.triu(torch.ones_like(matrix, dtype=torch.bool), diagonal=1)
    return matrix[mask]


def mean_pairwise_cosine_distance(x: torch.Tensor) -> float:
    """Mean pairwise cosine distance (1 - cosine_similarity).

    Higher value → more diverse features.

    Args:
        x: Feature matrix of shape [N, D] with N >= 2.

    Returns:
        Scalar mean cosine distance.
    """
    if x.shape[0] < 2:
        return 0.0
    sim = pairwise_cosine_matrix(x)
    tri_values = upper_triangle_values(sim)
    return (1.0 - tri_values.mean()).item()


def mean_pairwise_l2_distance(x: torch.Tensor) -> float:
    """Mean pairwise L2 (Euclidean) distance.

    Uses torch.cdist for efficient computation.

    Args:
        x: Feature matrix of shape [N, D] with N >= 2.

    Returns:
        Scalar mean L2 distance.
    """
    if x.shape[0] < 2:
        return 0.0
    # cdist requires 3D input: [batch, N, D]
    dist = torch.cdist(x.unsqueeze(0), x.unsqueeze(0)).squeeze(0)  # [N, N]
    tri_values = upper_triangle_values(dist)
    return tri_values.mean().item()


def feature_variance_score(x: torch.Tensor) -> float:
    """Feature variance score: mean of per-dimension variances.

    Measures how spread out the feature vectors are in embedding space.

    Args:
        x: Feature matrix of shape [N, D].

    Returns:
        Scalar variance score.
    """
    if x.shape[0] < 2:
        return 0.0
    return x.var(dim=0).mean().item()


def compute_all_metrics(x: torch.Tensor) -> dict[str, float]:
    """Compute all diversity metrics for a set of features.

    Args:
        x: Feature matrix of shape [N, D].

    Returns:
        Dict with keys: n_samples, cosine_distance, l2_distance, variance_score.
    """
    return {
        "n_samples": x.shape[0],
        "cosine_distance": mean_pairwise_cosine_distance(x),
        "l2_distance": mean_pairwise_l2_distance(x),
        "variance_score": feature_variance_score(x),
    }
