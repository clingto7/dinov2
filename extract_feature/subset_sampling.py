#!/usr/bin/env python3
"""
Subset sampling strategies for diversity analysis.

Generates index lists that partition or sample from a collection of items
(images or videos) into multiple subsets for diversity comparison.
"""

import random


def generate_subset_indices(
    n_items: int,
    subset_size: int,
    n_subsets: int,
    seed: int = 42,
    strategy: str = "random",
) -> list[list[int]]:
    """Generate multiple subsets of indices from a collection.

    Args:
        n_items: Total number of items in the collection.
        subset_size: Number of items per subset.
        n_subsets: Number of subsets to generate.
        seed: Random seed for reproducibility.
        strategy: Sampling strategy, one of:
            - "random": Random sampling without replacement within each subset.
            - "sliding": Sliding window with automatic step size.

    Returns:
        List of index lists, each of length <= subset_size.

    Raises:
        ValueError: If subset_size > n_items or invalid strategy.
    """
    if subset_size > n_items:
        raise ValueError(f"subset_size ({subset_size}) > n_items ({n_items})")
    if subset_size < 1 or n_subsets < 1:
        raise ValueError(f"subset_size and n_subsets must be >= 1")

    if strategy == "random":
        return _random_subsets(n_items, subset_size, n_subsets, seed)
    elif strategy == "sliding":
        return _sliding_subsets(n_items, subset_size, n_subsets)
    else:
        raise ValueError(
            f"Unknown strategy: {strategy!r}. Choose 'random' or 'sliding'."
        )


def _random_subsets(
    n_items: int, subset_size: int, n_subsets: int, seed: int
) -> list[list[int]]:
    """Generate random subsets (sampling without replacement per subset)."""
    rng = random.Random(seed)
    all_indices = list(range(n_items))
    subsets = []
    for _ in range(n_subsets):
        chosen = rng.sample(all_indices, subset_size)
        chosen.sort()  # deterministic ordering within subset
        subsets.append(chosen)
    return subsets


def _sliding_subsets(n_items: int, subset_size: int, n_subsets: int) -> list[list[int]]:
    """Generate sliding window subsets with automatic step.

    If n_subsets would exceed the number of possible windows, we clamp.
    """
    max_start = n_items - subset_size
    if max_start <= 0:
        # Only one possible window
        return [list(range(n_items))]

    if n_subsets == 1:
        return [list(range(subset_size))]

    # Evenly space start positions
    step = max_start / (n_subsets - 1)
    subsets = []
    for i in range(n_subsets):
        start = int(round(i * step))
        start = min(start, max_start)  # clamp
        subsets.append(list(range(start, start + subset_size)))
    return subsets
