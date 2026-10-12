from __future__ import annotations

"""Noise/secret samplers for toy Ring-LWE experiments."""

import random
from typing import List

IntVector = List[int]


def sample_uniform_poly(n: int, q: int, rng: random.Random) -> IntVector:
    return [rng.randrange(q) for _ in range(n)]


def sample_binary_poly(n: int, q: int, rng: random.Random) -> IntVector:
    return [rng.randint(0, 1) % q for _ in range(n)]


def sample_ternary_poly(n: int, q: int, rng: random.Random) -> IntVector:
    return [rng.choice([-1, 0, 1]) % q for _ in range(n)]


def sample_discrete_gaussian_poly(n: int, q: int, sigma: float, rng: random.Random) -> IntVector:
    """
    Educational approximation by rounding normal samples.
    """
    return [int(round(rng.gauss(0.0, sigma))) % q for _ in range(n)]


def sample_small_poly(n: int, q: int, sigma: float, rng: random.Random, mode: str = "gaussian") -> IntVector:
    if mode == "gaussian":
        return sample_discrete_gaussian_poly(n, q, sigma, rng)
    if mode == "binary":
        return sample_binary_poly(n, q, rng)
    if mode == "ternary":
        return sample_ternary_poly(n, q, rng)
    raise ValueError("mode must be one of: gaussian, binary, ternary")
