from __future__ import annotations

"""Polynomial arithmetic for R_q = Z_q[X] / (X^n + 1)."""

from typing import List, Sequence

IntVector = List[int]


def _mod_q(x: int, q: int) -> int:
    return x % q


def _check_len(a: Sequence[int], n: int) -> None:
    if len(a) != n:
        raise ValueError(f"expected polynomial length n={n}, got {len(a)}")


def poly_add(a: Sequence[int], b: Sequence[int], q: int) -> IntVector:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return [(_mod_q(x + y, q)) for x, y in zip(a, b)]


def poly_sub(a: Sequence[int], b: Sequence[int], q: int) -> IntVector:
    if len(a) != len(b):
        raise ValueError("length mismatch")
    return [(_mod_q(x - y, q)) for x, y in zip(a, b)]


def poly_scalar_mul(a: Sequence[int], k: int, q: int) -> IntVector:
    return [(_mod_q(k * x, q)) for x in a]


def poly_mul_naive_negacyclic(a: Sequence[int], b: Sequence[int], q: int, n: int | None = None) -> IntVector:
    """
    Schoolbook multiplication in Z_q[X]/(X^n + 1): X^n = -1.
    """
    if n is None:
        if len(a) != len(b):
            raise ValueError("length mismatch")
        n = len(a)

    _check_len(a, n)
    _check_len(b, n)

    out = [0] * n
    for i in range(n):
        ai = a[i]
        for j in range(n):
            idx = i + j
            v = ai * b[j]
            if idx >= n:
                out[idx - n] -= v  # X^n = -1 fold
            else:
                out[idx] += v

    return [x % q for x in out]


def poly_centered(a: Sequence[int], q: int) -> IntVector:
    out = []
    for x in a:
        r = x % q
        if r > q // 2:
            r -= q
        out.append(r)
    return out


def zero_poly(n: int) -> IntVector:
    return [0] * n
