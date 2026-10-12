from __future__ import annotations

"""
Member 3 scope: from-scratch lattice tooling for Ring-LWE/LWE security analysis.

Implements:
- LLL basis reduction (pure Python, no external lattice libraries)
- Babai nearest-plane CVP heuristic
- q-ary attack lattice construction for LWE instances
- helper routines to generate tiny toy-LWE instances and attempt secret recovery
"""

from dataclasses import dataclass
import math
import random
from typing import List, Sequence, Tuple

Vector = List[float]
IntVector = List[int]
Matrix = List[List[float]]
IntMatrix = List[List[int]]


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _sub(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x - y for x, y in zip(a, b)]


def _add(a: Sequence[float], b: Sequence[float]) -> Vector:
    return [x + y for x, y in zip(a, b)]


def _scale(a: Sequence[float], k: float) -> Vector:
    return [k * x for x in a]


def _transpose(m: Sequence[Sequence[float]]) -> Matrix:
    if not m:
        return []
    return [list(col) for col in zip(*m)]


def _center_mod(x: int, q: int) -> int:
    r = x % q
    if r > q // 2:
        r -= q
    return r


def gram_schmidt_rows(basis_rows: IntMatrix | Matrix) -> Tuple[Matrix, Matrix, List[float]]:
    """
    Gram-Schmidt on row basis vectors.

    Returns:
        b_star: orthogonalized vectors
        mu: GS coefficients mu[i][j]
        norms: ||b*_i||^2
    """
    n = len(basis_rows)
    if n == 0:
        return [], [], []

    d = len(basis_rows[0])
    b_star: Matrix = [[0.0] * d for _ in range(n)]
    mu: Matrix = [[0.0] * n for _ in range(n)]
    norms = [0.0] * n

    for i in range(n):
        v = [float(x) for x in basis_rows[i]]
        for j in range(i):
            if norms[j] == 0.0:
                mu[i][j] = 0.0
                continue
            mu[i][j] = _dot(basis_rows[i], b_star[j]) / norms[j]
            v = _sub(v, _scale(b_star[j], mu[i][j]))
        b_star[i] = v
        norms[i] = _dot(v, v)

    return b_star, mu, norms


def lll_reduce_rows(basis_rows: IntMatrix, delta: float = 0.99) -> IntMatrix:
    """
    Classic LLL reduction on an integer row basis.

    Args:
        basis_rows: list of row vectors (integer basis)
        delta: Lovasz parameter in (1/4, 1)

    Returns:
        Reduced integer row basis.
    """
    if not (0.25 < delta < 1.0):
        raise ValueError("delta must satisfy 1/4 < delta < 1")
    if not basis_rows:
        return []

    b = [row[:] for row in basis_rows]
    n = len(b)

    k = 1
    while k < n:
        # Size reduction
        _, mu, _ = gram_schmidt_rows(b)
        for j in range(k - 1, -1, -1):
            q = int(round(mu[k][j]))
            if q != 0:
                b[k] = [bk - q * bj for bk, bj in zip(b[k], b[j])]

        # Lovasz check
        _, mu, norms = gram_schmidt_rows(b)
        if norms[k - 1] == 0.0:
            k += 1
            continue

        lhs = norms[k]
        rhs = (delta - mu[k][k - 1] ** 2) * norms[k - 1]
        if lhs >= rhs:
            k += 1
        else:
            b[k], b[k - 1] = b[k - 1], b[k]
            k = max(k - 1, 1)

    return b


def lll_reduce_columns(column_basis: IntMatrix, delta: float = 0.99) -> IntMatrix:
    """
    LLL-reduce a column basis by transposing, reducing rows, and transposing back.
    """
    as_rows = _transpose(column_basis)
    reduced_rows = lll_reduce_rows([[int(round(x)) for x in row] for row in as_rows], delta=delta)
    reduced_cols = _transpose(reduced_rows)
    return [[int(round(x)) for x in row] for row in reduced_cols]


def gram_schmidt_columns(column_basis: IntMatrix | Matrix) -> Tuple[Matrix, Matrix, List[float]]:
    """
    Gram-Schmidt on column basis vectors.

    Returns:
        b_star_cols: orthogonalized columns
        mu[i][j] where i,j index columns
        norms of b_star columns
    """
    d = len(column_basis)
    if d == 0:
        return [], [], []
    n = len(column_basis[0])

    cols = [[float(column_basis[r][c]) for r in range(d)] for c in range(n)]

    b_star_cols: Matrix = [[0.0] * d for _ in range(n)]
    mu: Matrix = [[0.0] * n for _ in range(n)]
    norms = [0.0] * n

    for i in range(n):
        v = cols[i][:]
        for j in range(i):
            if norms[j] == 0.0:
                mu[i][j] = 0.0
                continue
            mu[i][j] = _dot(cols[i], b_star_cols[j]) / norms[j]
            v = _sub(v, _scale(b_star_cols[j], mu[i][j]))
        b_star_cols[i] = v
        norms[i] = _dot(v, v)

    return b_star_cols, mu, norms


def babai_nearest_plane_columns(column_basis: IntMatrix, target: Sequence[float]) -> Tuple[IntVector, IntVector]:
    """
    Babai nearest-plane using a *column* basis.

    Returns:
        nearest_vector in lattice (ambient dimension d)
        integer coefficient vector relative to column basis
    """
    d = len(column_basis)
    if d == 0:
        return [], []
    n = len(column_basis[0])
    if len(target) != d:
        raise ValueError("target dimension mismatch")

    cols = [[float(column_basis[r][c]) for r in range(d)] for c in range(n)]
    b_star, _, norms = gram_schmidt_columns(column_basis)

    y = [float(x) for x in target]
    coeffs = [0] * n

    for i in range(n - 1, -1, -1):
        if norms[i] == 0.0:
            coeff = 0
        else:
            c_real = _dot(y, b_star[i]) / norms[i]
            coeff = int(round(c_real))
        coeffs[i] = coeff
        y = _sub(y, _scale(cols[i], coeff))

    nearest = _sub(target, y)
    nearest_int = [int(round(x)) for x in nearest]
    return nearest_int, coeffs


def build_qary_attack_basis(A: IntMatrix, q: int) -> IntMatrix:
    """
    Build column basis for the q-ary lattice used in primal-style LWE CVP attacks.

    If A is m x n, returns B in Z^{(m+n) x (m+n)}:

        B = [ q I_m   A ]
            [  0     I_n ]

    Any lattice vector has the form [q z + A s ; s].
    """
    m = len(A)
    if m == 0:
        raise ValueError("A must be non-empty")
    n = len(A[0])

    d = m + n
    B = [[0 for _ in range(d)] for _ in range(d)]

    # q I_m
    for i in range(m):
        B[i][i] = q

    # top-right A
    for i in range(m):
        for j in range(n):
            B[i][m + j] = int(A[i][j])

    # bottom-right I_n
    for i in range(n):
        B[m + i][m + i] = 1

    return B


@dataclass
class LWEInstance:
    A: IntMatrix
    b: IntVector
    s: IntVector
    e: IntVector
    q: int


def sample_lwe_instance(
    n: int,
    m: int,
    q: int,
    sigma: float = 1.5,
    secret_distribution: str = "binary",
    seed: int | None = None,
) -> LWEInstance:
    """
    Create a tiny toy-LWE instance b = A s + e (mod q).

    Intended for educational/testing scales only.
    """
    rng = random.Random(seed)

    if secret_distribution == "binary":
        s = [rng.randint(0, 1) for _ in range(n)]
    elif secret_distribution == "ternary":
        s = [rng.choice([-1, 0, 1]) for _ in range(n)]
    else:
        raise ValueError("secret_distribution must be 'binary' or 'ternary'")

    A = [[rng.randrange(q) for _ in range(n)] for _ in range(m)]
    e = [int(round(rng.gauss(0.0, sigma))) for _ in range(m)]

    b = []
    for i in range(m):
        v = sum(A[i][j] * s[j] for j in range(n)) + e[i]
        b.append(v % q)

    return LWEInstance(A=A, b=b, s=s, e=e, q=q)


@dataclass
class AttackResult:
    recovered_secret_raw: IntVector
    recovered_secret_binary: IntVector
    success_exact_binary: bool


def recover_binary_secret_with_lll(
    A: IntMatrix,
    b: IntVector,
    q: int,
    delta: float = 0.99,
) -> AttackResult:
    """
    Attempt binary secret recovery via:
      1) q-ary lattice construction
      2) LLL reduction
      3) Babai nearest-plane to target [b_centered ; 0]

    This is intentionally for small educational parameter sets.
    """
    m = len(A)
    if m == 0:
        raise ValueError("A must be non-empty")
    n = len(A[0])
    if len(b) != m:
        raise ValueError("b length must equal number of rows of A")

    B = build_qary_attack_basis(A, q)
    B_red = lll_reduce_columns(B, delta=delta)

    b_center = [_center_mod(x, q) for x in b]
    target = [float(x) for x in (b_center + [0] * n)]

    nearest, _ = babai_nearest_plane_columns(B_red, target)

    # In the original basis, tail coordinates encode s exactly;
    # after reduction/Babai they are approximate. We map to binary by nearest in {0,1}.
    raw_tail = nearest[m:]
    raw_centered = [_center_mod(x, q) for x in raw_tail]

    s_bin = []
    for x in raw_centered:
        d0 = abs(x - 0)
        d1 = abs(x - 1)
        s_bin.append(0 if d0 <= d1 else 1)

    return AttackResult(
        recovered_secret_raw=raw_centered,
        recovered_secret_binary=s_bin,
        success_exact_binary=False,  # caller can compare against known secret
    )


if __name__ == "__main__":
    # quick smoke demo for tiny dimensions
    n = 8
    m = 2 * n
    q = 257

    inst = sample_lwe_instance(n=n, m=m, q=q, sigma=1.2, secret_distribution="binary", seed=7)
    res = recover_binary_secret_with_lll(inst.A, inst.b, q=q)

    ok = res.recovered_secret_binary == inst.s
    print("secret:", inst.s)
    print("guess :", res.recovered_secret_binary)
    print("exact success:", ok)
