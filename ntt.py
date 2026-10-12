from __future__ import annotations

"""From-scratch NTT + negacyclic polynomial multiplication."""

from typing import List, Sequence

IntVector = List[int]


def mod_pow(a: int, e: int, q: int) -> int:
    return pow(a, e, q)


def inv_mod(a: int, q: int) -> int:
    return pow(a, q - 2, q)


def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _prime_factors(x: int) -> list[int]:
    f: list[int] = []
    d = 2
    while d * d <= x:
        if x % d == 0:
            f.append(d)
            while x % d == 0:
                x //= d
        d += 1
    if x > 1:
        f.append(x)
    return f


def find_primitive_root(q: int) -> int:
    """Find a generator of Z_q^* (q prime)."""
    phi = q - 1
    factors = _prime_factors(phi)

    for g in range(2, q):
        ok = True
        for p in factors:
            if pow(g, phi // p, q) == 1:
                ok = False
                break
        if ok:
            return g
    raise ValueError("no primitive root found")


def find_psi_2nth_root(q: int, n: int) -> int:
    """
    Find psi of exact order 2n (requires q ≡ 1 mod 2n).
    Then omega = psi^2 has order n.
    """
    if (q - 1) % (2 * n) != 0:
        raise ValueError("q-1 must be divisible by 2n")
    g = find_primitive_root(q)
    psi = pow(g, (q - 1) // (2 * n), q)
    if pow(psi, 2 * n, q) != 1 or pow(psi, n, q) != q - 1:
        raise ValueError("failed to construct 2n-th primitive root")
    return psi


def _bit_reverse_permute(a: IntVector) -> IntVector:
    n = len(a)
    bits = n.bit_length() - 1
    out = [0] * n
    for i, v in enumerate(a):
        r = 0
        x = i
        for _ in range(bits):
            r = (r << 1) | (x & 1)
            x >>= 1
        out[r] = v
    return out


def ntt(a: Sequence[int], root: int, q: int) -> IntVector:
    n = len(a)
    if not _is_power_of_two(n):
        raise ValueError("length must be a power of two")

    a = _bit_reverse_permute([x % q for x in a])

    length = 2
    while length <= n:
        w_len = pow(root, n // length, q)
        half = length // 2
        for i in range(0, n, length):
            w = 1
            for j in range(half):
                u = a[i + j]
                v = (a[i + j + half] * w) % q
                a[i + j] = (u + v) % q
                a[i + j + half] = (u - v) % q
                w = (w * w_len) % q
        length *= 2

    return a


def intt(a: Sequence[int], root: int, q: int) -> IntVector:
    n = len(a)
    root_inv = inv_mod(root, q)
    out = ntt(a, root_inv, q)
    n_inv = inv_mod(n, q)
    return [(x * n_inv) % q for x in out]


def poly_mul_ntt_negacyclic(a: Sequence[int], b: Sequence[int], q: int, n: int, psi: int | None = None) -> IntVector:
    """
    Negacyclic product in Z_q[X]/(X^n+1) via psi-twist + cyclic NTT size n.
    """
    if len(a) != n or len(b) != n:
        raise ValueError("a and b must both have length n")
    if not _is_power_of_two(n):
        raise ValueError("n must be power of two")

    if psi is None:
        psi = find_psi_2nth_root(q, n)
    omega = (psi * psi) % q  # primitive n-th root

    a_tw = [0] * n
    b_tw = [0] * n

    p = 1
    for i in range(n):
        a_tw[i] = (a[i] % q) * p % q
        b_tw[i] = (b[i] % q) * p % q
        p = (p * psi) % q

    A = ntt(a_tw, omega, q)
    B = ntt(b_tw, omega, q)
    C = [(x * y) % q for x, y in zip(A, B)]
    c_tw = intt(C, omega, q)

    psi_inv = inv_mod(psi, q)
    out = [0] * n
    p = 1
    for i in range(n):
        out[i] = c_tw[i] * p % q
        p = (p * psi_inv) % q

    return out
