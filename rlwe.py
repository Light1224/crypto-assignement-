from __future__ import annotations

"""Simplified Ring-LWE PKE (NewHope-style toy prototype)."""

from dataclasses import dataclass
import random
import time
from typing import Callable, List, Sequence

from poly import poly_add, poly_centered, poly_mul_naive_negacyclic, poly_sub
from ntt import find_psi_2nth_root, poly_mul_ntt_negacyclic
from sampler import sample_small_poly, sample_uniform_poly

IntVector = List[int]


@dataclass
class RLWEParams:
    n: int = 256
    q: int = 12289
    sigma: float = 3.2
    use_ntt: bool = True
    small_dist: str = "gaussian"


@dataclass
class PublicKey:
    a: IntVector
    b: IntVector


@dataclass
class SecretKey:
    s: IntVector


@dataclass
class Ciphertext:
    u: IntVector
    v: IntVector


def _mul(a: Sequence[int], b: Sequence[int], params: RLWEParams, psi: int | None) -> IntVector:
    if params.use_ntt:
        return poly_mul_ntt_negacyclic(a, b, params.q, params.n, psi=psi)
    return poly_mul_naive_negacyclic(a, b, params.q, params.n)


def _msg_bit_poly(bit: int, n: int, q: int) -> IntVector:
    m = [0] * n
    m[0] = 0 if bit == 0 else (q // 2)
    return m


def keygen(params: RLWEParams, rng: random.Random | None = None) -> tuple[PublicKey, SecretKey]:
    if rng is None:
        rng = random.Random()

    psi = find_psi_2nth_root(params.q, params.n) if params.use_ntt else None

    a = sample_uniform_poly(params.n, params.q, rng)
    s = sample_small_poly(params.n, params.q, params.sigma, rng, mode=params.small_dist)
    e = sample_small_poly(params.n, params.q, params.sigma, rng, mode=params.small_dist)

    b = poly_add(_mul(a, s, params, psi), e, params.q)
    return PublicKey(a=a, b=b), SecretKey(s=s)


def encrypt_bit(pk: PublicKey, bit: int, params: RLWEParams, rng: random.Random | None = None) -> Ciphertext:
    if rng is None:
        rng = random.Random()

    psi = find_psi_2nth_root(params.q, params.n) if params.use_ntt else None

    r = sample_small_poly(params.n, params.q, params.sigma, rng, mode=params.small_dist)
    e1 = sample_small_poly(params.n, params.q, params.sigma, rng, mode=params.small_dist)
    e2 = sample_small_poly(params.n, params.q, params.sigma, rng, mode=params.small_dist)

    u = poly_add(_mul(pk.a, r, params, psi), e1, params.q)
    v = poly_add(poly_add(_mul(pk.b, r, params, psi), e2, params.q), _msg_bit_poly(bit, params.n, params.q), params.q)

    return Ciphertext(u=u, v=v)


def decrypt_bit(sk: SecretKey, ct: Ciphertext, params: RLWEParams) -> int:
    psi = find_psi_2nth_root(params.q, params.n) if params.use_ntt else None

    su = _mul(sk.s, ct.u, params, psi)
    d = poly_sub(ct.v, su, params.q)

    # Decode using coefficient 0 (constant term message encoding)
    x = d[0] % params.q
    d0 = min(x, params.q - x)
    d1 = min((x - params.q // 2) % params.q, (params.q - ((x - params.q // 2) % params.q)) % params.q)
    return 0 if d0 <= d1 else 1


def encrypt_bits(pk: PublicKey, bits: Sequence[int], params: RLWEParams, rng: random.Random | None = None) -> list[Ciphertext]:
    if rng is None:
        rng = random.Random()
    return [encrypt_bit(pk, int(b), params, rng) for b in bits]


def decrypt_bits(sk: SecretKey, cts: Sequence[Ciphertext], params: RLWEParams) -> list[int]:
    return [decrypt_bit(sk, ct, params) for ct in cts]


def correctness_trials(
    params: RLWEParams,
    trials: int = 200,
    seed: int = 42,
) -> dict[str, float | int]:
    rng = random.Random(seed)
    pk, sk = keygen(params, rng)

    failures = 0
    for _ in range(trials):
        bit = rng.randint(0, 1)
        ct = encrypt_bit(pk, bit, params, rng)
        got = decrypt_bit(sk, ct, params)
        if got != bit:
            failures += 1

    return {
        "trials": trials,
        "failures": failures,
        "failure_rate": failures / trials,
    }


def benchmark_multiplication(
    ns: Sequence[int] = (64, 128, 256),
    q: int = 12289,
    rounds: int = 10,
    seed: int = 7,
) -> list[dict[str, float | int]]:
    """Compare naive vs NTT multiplication timings."""
    rng = random.Random(seed)
    out: list[dict[str, float | int]] = []

    for n in ns:
        a = [rng.randrange(q) for _ in range(n)]
        b = [rng.randrange(q) for _ in range(n)]

        # Naive
        t0 = time.perf_counter()
        for _ in range(rounds):
            _ = poly_mul_naive_negacyclic(a, b, q, n)
        naive_t = (time.perf_counter() - t0) / rounds

        # NTT
        psi = find_psi_2nth_root(q, n)
        t0 = time.perf_counter()
        for _ in range(rounds):
            _ = poly_mul_ntt_negacyclic(a, b, q, n, psi=psi)
        ntt_t = (time.perf_counter() - t0) / rounds

        out.append(
            {
                "n": n,
                "naive_time_s": naive_t,
                "ntt_time_s": ntt_t,
                "speedup": (naive_t / ntt_t) if ntt_t > 0 else float("inf"),
            }
        )

    return out


if __name__ == "__main__":
    p = RLWEParams(n=256, q=12289, sigma=3.2, use_ntt=True, small_dist="gaussian")
    stats = correctness_trials(p, trials=100, seed=1)
    print("correctness:", stats)
    print("bench:", benchmark_multiplication(rounds=3))
