from __future__ import annotations

"""Validation suite for all tasks due by Oct 12 (through Day 16)."""

import json
import random
from pathlib import Path

from experiments import run_attack_sweep
from ntt import find_psi_2nth_root, poly_mul_ntt_negacyclic
from poly import poly_mul_naive_negacyclic
from rlwe import RLWEParams, correctness_trials, encrypt_bit, keygen


def check_ntt_equals_naive() -> dict:
    rng = random.Random(0)
    q = 12289
    trials_per_n = 4
    ns = (64, 128, 256)

    checked = 0
    for n in ns:
        psi = find_psi_2nth_root(q, n)
        for _ in range(trials_per_n):
            a = [rng.randrange(q) for _ in range(n)]
            b = [rng.randrange(q) for _ in range(n)]
            c1 = poly_mul_naive_negacyclic(a, b, q, n)
            c2 = poly_mul_ntt_negacyclic(a, b, q, n, psi)
            if c1 != c2:
                return {"ok": False, "n": n}
            checked += 1

    return {"ok": True, "instances_checked": checked}


def check_ciphertext_equivalence() -> dict:
    # Same seed, same params except mul backend => identical outputs expected.
    seed = 1337
    p_naive = RLWEParams(n=256, q=12289, sigma=3.2, use_ntt=False, small_dist="gaussian")
    p_ntt = RLWEParams(n=256, q=12289, sigma=3.2, use_ntt=True, small_dist="gaussian")

    rng1 = random.Random(seed)
    rng2 = random.Random(seed)

    pk1, sk1 = keygen(p_naive, rng1)
    pk2, sk2 = keygen(p_ntt, rng2)

    if pk1.a != pk2.a or pk1.b != pk2.b or sk1.s != sk2.s:
        return {"ok": False, "stage": "keygen"}

    for i in range(16):
        bit = i % 2
        ct1 = encrypt_bit(pk1, bit, p_naive, rng1)
        ct2 = encrypt_bit(pk2, bit, p_ntt, rng2)
        if ct1.u != ct2.u or ct1.v != ct2.v:
            return {"ok": False, "stage": f"encrypt_{i}"}

    return {"ok": True, "samples": 16}


def check_correctness_1000() -> dict:
    p = RLWEParams(n=256, q=12289, sigma=3.2, use_ntt=True, small_dist="gaussian")
    stats = correctness_trials(p, trials=1000, seed=11)
    return {"ok": stats["failures"] == 0, **stats}


def check_lll_small_attack() -> dict:
    # Timeout-bounded check aligned with experiments pipeline.
    rows = run_attack_sweep(
        ns=(8, 12, 16, 20),
        m_factor=2,
        q=257,
        sigma=1.2,
        trials=1,
        delta=0.99,
        per_trial_timeout_s=12.0,
        seed_base=9000,
    )

    summary = {
        str(r.n): {
            "successes": r.successes,
            "timeouts": r.timeouts,
            "mean_time_s": r.mean_time_s,
        }
        for r in rows
    }
    return {"ok": True, "bounded_attack_summary": summary}


def main() -> None:
    report = {
        "ntt_vs_naive": check_ntt_equals_naive(),
        "ciphertext_equivalence": check_ciphertext_equivalence(),
        "correctness_1000": check_correctness_1000(),
        "lll_small_attack": check_lll_small_attack(),
    }

    report["all_ok"] = all(v.get("ok", False) for v in report.values() if isinstance(v, dict))

    out_path = Path("results") / "oct12_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("[done] wrote", out_path)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
