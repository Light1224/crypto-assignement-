from __future__ import annotations

"""
Member 3 scope: experiments and comparison tooling.

PDF-aligned responsibilities implemented here:
- Run LLL attack parameter sweeps (n = 8, 12, 16, 20 by default)
- Measure attack timing and success rate
- Save CSV data for report reproducibility
- Generate attack-scaling plot with exponential fit (optional, if matplotlib exists)
- Build Ring-LWE vs RSA/ECC size comparison table
"""

from dataclasses import dataclass
import csv
import math
import multiprocessing as mp
import statistics
import time
from pathlib import Path
from typing import List, Sequence

from lll import sample_lwe_instance, recover_binary_secret_with_lll
from rlwe import RLWEParams, benchmark_multiplication, correctness_trials


@dataclass
class SweepRow:
    n: int
    m: int
    q: int
    sigma: float
    trials: int
    successes: int
    timeouts: int
    completed_trials: int
    success_rate: float
    success_rate_completed: float
    timeout_rate: float
    mean_time_s: float
    median_time_s: float
    mean_completed_time_s: float
    median_completed_time_s: float


def _attack_trial_worker(
    n: int,
    m: int,
    q: int,
    sigma: float,
    delta: float,
    seed: int,
    out_queue: mp.Queue,
) -> None:
    inst = sample_lwe_instance(
        n=n,
        m=m,
        q=q,
        sigma=sigma,
        secret_distribution="binary",
        seed=seed,
    )
    attack_res = recover_binary_secret_with_lll(inst.A, inst.b, q=q, delta=delta)
    success = attack_res.recovered_secret_binary == inst.s
    out_queue.put(success)


def _run_attack_trial_with_timeout(
    n: int,
    m: int,
    q: int,
    sigma: float,
    delta: float,
    seed: int,
    timeout_s: float,
) -> tuple[bool, float, bool]:
    """
    Returns: (success, elapsed_seconds, timed_out)
    """
    ctx = mp.get_context("spawn")
    q_out: mp.Queue = ctx.Queue(maxsize=1)
    p = ctx.Process(
        target=_attack_trial_worker,
        args=(n, m, q, sigma, delta, seed, q_out),
    )

    t0 = time.perf_counter()
    p.start()
    p.join(timeout=timeout_s)

    if p.is_alive():
        p.terminate()
        p.join()
        elapsed = time.perf_counter() - t0
        return False, elapsed, True

    elapsed = time.perf_counter() - t0
    success = False
    if p.exitcode == 0 and not q_out.empty():
        success = bool(q_out.get())
    return success, elapsed, False


def run_attack_sweep(
    ns: Sequence[int] = (8, 12, 16, 20),
    m_factor: int = 2,
    q: int = 257,
    sigma: float = 1.2,
    trials: int = 5,
    delta: float = 0.99,
    seed_base: int = 1000,
    per_trial_timeout_s: float = 25.0,
) -> List[SweepRow]:
    """
    Run LLL-attack experiments over dimension sweep.

    Args:
        ns: secret dimensions to sweep.
        m_factor: use m = m_factor * n equations.
        q: LWE modulus.
        sigma: noise stdev for toy-LWE sampler.
        trials: trials per n.
        delta: LLL delta.
        seed_base: deterministic base for reproducibility.
        per_trial_timeout_s: per-instance timeout (seconds).

    Returns:
        List of sweep rows, one per n.
    """
    if trials <= 0:
        raise ValueError("trials must be > 0")

    out: List[SweepRow] = []

    for n in ns:
        m = m_factor * n
        times: List[float] = []
        completed_times: List[float] = []
        successes = 0
        timeouts = 0

        for t in range(trials):
            seed = seed_base + n * 1000 + t
            success, elapsed, timed_out = _run_attack_trial_with_timeout(
                n=n,
                m=m,
                q=q,
                sigma=sigma,
                delta=delta,
                seed=seed,
                timeout_s=per_trial_timeout_s,
            )

            times.append(elapsed)
            if timed_out:
                timeouts += 1
            else:
                completed_times.append(elapsed)
                if success:
                    successes += 1

        mean_t = statistics.fmean(times)
        median_t = statistics.median(times)
        completed_trials = trials - timeouts
        success_rate = successes / trials
        success_rate_completed = (successes / completed_trials) if completed_trials > 0 else 0.0
        timeout_rate = timeouts / trials
        mean_completed_t = statistics.fmean(completed_times) if completed_times else float("nan")
        median_completed_t = statistics.median(completed_times) if completed_times else float("nan")

        out.append(
            SweepRow(
                n=n,
                m=m,
                q=q,
                sigma=sigma,
                trials=trials,
                successes=successes,
                timeouts=timeouts,
                completed_trials=completed_trials,
                success_rate=success_rate,
                success_rate_completed=success_rate_completed,
                timeout_rate=timeout_rate,
                mean_time_s=mean_t,
                median_time_s=median_t,
                mean_completed_time_s=mean_completed_t,
                median_completed_time_s=median_completed_t,
            )
        )

    return out


def save_sweep_csv(rows: Sequence[SweepRow], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "n",
                "m",
                "q",
                "sigma",
                "trials",
                "successes",
                "timeouts",
                "completed_trials",
                "success_rate",
                "success_rate_completed",
                "timeout_rate",
                "mean_time_s",
                "median_time_s",
                "mean_completed_time_s",
                "median_completed_time_s",
            ]
        )
        for r in rows:
            w.writerow(
                [
                    r.n,
                    r.m,
                    r.q,
                    r.sigma,
                    r.trials,
                    r.successes,
                    r.timeouts,
                    r.completed_trials,
                    f"{r.success_rate:.6f}",
                    f"{r.success_rate_completed:.6f}",
                    f"{r.timeout_rate:.6f}",
                    f"{r.mean_time_s:.6f}",
                    f"{r.median_time_s:.6f}",
                    f"{r.mean_completed_time_s:.6f}" if not math.isnan(r.mean_completed_time_s) else "NaN",
                    f"{r.median_completed_time_s:.6f}" if not math.isnan(r.median_completed_time_s) else "NaN",
                ]
            )


def _fit_exponential(ns: Sequence[int], ys: Sequence[float]) -> tuple[float, float]:
    """
    Fit y ≈ a * exp(b n) by linear regression on ln(y).

    Returns (a, b). Requires all y > 0.
    """
    if len(ns) != len(ys) or len(ns) < 2:
        raise ValueError("need same-length ns/ys with at least 2 points")
    if any(y <= 0.0 for y in ys):
        raise ValueError("all y must be > 0 for log fit")

    x = [float(v) for v in ns]
    ly = [math.log(v) for v in ys]

    x_mean = statistics.fmean(x)
    y_mean = statistics.fmean(ly)

    num = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, ly))
    den = sum((xi - x_mean) ** 2 for xi in x)
    if den == 0.0:
        raise ValueError("cannot fit: zero variance in x")

    b = num / den
    ln_a = y_mean - b * x_mean
    a = math.exp(ln_a)
    return a, b


def _get_plt():
    try:
        import matplotlib.pyplot as plt  # type: ignore
        return plt
    except Exception:
        return None


def maybe_plot_attack_scaling(rows: Sequence[SweepRow], out_png: str | Path) -> bool:
    """
    Try generating attack-time scaling plot. Returns False if matplotlib is unavailable.
    """
    plt = _get_plt()
    if plt is None:
        return False

    ns = [r.n for r in rows]
    ys_all = [r.mean_time_s for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, ys_all, marker="o", label="Mean time (including timeouts)")

    valid_fit_points = [(r.n, r.mean_completed_time_s) for r in rows if not math.isnan(r.mean_completed_time_s) and r.mean_completed_time_s > 0]
    if valid_fit_points:
        x_ok = [p[0] for p in valid_fit_points]
        y_ok = [p[1] for p in valid_fit_points]
        ax.plot(x_ok, y_ok, marker="s", linestyle="--", label="Mean completed-trial time")

    # Exponential fit on completed trials only
    if len(valid_fit_points) >= 2:
        a, b = _fit_exponential(x_ok, y_ok)
        xfit = list(range(min(x_ok), max(x_ok) + 1))
        yfit = [a * math.exp(b * x) for x in xfit]
        ax.plot(xfit, yfit, ":", label=f"Fit on completed: t ≈ {a:.2e}·exp({b:.3f}n)")

    ax.set_yscale("log")
    ax.set_xlabel("Secret dimension n")
    ax.set_ylabel("Attack time (seconds, log scale)")
    ax.set_title("LLL Attack Scaling on Toy-LWE Instances")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def maybe_plot_ntt_speedup(rows: Sequence[dict[str, float | int]], out_png: str | Path) -> bool:
    plt = _get_plt()
    if plt is None:
        return False

    ns = [int(r["n"]) for r in rows]
    naive = [float(r["naive_time_s"]) for r in rows]
    ntt = [float(r["ntt_time_s"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, naive, marker="o", label="Naive negacyclic mul")
    ax.plot(ns, ntt, marker="o", label="NTT negacyclic mul")
    ax.set_xlabel("Polynomial degree n")
    ax.set_ylabel("Time per multiplication (seconds)")
    ax.set_title("Naive vs NTT Multiplication Timing")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def maybe_plot_noise_failure(rows: Sequence[dict[str, float | int]], out_png: str | Path) -> bool:
    plt = _get_plt()
    if plt is None:
        return False

    sigmas = [float(r["sigma"]) for r in rows]
    failure = [float(r["failure_rate"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sigmas, failure, marker="o", label="Measured failure rate")
    ax.set_xlabel("Noise sigma")
    ax.set_ylabel("Decryption failure rate")
    ax.set_title("Failure Rate vs Noise Level")
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    return True


def build_size_comparison_table(
    n_ring: int = 256,
    q_ring: int = 12289,
) -> list[list[str]]:
    """
    Build a practical size-comparison table for report use.

    Notes:
    - Ring-LWE values are estimated from coefficient counts.
    - RSA values use modulus size as key and ciphertext size proxy.
    - ECC-256 ciphertext is application-dependent; we use a typical ECIES-style
      envelope estimate (ephemeral pubkey + symmetric payload tag/nonce overhead)
      as an approximate educational baseline.
    """
    coeff_bits = math.ceil(math.log2(q_ring))

    # Simplified Ring-LWE estimates (public key: (a,b), ciphertext: (u,v)).
    ring_pub_bits = 2 * n_ring * coeff_bits
    ring_ct_bits = 2 * n_ring * coeff_bits

    rows = [
        ["Scheme", "Public Key (bits)", "Ciphertext (bits)", "Notes"],
        [
            f"Ring-LWE (n={n_ring}, q={q_ring})",
            str(ring_pub_bits),
            str(ring_ct_bits),
            "Estimated uncompressed polynomial coefficients",
        ],
        ["RSA-2048", "2048", "2048", "Modulus-size baseline"],
        ["RSA-4096", "4096", "4096", "Modulus-size baseline"],
        ["ECC-256", "512", "800", "Approx ECIES-style envelope baseline"],
    ]

    return rows


def save_size_table_csv(rows: Sequence[Sequence[str]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for row in rows:
            w.writerow(list(row))


def run_noise_failure_sweep(
    sigmas: Sequence[float] = (1.0, 2.0, 3.2, 5.0),
    n: int = 256,
    q: int = 12289,
    trials: int = 200,
    seed_base: int = 500,
) -> list[dict[str, float | int]]:
    """Member 2 scope: measure decrypt failure rate vs sigma."""
    rows: list[dict[str, float | int]] = []
    q4 = q / 4.0
    for i, sigma in enumerate(sigmas):
        params = RLWEParams(n=n, q=q, sigma=sigma, use_ntt=True, small_dist="gaussian")
        stats = correctness_trials(params, trials=trials, seed=seed_base + i)
        rows.append(
            {
                "sigma": sigma,
                "sigma_over_q4": float(sigma / q4),
                "trials": int(stats["trials"]),
                "failures": int(stats["failures"]),
                "failure_rate": float(stats["failure_rate"]),
            }
        )
    return rows


def save_dict_rows_csv(rows: Sequence[dict[str, float | int]], path: str | Path, field_order: Sequence[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(list(field_order))
        for r in rows:
            out = []
            for k in field_order:
                v = r[k]
                if isinstance(v, float):
                    out.append(f"{v:.6f}")
                else:
                    out.append(str(v))
            w.writerow(out)


def main() -> None:
    """
    Full all-member experiment pipeline from the project PDF:
      - Member 1: NTT speedup sweep
      - Member 2: noise/failure sweep
      - Member 3: LLL attack sweep + size comparison
    """
    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Member 1 (poly/NTT): speed benchmark
    speed_rows = benchmark_multiplication(ns=(64, 128, 256), q=12289, rounds=3, seed=7)
    speed_csv = out_dir / "ntt_speedup.csv"
    save_dict_rows_csv(speed_rows, speed_csv, field_order=("n", "naive_time_s", "ntt_time_s", "speedup"))
    speed_plot_ok = maybe_plot_ntt_speedup(speed_rows, out_dir / "ntt_speedup.png")

    # Member 2 (protocol): failure rate vs sigma
    noise_rows = run_noise_failure_sweep(sigmas=(1.0, 2.0, 3.2, 5.0), n=256, q=12289, trials=120)
    noise_csv = out_dir / "noise_failure_sweep.csv"
    save_dict_rows_csv(
        noise_rows,
        noise_csv,
        field_order=("sigma", "sigma_over_q4", "trials", "failures", "failure_rate"),
    )
    noise_plot_ok = maybe_plot_noise_failure(noise_rows, out_dir / "noise_failure_sweep.png")

    # If the required sigma range shows no failures, automatically run a stress sweep
    # so the report still contains a visible transition region.
    stress_rows: list[dict[str, float | int]] = []
    stress_csv: Path | None = None
    if all(float(r["failure_rate"]) == 0.0 for r in noise_rows):
        stress_sigmas = (8.0, 16.0, 32.0, 64.0, 128.0)
        stress_rows = run_noise_failure_sweep(sigmas=stress_sigmas, n=256, q=12289, trials=80, seed_base=8000)
        stress_csv = out_dir / "noise_failure_sweep_stress.csv"
        save_dict_rows_csv(
            stress_rows,
            stress_csv,
            field_order=("sigma", "sigma_over_q4", "trials", "failures", "failure_rate"),
        )

    # Member 3 (security): attack scaling
    attack_rows = run_attack_sweep(
        ns=(8, 12, 16, 20),
        m_factor=2,
        q=257,
        sigma=1.2,
        trials=2,
        delta=0.99,
        per_trial_timeout_s=20.0,
    )
    attack_csv = out_dir / "attack_sweep.csv"
    save_sweep_csv(attack_rows, attack_csv)

    plot_path = out_dir / "attack_scaling.png"
    plot_ok = maybe_plot_attack_scaling(attack_rows, plot_path)

    # Member 3 (comparison)
    size_rows = build_size_comparison_table(n_ring=256, q_ring=12289)
    size_csv = out_dir / "size_comparison.csv"
    save_size_table_csv(size_rows, size_csv)

    print("[done] ntt speed CSV:", speed_csv)
    print("[done] noise sweep CSV:", noise_csv)
    if stress_csv is not None:
        print("[done] noise stress CSV:", stress_csv)
    print("[done] attack sweep CSV:", attack_csv)
    print("[done] size table CSV:", size_csv)

    if speed_plot_ok:
        print("[done] ntt speed plot:", out_dir / "ntt_speedup.png")
    if noise_plot_ok:
        print("[done] noise sweep plot:", out_dir / "noise_failure_sweep.png")
    if plot_ok:
        print("[done] attack scaling plot:", plot_path)
    if not (speed_plot_ok and noise_plot_ok and plot_ok):
        print("[warn] matplotlib not available; skipped one or more plots")


if __name__ == "__main__":
    main()
