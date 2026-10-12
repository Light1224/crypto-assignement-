# Oct 12 Completion Checklist (Project PDF Aligned)

Reference baseline: the schedule in `project.pdf` (30-day plan and work-distribution sections).
Assumption used: Day 1 starts at the project date in PDF (Sep 27), so **Oct 12 corresponds to Day 16**.

## Scope that must be complete by Oct 12 (Day 16)

## Week 1 (Days 1–7)
- [x] Ring polynomial arithmetic in `Z_q[X]/(X^n+1)` from scratch
  - `poly.py`
- [x] Naive polynomial multiplication
  - `poly_mul_naive_negacyclic`
- [x] Samplers for small noise/secrets
  - `sampler.py`
- [x] Ring-LWE keygen/encrypt/decrypt (naive path available)
  - `rlwe.py`
- [x] LLL from scratch for tiny matrices
  - `lll.py`

## Week 2 (Days 8–14)
- [x] NTT implementation and inverse
  - `ntt.py`
- [x] NTT integrated into RLWE path
  - `rlwe.py` (`use_ntt=True`)
- [x] NTT correctness against naive multiplication
  - checked by `validate_oct12.py`
- [x] LLL attack lattice construction + tiny attack runs
  - `lll.py`, `experiments.py`

## Week 3 up to Day 16 (Days 15–16 only)
- [x] Parameter sweep for NTT performance (`n=64,128,256`)
  - `results/ntt_speedup.csv`
  - `results/ntt_speedup.png`
- [x] Noise sweep (`sigma=1,2,3.2,5`) with failure rates
  - `results/noise_failure_sweep.csv`
  - `results/noise_failure_sweep.png`
- [x] Auto-generated stress noise sweep when baseline has no failures
  - `results/noise_failure_sweep_stress.csv`
- [x] Attack scaling sweep (`n=8,12,16,20`) with timing/timeouts
  - `results/attack_sweep.csv`
  - `results/attack_scaling.png`
- [x] Size comparison table (Ring-LWE vs RSA/ECC)
  - `results/size_comparison.csv`

## Validation completed
- [x] Syntax check all modules
- [x] NTT == naive multiplication (multiple instances)
- [x] NTT and naive produce identical RLWE outputs with same randomness
- [x] 1000-trial encrypt/decrypt correctness run
- [x] Small-dimension LLL attack sanity runs (timeout-bounded)

Validation artifact:
- `results/oct12_validation.json`

Execution/package-management mode:
- All project commands and scripts executed via `uv` (`uv sync`, `uv run ...`).

## Notes
- `attack_sweep.csv` now includes timeout-aware metrics (`completed_trials`, `success_rate_completed`, `timeout_rate`).
- This checklist intentionally stops at Day 16 (Oct 12) and does not claim Week 4 write-up/viva tasks are complete.
