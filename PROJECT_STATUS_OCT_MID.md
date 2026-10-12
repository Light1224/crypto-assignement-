# Quantum-Resistant Crypto Project — Status up to Mid-October

This repository now contains the full **all-member technical implementation scope** from the project PDF.

## Implemented Code (All 3 Members)

## Member 1 — Ring Arithmetic + NTT
- `poly.py`
  - Polynomial add/sub/scalar operations in `Z_q[X]/(X^n+1)`
  - Naive negacyclic multiplication (`O(n^2)`)
- `ntt.py`
  - Primitive root and `2n`-th root (`psi`) construction
  - Forward/inverse NTT (iterative Cooley-Tukey)
  - Negacyclic NTT multiplication via psi-twist (`O(n log n)`)

## Member 2 — RLWE Protocol
- `sampler.py`
  - Uniform, binary, ternary, and discrete-Gaussian toy samplers
- `rlwe.py`
  - `keygen`, `encrypt_bit`, `decrypt_bit`
  - Batch helpers and correctness trials
  - Naive-vs-NTT multiplication benchmark

## Member 3 — Security + Comparison
- `lll.py`
  - From-scratch LLL reduction
  - Gram-Schmidt and Babai nearest-plane
  - q-ary lattice construction and toy LWE secret-recovery attempt
- `experiments.py`
  - Attack scaling sweep (`n = 8,12,16,20`) with timeout handling
  - Noise/failure sweep (`sigma = 1,2,3.2,5`)
  - NTT speed sweep (`n = 64,128,256`)
  - Size comparison table (Ring-LWE vs RSA/ECC)
  - CSV output generation and optional plot generation

## Outputs Generated
Under `results/`:
- `ntt_speedup.csv`
- `noise_failure_sweep.csv`
- `attack_sweep.csv`
- `size_comparison.csv`
- `attack_scaling.png` (if `matplotlib` is installed)

## What this corresponds to in the PDF timeline
By mid-October, this covers the technical core expected by end of:
- Week 1: core arithmetic + basic protocol + basic LLL
- Week 2: NTT integration + attack lattice experiments
- Week 3: parameter sweeps + core data tables/plots

## Remaining non-code tasks (Week 4 style)
- Final notebook write-up structure
- Final 8–10 page report writing
- Viva rehearsal and live demo script
