# Training Dynamics & Run Comparison Report (Section 4)

Data generated in **live mode**: short PPO runs with different seeds, including periodic success-rate (Mars capture) evaluation.

## 1. Per-Run Stability Summary

| Run | Peak value loss | Final value loss | Final explained variance | Final entropy | Final approx_kl | Final std | EV≥0.9 step | Oscillation index |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| live_seed_1 | 482.5 | 142.8 | 0.04065 | -4.266 | 0.002003 | n/a | n/a | n/a |

## 3. Success Rate Over Training

Success (Mars capture, final distance < 577,000 km) is evaluated periodically during training; per-run success-rate curves are exported to `figures/training_success_rate_*.png` and raw values to `training_success_rate.csv`.

## 4. Data Products

- `training_curves.csv` — long-format (run, step, metric, value) for reward / value loss / explained variance / entropy / KL / std.
- `training_runs_summary.csv` — per-run stability and convergence metrics.
- `figures/` — per-run 4-panel curves, phase comparison, runs summary bars, (live) success-rate curves.
