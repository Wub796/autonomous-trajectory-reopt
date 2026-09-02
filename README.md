# Autonomous Trajectory Re-optimization & Robust Deep GNC

Reinforcement learning framework for autonomous interplanetary low-thrust trajectory design, robust closed-loop guidance, and flight hardware validation.

---

## Key Research Capabilities

1. **Uncertainty & Disturbance Modeling (Zavoli & Federici, 2021)**:
   - Additive process noise on heliocentric orbital dynamics ($\sigma_r, \sigma_v$).
   - Navigation sensor noise (DSN & optical navigation state estimation errors).
   - Thruster execution errors ($\pm 2.5\%$ magnitude dispersion, $\pm 0.5^\circ$ pointing jitter).
   - Stochastic thruster outages / safe-mode downtime events.
2. **Neural Network Architecture Comparison (Capra, Brandonisio, & Lavagna, 2022)**:
   - Feed-Forward MLP (PPO Baseline) vs. Recurrent LSTM (`RecurrentPPO`) in partially observable Markov Decision Processes (POMDP).
   - Temporal memory filtering of navigation sensor noise and recovery after missed thrust events.
3. **Processor-in-the-Loop (PIL) Validation (Capra, Brandonisio, & Lavagna, 2025)**:
   - ONNX and TorchScript flight binary export.
   - Single-core embedded CPU latency profiling (<10 μs mean latency, >99.999% real-time margin).
   - Closed-loop PIL flight simulation with embedded inference engines.
4. **Extended Monte Carlo Robustness & Sensitivity Analysis (Capra et al., 2022)**:
   - High-fidelity $N=100-1000$ dispersed Monte Carlo trajectory simulations.
   - Parametric sensitivity sweeps across 5 disturbance dimensions.
   - 3D/2D heliocentric dispersion cones, histogram distributions, and tornado charts.

---

## Project Structure

```
research/
├── src/
│   ├── env/
│   │   ├── spacecraft_env.py          # Deterministic Gymnasium physics environment
│   │   ├── robust_spacecraft_env.py   # Stochastic environment with process/sensor/actuation noise
│   │   └── uncertainty.py             # Disturbance modeling & Zavoli-Federici configurations
│   ├── models/
│   │   └── architectures.py           # Feed-Forward MLP, Recurrent LSTM (PPO) & loaders
│   ├── deployment/
│   │   ├── exporter.py                # ONNX and TorchScript export utilities
│   │   └── pil_runner.py              # Processor-in-the-Loop latency benchmark & runner
│   ├── analysis/
│   │   ├── monte_carlo.py             # Multi-trajectory Monte Carlo ensemble engine
│   │   ├── sensitivity.py             # Parametric perturbation sweeps
│   │   └── plotting.py                # Publication-grade astrodynamics visualizations
│   └── utils/
│       └── ephemeris.py               # Astropy planetary state-vector helper
├── scripts/
│   ├── train.py                       # Baseline PPO training entrypoint
│   ├── trajectory.py                  # Optimal trajectory exporter
│   ├── compare_architectures.py       # Architecture benchmark (MLP vs. LSTM under POMDP)
│   ├── pil_benchmark.py               # PIL validation, layer breakdown, memory/power, repeatability
│   ├── robustness_analysis.py         # MC robustness, scenario comparison & sensitivity analysis
│   ├── anomaly_detection_analysis.py  # Isolation Forest metrics, ROC, threshold & mission profile
│   ├── training_analysis.py           # Training curves & Run 1/2/3 comparison (+ live success-rate)
│   └── thruster_degradation_analysis.py  # Log degradation model validation & sensitivity
├── tests/
│   └── test_framework.py              # Comprehensive unit and integration test suite
├── artifacts/                         # Generated models, reports, plots, and CSV telemetry
│   ├── figures/                       # Trajectory dispersion, histogram & tornado figures
│   ├── deployment/                    # ONNX & TorchScript flight binaries
│   ├── architecture_comparison_report.md
│   ├── pil_benchmark_report.md
│   └── robustness_report.md
└── ppo_mars_logs/                     # TensorBoard event files
```

---

## Setup

```bash
uv venv --clear --python 3.11 .venv
source .venv/bin/activate
uv pip install -p .venv "stable-baselines3[extra]" "sb3-contrib" gymnasium astropy scikit-learn pandas numpy matplotlib onnx onnxruntime onnxscript scipy
```

---

## Running Research Workflows

> **All commands should be executed from the project root with `PYTHONPATH=.`**.

### 1. Neural Network Architecture Benchmark (Capra et al. 2022)
Compare Feed-Forward MLP vs. Recurrent LSTM under partial observability and noise, and export the
Section 5.5 data products (training dynamics curves, stability metrics, hyperparameters, host inference timing):
```bash
PYTHONPATH=. .venv/bin/python scripts/compare_architectures.py --timesteps 50000 --eval-episodes 10 --uncertainty zavoli
# Re-run the same command on a Raspberry Pi 4 to populate the embedded inference row
```
Outputs: `artifacts/architecture_comparison_results.csv`, `artifacts/architecture_training_curves.csv`,
`artifacts/training_curves_<arch>.csv`, `artifacts/architecture_hyperparameters.csv`,
`artifacts/figures/architecture_training_curves.png`, `artifacts/architecture_comparison_report.md`.

### 2. Processor-in-the-Loop (PIL) Flight Validation (Capra et al. 2025)
Export ONNX/TorchScript models and profile single-core embedded latency, with Section 5.6 data products
(per-layer breakdown, ONNX/TorchScript op profiles, memory, RPi 4 power estimate, repeatability):
```bash
PYTHONPATH=. .venv/bin/python scripts/pil_benchmark.py --trials 5000 --repeat-runs 3
```
Outputs: `artifacts/pil_benchmark_results.csv`, `artifacts/pil_layer_breakdown_results.csv`,
`artifacts/pil_onnx_node_breakdown_results.csv`, `artifacts/pil_torchscript_op_breakdown_results.csv`,
`artifacts/pil_memory_usage.csv`, `artifacts/pil_power_estimate.csv`, `artifacts/pil_repeatability_results.csv`,
`artifacts/pil_benchmark_report.md`.

### 3. Monte Carlo Robustness & Sensitivity Analysis (Capra et al. 2022)
Run $N$-run dispersed trajectory simulations, sensitivity sweeps, per-category scenario comparison
(Section 5.4: reward/fuel std, failure-mode distribution, time-to-convergence) and nominal-reference
deviation metrics (Section 5.7):
```bash
PYTHONPATH=. .venv/bin/python scripts/robustness_analysis.py --episodes 100 --sensitivity-episodes 10 --scenario-episodes 20 --deviation-runs 25
```
Outputs: `artifacts/monte_carlo_results.csv`, `artifacts/mc_scenario_comparison.csv`,
`artifacts/mc_scenario_summary.csv`, `artifacts/nominal_vs_dispersed_deviation.csv`,
plus `figures/scenario_comparison.png`, `figures/failure_mode_breakdown.png`,
`figures/sensitivity_curves.png`, `figures/nominal_deviation.png`, `artifacts/robustness_report.md`.

### 4. Anomaly Detection Metrics (Section 3.3 / Section 4)
Formal TP/FP/TN/FN, precision/recall/F1, ROC/AUC, threshold analysis, comparison against OC-SVM /
LOF / Elliptic-Envelope, and a full-mission detection profile (false positives & detection delay):
```bash
PYTHONPATH=. .venv/bin/python scripts/anomaly_detection_analysis.py
PYTHONPATH=. .venv/bin/python scripts/anomaly_detection_analysis.py --onset-hour 1200 --n-nominal 10000 --n-anomalous 3000
```
Outputs: `artifacts/anomaly_detection/anomaly_metrics.csv`, `anomaly_roc_data.csv`,
`anomaly_threshold_analysis.csv`, `anomaly_mission_detection.csv`, figures, `anomaly_detection_report.md`.

### 5. Training Dynamics & Run Comparison (Section 4)
Extract training curves (reward, value loss, explained variance, entropy, KL, std) and the Run 1 / Run 2 /
Run 3 phase comparison from the existing TensorBoard logs; optionally run fresh short PPO runs with
periodic success-rate (Mars capture) evaluation:
```bash
PYTHONPATH=. .venv/bin/python scripts/training_analysis.py
PYTHONPATH=. .venv/bin/python scripts/training_analysis.py --phase-runs PPO_1 PPO_2 PPO_3
PYTHONPATH=. .venv/bin/python scripts/training_analysis.py --live --live-timesteps 44160 --live-seeds 1 2 3
```
Outputs: `artifacts/training_analysis/training_curves.csv`, `training_runs_summary.csv`,
`training_success_rate_*.csv` (live), figures, `training_analysis_report.md`.

### 6. Thruster Degradation Model Validation (Section 2.3)
Fit and validate the logarithmic degradation model $P(h) = P_0 - k\,\ln(1 + h/\tau)$ against
literature-calibrated SPT-140 wear-test data (Kamhawi et al. 2014), with parameter sensitivity:
```bash
PYTHONPATH=. .venv/bin/python scripts/thruster_degradation_analysis.py
PYTHONPATH=. .venv/bin/python scripts/thruster_degradation_analysis.py --data my_measured.csv --mission-hours 11040
```
Outputs: `artifacts/thruster_degradation/degradation_empirical_data.csv`, `degradation_fit_results.csv`,
`degradation_residuals.csv`, `degradation_sensitivity.csv`, figures, `thruster_degradation_report.md`.

### 7. Run Unit & Regression Tests
```bash
PYTHONPATH=. .venv/bin/python -m unittest tests/test_framework.py
```

---

## References

- **Zavoli, A., & Federici, L. (2021)**. *Reinforcement Learning for Robust Trajectory Design of Interplanetary Missions*. Journal of Guidance, Control, and Dynamics, 44(8), 1440–1453.
- **Capra, L., Brandonisio, A., & Lavagna, M. (2022)**. *Network architecture and action space analysis for deep reinforcement learning towards spacecraft autonomous guidance*. Advances in Space Research.
- **Capra, L., Brandonisio, A., & Lavagna, M. (2025)**. *Reinforced Model Predictive Guidance and Control for Spacecraft Proximity Operations*. Aerospace, 12(1).