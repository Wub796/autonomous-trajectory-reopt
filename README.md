# Optimization of an Onboard Autonomous Flight Director Using Anomaly Detection and Reinforcement Learning Models
### High-School STEM & AP Calculus AB / AP Statistics / Machine Learning Edition

This repository contains an autonomous guidance and anomaly-recovery framework for interplanetary low-thrust spacecraft (Earth-to-Mars transfer using an SPT-140 Hall-effect thruster).

**Research Question:**
> *How can an autonomous flight director dynamically detect progressive thruster degradation and recalculate a continuous Earth–Mars trajectory within the limits of flight-grade computers?*

**Research Gap (Synthesized Across 32 Sources, #1–#32):**
> Anomaly detection studies (#1, #20, #26, #31) spot abnormal telemetry without steering the spacecraft, while reinforcement learning guidance studies (#3, #4, #19, #21) steer Earth–Mars paths without a detector for progressive Hall thruster wear (#17, #32). This project bridges both domains on flight-grade hardware.

All mathematics and algorithms are formulated using **AP Calculus AB** (rates of change, derivatives, secant slopes, tangent-line Euler integration, logarithmic wear), **AP Statistics** (least-squares regression, standard error of the mean, Z-score detection, controlled & factorial experimental design), and **introductory Machine Learning** (Collective Isolation Forests and finite-difference policy gradients with proximal clipping).

---

## Key Experimental Results

| Experiment | Metric | Result | Physical & Literature Interpretation |
| :--- | :--- | :--- | :--- |
| **Exp 1: Degradation Fit** | Goodness-of-fit ($R^2$) | **0.9966** | Logarithmic model $T(h) = P_0 - k \ln(1 + h/\tau)$ matches SPT-140 wear tests (#17, #32). |
| **Exp 2: Collective Detection** | $F_1$ Score (Window $w=24$) | **0.990 (IF)** / **0.959 (Z)** | 24h rolling features + Isolation Forest (#20, #26, #31) jumps from $F_1=0.543$ ($w=1$) to **0.990** (98% recall, 0 false alarms). |
| **Exp 3: AI Pilot Training** | Adaptive Miss Distance | **1.11M km** (20% cap) | Compact 243-parameter policy trained with clipped updates (#19, #27) adapts to thrust loss. |
| **Exp 4A: Controlled Gap (#28)** | Miss Distance Reduction | **55.8% Error Cut** | Onboard IF + Pilot cuts median miss from **5.87M km** (pre-planned) to **2.60M km**. |
| **Exp 4B: Factorial (#29)** | Controller $\times$ Wear ($10\text{–}25\%$) | **2.45M $\to$ 4.13M km saved** | Error prevented by onboard autonomy grows monotonically as thruster degradation worsens. |
| **Exp 5: Monte Carlo ($N=100$)** | Multi-Source Disturbances | **5.19M ± 1.81M km** | Robust against launch errors, pointing jitter, sensor noise, and engine outages (#3, #12). |
| **Exp 6: Flight Computer** | Decision Latency / RAM | **1.94 μs** / **1.90 KB** | $>99.99999\%$ real-time margin on low-power spacecraft processors (#6, #22, #26). |

---

## Project Structure

```
research/
├── mission/                           # Core self-contained physics and AI modules
│   ├── physics.py                     # 3D gravity, Euler tangent-line steps, Mars orbit (#8)
│   ├── thruster.py                    # SPT-140 thruster (#32), log wear (#17), fault generator (#15)
│   ├── detector.py                    # Point IF (w=1), Z-score (w=24), & Collective IF (w=24) (#20, #26, #31)
│   ├── pilot.py                       # 243-parameter NumPy neural network guidance policy (#5, #18, #22)
│   ├── simulate.py                    # Batched mission flight simulator & multi-reward fitness (#4, #10, #16)
│   └── trainer.py                     # Finite-difference policy gradients w/ proximal clipping (#19, #27, #30)
├── experiments/                       # Automated research benchmark scripts
│   ├── 01_fit_degradation.py          # Exp 1: Fits logarithmic wear model to SPT-140 lab data
│   ├── 02_anomaly_detection.py        # Exp 2: Benchmarks w=1 vs w=24 collective anomaly detectors
│   ├── 03_train_pilot.py              # Exp 3: Trains neural pilot policy with clipped updates
│   ├── 04_gap_experiment.py           # Exp 4: Controlled comparison (#28) & 3x4 Factorial design (#29)
│   ├── 05_monte_carlo.py              # Exp 5: 100-run dispersed Monte Carlo robustness (#3, #12)
│   └── 06_onboard_timing.py           # Exp 6: Microsecond flight computer latency benchmark (#6, #26)
├── paper/                             # Research publication, math primer, & synthesis matrix
│   ├── paper.md                       # Full scientific paper citing all 32 sources (#1-#32)
│   ├── math_primer.md                 # Step-by-step derivations from AP Calculus AB & AP Stats
│   └── sources_and_synthesis_matrix.md# Annotated bibliography & 4-theme synthesis matrix (#1-#32)
├── tests/
│   └── test_mission.py                # 14 comprehensive unit tests (<0.1s execution)
├── artifacts/                         # Generated data tables and figures
│   ├── figures/                       # High-resolution plots (PNG)
│   └── *.csv, *.npz, *.npy            # Telemetry tables, factorial results, weights, and plans
└── legacy/                            # Original graduate-level codebase preserved intact
```

---

## Quick Start

### 1. Run the Unit Tests
```bash
.venv/bin/python -m unittest tests/test_mission.py
```

### 2. Run Any Experiment
```bash
.venv/bin/python experiments/01_fit_degradation.py
.venv/bin/python experiments/02_anomaly_detection.py
.venv/bin/python experiments/03_train_pilot.py
.venv/bin/python experiments/04_gap_experiment.py
.venv/bin/python experiments/05_monte_carlo.py
.venv/bin/python experiments/06_onboard_timing.py
```

### 3. Read the Paper, Math Primer & Synthesis Matrix
- **Scientific Paper (with all 32 sources):** [paper/paper.md](file:///Users/bnjw/Documents/Projects/research/paper/paper.md)
- **Calculus & Statistics Primer:** [paper/math_primer.md](file:///Users/bnjw/Documents/Projects/research/paper/math_primer.md)
- **32-Source Synthesis Matrix & Annotations:** [paper/sources_and_synthesis_matrix.md](file:///Users/bnjw/Documents/Projects/research/paper/sources_and_synthesis_matrix.md)
