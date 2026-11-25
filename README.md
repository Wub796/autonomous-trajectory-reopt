# Detect, Then Steer: An Onboard Flight Director for Progressive Hall-Thruster Degradation on Earth–Mars Transfers

This repo is a simulator for an Earth-to-Mars transfer flown by an SPT-140 Hall-effect thruster, plus the experiments I ran to test whether a small onboard detector and guidance policy can do anything useful while the thruster slowly wears out.

The question I started with: can a lightweight onboard detector–policy pair turn progressive Hall-thruster degradation into a smaller final Mars miss, without going over a flight computer's compute budget?

## The gap I was trying to fill

Anomaly-detection work (Hundman et al., 2018; Yang et al., 2022; Liu et al., 2008; Chandola et al., 2009) flags telemetry anomalies but never steers the spacecraft. Reinforcement-learning guidance work steers low-thrust trajectories — Earth–Mars in Zavoli and Federici (2021) and Miller et al. (2019), Earth–Moon halo transfers in Sullivan et al. (2021) — but does not model progressive Hall-thruster wear (Snyder et al., 2019) or the measured SPT-140 performance envelope (Snyder et al., 2018). This project closes both loops in one simulator: the detector's alarm hands a health estimate to a 243-parameter guidance policy, and the resulting trajectory is scored against pre-planned and ground-delayed baselines.

The maths is kept deliberately basic: elementary calculus (rates of change, derivatives, secant slopes, semi-implicit Euler integration, a logarithmic wear law), basic statistics (least-squares regression, standard error of the mean, z-score detection, controlled and factorial designs) and simple machine learning (collective isolation forests, finite-difference evolution strategies with a PPO-inspired update cap). There is no deep-learning framework involved — the policy is a 243-parameter NumPy MLP.

## What the experiments show

**Degradation fit (Exp 1).** A log model, $T(h) = P_0 - k \ln(1 + h/\tau)$, fits the ten-point wear series with $R^2 = 0.9966$. The 4.5 kW operating point (280 mN, 1782 s) sits close to the published SPT-140 measurements of 279.3 mN and 1794 s (Snyder et al., 2018), and the wear trend follows the ~8% loss over the first 2,000 h seen in the SPT-140 life test (Snyder et al., 2019); the synthetic series loses about 1.6% over that same interval.

**Detector (Exp 2).** With 24-hour rolling features and an isolation forest (Liu et al., 2008), feature extraction motivated by Yang et al. (2022) and collective-anomaly framing from Chandola et al. (2009), $F_1$ goes from 0.543 at $w=1$ to 0.990 at $w=24$, with 98% recall and zero false alarms across 50 healthy streams. The z-score detector reaches 0.959: it fires 6.8 h earlier but misses three faults.

**Pilot training (Exp 3).** The policy has 243 parameters and is trained by antithetic finite-difference evolution strategies with rank centring (Schulman et al., 2016) and a PPO-inspired per-parameter update cap (Schulman et al., 2017). It is not PPO: no critic, no ratio clipping, no generalised advantage estimation. The best validation checkpoint in Stage 2 (generation 500) reached a 0.541M km median miss across 10 fixed scenarios.

**Controlled comparison (Exp 4A).** Onboard IF plus the pilot cuts median miss from 2.35M km (pre-planned) to 1.14M km, a 51.5% reduction, for about 46 kg of propellant, and lifts capture from 0% to 15%. A seven-day ground replan is close: 0.91M km and 32% capture in this run. Both reductions survive retraining (onboard 52–67%, ground 57–73% across five seeds), even though the pre-planned baseline itself moves from 2.35M to 4.84M km between draws.

**Factorial (Exp 4B).** Controller × wear, 10–25%: at 10% the onboard policy is slightly worse than the plan (1.74M against 1.57M km), but the gain grows with severity to 0.82M km at 15% and 3.21M km at 25%.

**Monte Carlo (Exp 5).** With the full disturbance stack (launch errors, pointing jitter, safing events, navigation noise), the mean miss is 2.85M ± 1.67M km (median 2.49M) with 3% capture, no better than the fixed plan's clean mean of 2.72M km. The clean-dynamics gain does not survive. This measures the limit of the training distribution; it is not evidence of robustness.

**Compute (Exp 6).** Inference takes 1.94 μs for 1.9 KB of parameters, measured on a host CPU in float64 NumPy, for the policy only — forest inference is not timed. That still leaves roughly ten orders of magnitude inside a 12-hour guidance cycle, so compute is not the binding constraint. This is not a flight-hardware measurement.

**What this does not show.** The absolute miss is a property of the training draw as much as of the mission: five retrainings moved the pre-planned median from 2.35M to 4.84M km, and the closed-loop medians from 0.90M to 2.07M km (ground) and 1.12M to 2.28M km (onboard), while the ground–onboard difference stayed between −76,000 and +192,000 km. The reduction is the stable part — both controllers beat their own plan in all five seeds — but the ordering between them is unresolved. A crossed check (all five policies against all five plans, 25 cells) puts 56% of the variance in that difference on the plan draw and 31% on the policy, with ground ahead in 18 of 25 cells. Five seeds is also a small sample: the seed-level mean edge is 126,000 ± 114,000 km (t(4) = 2.5, p = 0.07), so the pooled p-value over 500 scenario-seed pairs overstates the evidence. Treat ground and onboard as comparable on slow faults; the case for autonomy here rests on availability and fast faults rather than on light-time. The dynamics are two-body with an analytic Mars orbit, not an ephemeris. Full detail is in Sections 6–7 and Appendix A of the paper.

## Project structure

```
research/
├── mission/                           # Core self-contained physics and guidance modules
│   ├── physics.py                     # 3D solar gravity, semi-implicit Euler, inclined Mars orbit
│   ├── thruster.py                    # SPT-140 nominal point, log wear model, fault generator
│   ├── detector.py                    # Point IF (w=1), Z-score (w=24), & Collective IF (w=24)
│   ├── pilot.py                       # 243-parameter NumPy MLP guidance policy
│   ├── simulate.py                    # Batched mission simulator & multi-reward fitness
│   └── trainer.py                     # Evolution strategies, update cap, shared two-stage training recipe
├── experiments/                       # Automated research benchmark scripts
│   ├── 01_fit_degradation.py          # Exp 1: Fits logarithmic wear model
│   ├── 02_anomaly_detection.py        # Exp 2: Benchmarks w=1 vs w=24 collective detectors
│   ├── 03_train_pilot.py              # Exp 3: Trains the policy, records the nominal plan
│   ├── 04_gap_experiment.py           # Exp 4: Controlled comparison & 3x4 factorial design
│   ├── 05_monte_carlo.py              # Exp 5: 100-run dispersed Monte Carlo
│   ├── 06_onboard_timing.py           # Exp 6: Host-CPU inference latency benchmark
│   └── 07_training_draw_sensitivity.py  # Exp 7: crossed plan x policy training-draw check
├── paper/
│   └── paper.md                       # Full scientific paper (33 references)
├── tests/
│   └── test_mission.py                # 14 unit tests (~0.08 s execution)
└── artifacts/                         # Generated data tables and figures
    ├── figures/                       # High-resolution plots (PNG)
    └── *.csv, *.npz, *.npy            # Telemetry tables, factorial results, weights, and plans
```

## Running it

The unit tests take well under a second:

```bash
.venv/bin/python -m unittest tests/test_mission.py
```

Experiments are standalone scripts; install the pinned versions first (`pip install -r requirements.txt`) and run whichever one you need (prefix `MPLBACKEND=Agg` on a headless machine):

```bash
.venv/bin/python experiments/01_fit_degradation.py
.venv/bin/python experiments/02_anomaly_detection.py
.venv/bin/python experiments/03_train_pilot.py
.venv/bin/python experiments/04_gap_experiment.py
.venv/bin/python experiments/05_monte_carlo.py
.venv/bin/python experiments/06_onboard_timing.py
.venv/bin/python experiments/07_training_draw_sensitivity.py
```

The last one re-trains four extra draws (about six minutes) and writes `artifacts/training_draw_sensitivity.csv`; trained draws are cached in `artifacts/training_draws/` so a re-run only re-flies the table.

## The paper

[paper/paper.md](paper/paper.md) is the authoritative write-up: 33 references, plus an appendix on reproducibility and data provenance. Where this README and the paper disagree, the paper wins.
