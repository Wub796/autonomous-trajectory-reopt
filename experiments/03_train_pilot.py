"""
Experiment 3: Pilot Policy Training with Evolution Strategies (ES)
Stage 1: Train nominal Earth-to-Mars trajectory (healthy thruster).
Stage 2: Fine-tune policy to adapt to thruster degradation scenarios.
Outputs:
  - artifacts/trained_pilot.npy
  - artifacts/nominal_plan.npz
  - artifacts/training_curves.csv
  - artifacts/figures/training_curves.png
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mission import physics as P
from mission import pilot as AI
from mission import simulate as S
from mission import trainer as T

os.makedirs("artifacts/figures", exist_ok=True)

def run():
    print("=== Experiment 3: AI Pilot Training with Evolution Strategies ===")
    rng = np.random.default_rng(42)

    # Initialize neural network weights (243 parameters)
    theta0 = AI.init_params(rng, scale=0.4)

    # Evaluation function for tracking progress
    def evaluate(theta):
        res = S.fly(theta, S.Scenario.healthy(1), dt_hours=S.DT_HOURS)
        score = float(S.fitness(res)[0])
        cap = float(res["captured"][0])
        miss = float(res["min_dist_km"][0])
        fuel = float(res["fuel_used_kg"][0])
        return {"score": score, "capture_rate": cap, "median_miss_km": miss, "fuel_used_kg": fuel}

    # --- Stage 1: Nominal Mission Training ---
    print("\n--- Stage 1: Nominal Flight Optimization (400 generations) ---")
    def sample_nominal(r, n):
        return S.Scenario.healthy(n)

    best_theta_stage1, hist_s1 = T.train_es(
        theta0=theta0,
        sample_scenarios=sample_nominal,
        generations=400,
        pairs=64,
        sigma=0.035,
        learning_rate=0.025,
        n_scen=1,
        eval_fn=evaluate,
        eval_every=20,
        seed=101
    )

    # Record the nominal reference plan from the best nominal pilot
    plan = S.record_plan(best_theta_stage1, dt_hours=S.DT_HOURS)
    np.savez("artifacts/nominal_plan.npz",
             throttle=plan["throttle"],
             direction=plan["direction"])
    print(f"Recorded nominal plan ({len(plan['throttle'])} steps, fuel used: {plan['result']['fuel_used_kg'][0]:.1f} kg).")

    # --- Stage 2: Adaptive Fine-Tuning under Degradation ---
    print("\n--- Stage 2: Degradation Adaptation Training (200 generations) ---")
    def sample_degraded(r, n):
        # Scenarios where thruster loses 10% to 25% thrust between day 30 and 150
        onset = r.uniform(30 * 24, 150 * 24, n)
        sev = r.uniform(0.10, 0.25, n)
        ramp = r.uniform(24, 72, n)
        scen = S.Scenario(onset, sev, ramp)
        # Assume perfect detector feedback in training
        H = S.MISSION_HOURS
        h = np.arange(H)
        # Believed health follows the degraded health after onset
        believed = np.ones((n, H))
        for i in range(n):
            mask = h >= onset[i]
            believed[i, mask] = 1.0 - sev[i]
        scen.believed = believed
        scen.takeover_hour = onset  # Pilot takes control from onset
        return scen

    def eval_robust(theta):
        # Evaluate on 10 fixed validation degradation scenarios
        val_rng = np.random.default_rng(999)
        val_scen = sample_degraded(val_rng, 10)
        res = S.fly(theta, val_scen, plan=plan, dt_hours=S.DT_HOURS)
        score = float(np.mean(S.fitness(res)))
        cap = float(np.mean(res["captured"]))
        miss = float(np.median(res["min_dist_km"]))
        fuel = float(np.mean(res["fuel_used_kg"]))
        return {"score": score, "capture_rate": cap, "median_miss_km": miss, "fuel_used_kg": fuel}

    best_theta_stage2, hist_s2 = T.train_es(
        theta0=best_theta_stage1,
        sample_scenarios=sample_degraded,
        generations=200,
        pairs=64,
        sigma=0.03,
        learning_rate=0.02,
        n_scen=2,
        plan=plan,
        eval_fn=eval_robust,
        eval_every=20,
        seed=202
    )

    # Save final weights
    np.save("artifacts/trained_pilot.npy", best_theta_stage2)
    print("Saved final trained weights to artifacts/trained_pilot.npy")

    # Combine history & save to CSV
    df_s1 = pd.DataFrame(hist_s1)
    df_s1["stage"] = "Stage 1 (Nominal)"
    df_s2 = pd.DataFrame(hist_s2)
    df_s2["stage"] = "Stage 2 (Adaptive)"
    df_s2["generation"] += 400
    df_all = pd.concat([df_s1, df_s2], ignore_index=True)
    df_all.to_csv("artifacts/training_curves.csv", index=False)

    # Plot training curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(df_s1["generation"], df_s1["median_miss_km"] / 1e6, "b-o", label="Stage 1: Nominal")
    ax1.plot(df_s2["generation"], df_s2["median_miss_km"] / 1e6, "g-s", label="Stage 2: Degraded")
    ax1.axhline(P.CAPTURE_RADIUS_KM / 1e6, color="red", linestyle="--", label="Mars Capture Bound (0.577M km)")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Median Miss Distance (Million km)")
    ax1.set_title("Miss Distance Reduction via Evolution Strategies")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax1.set_yscale("log")

    ax2.plot(df_all["generation"], df_all["score"], "purple", label="Fitness Score")
    ax2.axvline(400, color="gray", linestyle=":", label="Stage 1 -> 2 Transition")
    ax2.set_xlabel("Generation")
    ax2.set_ylabel("Fitness Score (Log Scaled)")
    ax2.set_title("Evolutionary Fitness Trajectory")
    ax2.legend()
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("artifacts/figures/training_curves.png", dpi=300)
    plt.close()
    print("Saved training plot to artifacts/figures/training_curves.png\n")

if __name__ == "__main__":
    run()
