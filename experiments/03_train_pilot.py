"""
Experiment 3: Pilot Policy Training with Evolution Strategies (ES)
Stage 1: Train nominal Earth-to-Mars trajectory (healthy thruster).
Stage 2: Fine-tune policy to adapt to thruster degradation scenarios.
The two-stage recipe is mission.trainer.train_policy, shared with Experiment 7.
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
from mission import trainer as T

os.makedirs("artifacts/figures", exist_ok=True)

def run():
    print("=== Experiment 3: Pilot Policy Training with Evolution Strategies ===")

    # Two-stage recipe shared with Experiment 7; these are the headline seeds.
    theta, plan, hist_s1, hist_s2 = T.train_policy(
        seed=42, stage1_seed=101, stage2_seed=202)

    np.savez("artifacts/nominal_plan.npz",
             throttle=plan["throttle"],
             direction=plan["direction"])
    print(f"Recorded nominal plan ({len(plan['throttle'])} steps, fuel used: {plan['result']['fuel_used_kg'][0]:.1f} kg).")

    # Save final weights
    np.save("artifacts/trained_pilot.npy", theta)
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
