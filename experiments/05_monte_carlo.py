"""
Experiment 5: Monte Carlo Robustness Analysis
Evaluates the autonomous closed-loop pilot under multi-source stochastic disturbances:
  - Launch injection errors
  - Thruster magnitude dispersion & attitude pointing jitter
  - Safe-mode engine outages
  - Navigation state estimation sensor noise
  - Thruster degradation faults
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mission import physics as P
from mission import thruster as T
from mission import detector as D
from mission import simulate as S

os.makedirs("artifacts/figures", exist_ok=True)

def run():
    print("=== Experiment 5: Monte Carlo Robustness Analysis (N=100 Dispersed Runs) ===")

    if not os.path.exists("artifacts/trained_pilot.npy"):
        raise FileNotFoundError("Run experiments/03_train_pilot.py first!")

    theta = np.load("artifacts/trained_pilot.npy")
    plan_npz = np.load("artifacts/nominal_plan.npz")
    plan = {"throttle": plan_npz["throttle"], "direction": plan_npz["direction"]}

    rng = np.random.default_rng(777)
    N = 100

    # 1. Stochastic Disturbances (Zavoli & Federici 2021 specifications)
    noise = S.NoiseConfig.zavoli_federici()

    # 2. Random Degradation Faults across the ensemble
    onset_hr = rng.uniform(30 * 24, 150 * 24, N)
    severity = rng.uniform(0.08, 0.22, N)
    ramp_hr = rng.uniform(24, 96, N)

    # 3. Onboard Detection
    tel = T.make_telemetry(rng, onset_hr, severity, ramp_hr, hours=T.MISSION_HOURS)
    z_det = D.ZScoreDetector(window=24, threshold=3.5, persist=3, baseline_hours=240)
    alarm_hr, _, believed_onboard = z_det.run(tel["thrust_mN"])
    takeover_hr = np.where(np.isinf(alarm_hr), T.MISSION_HOURS, alarm_hr)

    scen = S.Scenario(
        onset_hour=onset_hr,
        severity=severity,
        ramp_hours=ramp_hr,
        believed=believed_onboard,
        takeover_hour=takeover_hr
    )

    # 4. Simulate Dispersed Ensemble (with trajectory recording)
    print("Propagating 100 dispersed 3D trajectories with stochastic noise...")
    res = S.fly(theta, scen, plan=plan, noise=noise, rng=rng, dt_hours=S.DT_HOURS, record=True)

    miss_km = res["min_dist_km"]
    fuel_kg = res["fuel_used_kg"]
    captured = res["captured"]

    # 5. Statistical Dispersion Metrics
    mean_miss = np.mean(miss_km)
    median_miss = np.median(miss_km)
    std_miss = np.std(miss_km)
    p68_miss = np.percentile(miss_km, 68.3)
    p99_miss = np.percentile(miss_km, 99.7)
    mean_fuel = np.mean(fuel_kg)
    std_fuel = np.std(fuel_kg)
    cap_rate = np.mean(captured) * 100.0

    print(f"\n--- Monte Carlo Results (N={N}) ---")
    print(f"Mars Capture Rate:       {cap_rate:.1f}%")
    print(f"Median Miss Distance:    {median_miss:,.0f} km")
    print(f"Mean Miss Distance:      {mean_miss:,.0f} ± {std_miss:,.0f} km")
    print(f"1-Sigma Dispersion (68%): {p68_miss:,.0f} km")
    print(f"3-Sigma Dispersion (99%): {p99_miss:,.0f} km")
    print(f"Propellant Consumed:     {mean_fuel:.1f} ± {std_fuel:.1f} kg")

    # Save summary table
    df_mc = pd.DataFrame([{
        "Ensemble_Size": N,
        "Capture_Rate_Pct": cap_rate,
        "Median_Miss_km": median_miss,
        "Mean_Miss_km": mean_miss,
        "Std_Miss_km": std_miss,
        "Bound_1Sigma_km": p68_miss,
        "Bound_3Sigma_km": p99_miss,
        "Mean_Fuel_kg": mean_fuel,
        "Std_Fuel_kg": std_fuel
    }])
    df_mc.to_csv("artifacts/monte_carlo_results.csv", index=False)

    # 6. Trajectory Visualization (Ecliptic Plane Projection)
    traj = res["traj"]  # (steps+1, N, 3) in km
    traj_au = traj / P.AU

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))

    # Left: Heliocentric X-Y Trajectory Dispersions
    theta_orbit = np.linspace(0, 2 * np.pi, 200)
    ax1.plot(np.cos(theta_orbit), np.sin(theta_orbit), "b--", alpha=0.5, label="Earth Orbit (1.0 AU)")
    ax1.plot(1.524 * np.cos(theta_orbit), 1.524 * np.sin(theta_orbit), "r--", alpha=0.5, label="Mars Orbit (1.524 AU)")
    ax1.plot(0, 0, "yo", markersize=10, label="Sun")

    # Plot sample trajectories
    for i in range(min(40, N)):
        ax1.plot(traj_au[:, i, 0], traj_au[:, i, 1], color="teal", alpha=0.25, linewidth=0.8)

    # Highlight Mars position at day 290
    mars_r, _ = P.mars_state(290 * P.DAY)
    ax1.plot(mars_r[0] / P.AU, mars_r[1] / P.AU, "ro", markersize=8, label="Mars Encounter Zone")

    ax1.set_xlabel("Heliocentric X (AU)")
    ax1.set_ylabel("Heliocentric Y (AU)")
    ax1.set_title("Heliocentric Trajectory Dispersions (N=40 Shown)")
    ax1.axis("equal")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper left")

    # Right: Histogram of Miss Distances
    ax2.hist(miss_km / 1e6, bins=15, color="teal", edgecolor="black", alpha=0.7)
    ax2.axvline(median_miss / 1e6, color="purple", linestyle="-", linewidth=2, label=f"Median ({median_miss/1e6:.2f}M km)")
    ax2.axvline(P.CAPTURE_RADIUS_KM / 1e6, color="red", linestyle="--", linewidth=2, label="Capture Threshold (0.577M km)")
    ax2.set_xlabel("Final Miss Distance (Million km)")
    ax2.set_ylabel("Frequency (Count)")
    ax2.set_title("Terminal Dispersion Distribution")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("artifacts/figures/monte_carlo_dispersion.png", dpi=300)
    plt.close()
    print("Saved plot to artifacts/figures/monte_carlo_dispersion.png\n")

if __name__ == "__main__":
    run()
