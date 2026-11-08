"""
Experiment 4: The Core Research Gap & Factorial Experiment (#28 Creswell, #29 Leedy & Ormrod)
Evaluates Autonomous Onboard Re-optimization when progressive thruster degradation occurs.

Part A — Controlled Comparison Across 100 Dispersed Fault Scenarios (#28 Creswell, 2012):
  1. Pre-Planned Baseline (No adaptation)
  2. Ground-in-the-Loop (7-day delay due to DSN scheduling + ground re-planning)
  3. Autonomous Onboard: Physics-Informed Z-Score + Neural Pilot
  4. Autonomous Onboard: Collective Isolation Forest (#20, #26, #31) + Neural Pilot

Part B — Two-Way Factorial Design (#29 Leedy & Ormrod, 2015):
  Tests the interaction between Controller Setup (Factor A) and Thruster Degradation
  Severity Level (Factor B: 10%, 15%, 20%, 25% thrust loss) while holding starting
  orbital conditions and onset times fixed across groups (#28).
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
    print("=== Experiment 4: Research Gap & Factorial Evaluation (#28, #29) ===")

    if not os.path.exists("artifacts/trained_pilot.npy") or not os.path.exists("artifacts/nominal_plan.npz"):
        raise FileNotFoundError("Run experiments/03_train_pilot.py first!")

    theta = np.load("artifacts/trained_pilot.npy")
    plan_npz = np.load("artifacts/nominal_plan.npz")
    plan = {"throttle": plan_npz["throttle"], "direction": plan_npz["direction"]}

    rng = np.random.default_rng(2026)
    N = 100

    # -------------------------------------------------------------------------
    # Part A: 100 Dispersed Degradation Scenarios (Fixed Controls, #28 Creswell)
    # -------------------------------------------------------------------------
    onset_hr = rng.uniform(30 * 24, 120 * 24, N)
    severity = rng.uniform(0.10, 0.25, N)
    ramp_hr = rng.uniform(24, 96, N)

    tel = T.make_telemetry(rng, onset_hr, severity, ramp_hr, hours=T.MISSION_HOURS)

    # Onboard Detector 1: Physics-Informed Z-Score (w=24)
    z_det = D.ZScoreDetector(window=24, threshold=3.5, persist=3, baseline_hours=240)
    alarm_z, _, believed_z = z_det.run(tel["thrust_mN"])
    alarm_z_safe = np.where(np.isinf(alarm_z), T.MISSION_HOURS, alarm_z)

    # Onboard Detector 2: Collective Isolation Forest (w=24, #20 Fu et al., #26 Liu et al.)
    if_det = D.IsolationForestDetector(window=24, contamination=0.005, baseline_hours=240, persist=4, seed=42)
    alarm_if, _, believed_if = if_det.run(tel)
    alarm_if_safe = np.where(np.isinf(alarm_if), T.MISSION_HOURS, alarm_if)

    # Strategy 1: Pre-Planned (Open Loop)
    scen_preplanned = S.Scenario(
        onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
        believed=np.ones_like(believed_z), takeover_hour=np.full(N, np.inf)
    )
    res_preplanned = S.fly(theta, scen_preplanned, plan=plan, dt_hours=S.DT_HOURS)

    # Strategy 2: Ground-in-the-Loop (7-day communication + planning delay)
    GROUND_DELAY_HOURS = 7 * 24.0
    scen_ground = S.Scenario(
        onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
        believed=believed_if, takeover_hour=alarm_if_safe + GROUND_DELAY_HOURS
    )
    res_ground = S.fly(theta, scen_ground, plan=plan, dt_hours=S.DT_HOURS)

    # Strategy 3: Autonomous Onboard (Z-Score + Neural Pilot)
    scen_onboard_z = S.Scenario(
        onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
        believed=believed_z, takeover_hour=alarm_z_safe
    )
    res_onboard_z = S.fly(theta, scen_onboard_z, plan=plan, dt_hours=S.DT_HOURS)

    # Strategy 4: Autonomous Onboard (Collective Isolation Forest + Neural Pilot)
    scen_onboard_if = S.Scenario(
        onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
        believed=believed_if, takeover_hour=alarm_if_safe
    )
    res_onboard_if = S.fly(theta, scen_onboard_if, plan=plan, dt_hours=S.DT_HOURS)

    def summarize(name, res):
        miss = res["min_dist_km"]
        cap = res["captured"]
        fuel = res["fuel_used_kg"]
        return {
            "Strategy": name,
            "Capture_Rate_Pct": float(np.mean(cap) * 100),
            "Median_Miss_km": float(np.median(miss)),
            "Mean_Miss_km": float(np.mean(miss)),
            "Std_Miss_km": float(np.std(miss)),
            "Mean_Fuel_kg": float(np.mean(fuel)),
            "Std_Fuel_kg": float(np.std(fuel))
        }

    summary_rows = [
        summarize("1. Pre-Planned (No Adaptation)", res_preplanned),
        summarize("2. Ground-in-the-Loop (7-day delay)", res_ground),
        summarize("3. Onboard Z-Score + Pilot", res_onboard_z),
        summarize("4. Onboard Isolation Forest + Pilot", res_onboard_if),
    ]
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv("artifacts/gap_experiment_results.csv", index=False)
    print("\n=== Part A: Experimental Results across 100 Dispersed Scenarios (#28) ===")
    print(df_summary.to_string(index=False))

    df_runs = pd.DataFrame({
        "Scenario": np.arange(N),
        "Onset_Hour": onset_hr,
        "Severity_Pct": severity * 100,
        "Alarm_Z_Hour": alarm_z,
        "Alarm_IF_Hour": alarm_if,
        "Miss_Preplanned_km": res_preplanned["min_dist_km"],
        "Miss_Ground_km": res_ground["min_dist_km"],
        "Miss_Onboard_Z_km": res_onboard_z["min_dist_km"],
        "Miss_Onboard_IF_km": res_onboard_if["min_dist_km"],
    })
    df_runs.to_csv("artifacts/gap_experiment_runs.csv", index=False)

    # -------------------------------------------------------------------------
    # Part B: Factorial Experiment (Controller x Degradation Severity, #29)
    # -------------------------------------------------------------------------
    sev_levels = [0.10, 0.15, 0.20, 0.25]
    n_per_level = 25
    onset_grid = np.linspace(35 * 24, 110 * 24, n_per_level)
    ramp_fixed = np.full(n_per_level, 48.0)

    factorial_rows = []
    for sev_lvl in sev_levels:
        rng_f = np.random.default_rng(1000 + int(sev_lvl * 100))
        sev_arr = np.full(n_per_level, sev_lvl)
        tel_f = T.make_telemetry(rng_f, onset_grid, sev_arr, ramp_fixed, hours=T.MISSION_HOURS)
        al_if, _, bel_if = if_det.run(tel_f)
        al_safe = np.where(np.isinf(al_if), T.MISSION_HOURS, al_if)

        # Pre-planned (never switches to closed-loop pilot)
        r_pre = S.fly(theta, S.Scenario(onset_hour=onset_grid, severity=sev_arr, ramp_hours=ramp_fixed,
                                        believed=np.ones_like(bel_if), takeover_hour=np.full(n_per_level, np.inf)),
                      plan=plan, dt_hours=S.DT_HOURS)
        # Ground 7-day delay (DSN tracking + ground trajectory re-computation)
        r_gnd = S.fly(theta, S.Scenario(onset_hour=onset_grid, severity=sev_arr, ramp_hours=ramp_fixed,
                                        believed=bel_if, takeover_hour=al_safe + 7 * 24.0),
                      plan=plan, dt_hours=S.DT_HOURS)
        # Onboard Isolation Forest + Pilot (immediate takeover upon alarm)
        r_onb = S.fly(theta, S.Scenario(onset_hour=onset_grid, severity=sev_arr, ramp_hours=ramp_fixed,
                                        believed=bel_if, takeover_hour=al_safe),
                      plan=plan, dt_hours=S.DT_HOURS)

        for label, r_curr in [
            ("Pre-Planned", r_pre),
            ("Ground-Delayed (7d)", r_gnd),
            ("Onboard (IF + Pilot)", r_onb),
        ]:
            factorial_rows.append({
                "Degradation_Pct": int(sev_lvl * 100),
                "Controller": label,
                "Median_Miss_km": float(np.median(r_curr["min_dist_km"])),
                "Mean_Miss_km": float(np.mean(r_curr["min_dist_km"])),
                "Mean_Fuel_kg": float(np.mean(r_curr["fuel_used_kg"])),
            })

    df_fact = pd.DataFrame(factorial_rows)
    df_fact.to_csv("artifacts/factorial_experiment_results.csv", index=False)
    print("\n=== Part B: Two-Way Factorial Experiment (Controller x Degradation Level, #29) ===")
    print(df_fact.to_string(index=False))

    # -------------------------------------------------------------------------
    # Publication Visualization (2 Panels: Boxplot + Factorial Interaction)
    # -------------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    box_data = [
        res_preplanned["min_dist_km"] / 1e6,
        res_ground["min_dist_km"] / 1e6,
        res_onboard_z["min_dist_km"] / 1e6,
        res_onboard_if["min_dist_km"] / 1e6,
    ]
    labels = ["Pre-Planned\n(No Adapt)", "Ground Delay\n(7-day)", "Onboard\n(Z-Score)", "Onboard\n(IsoForest)"]
    bp = ax1.boxplot(box_data, tick_labels=labels, patch_artist=True,
                     medianprops=dict(color="black", linewidth=1.5))
    colors = ["#e74c3c", "#f39c12", "#3498db", "#2ecc71"]
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.75)

    ax1.axhline(P.CAPTURE_RADIUS_KM / 1e6, color="red", linestyle="--", label="Mars Capture Bound (0.577M km)")
    ax1.set_ylabel("Final Miss Distance (Million km)")
    ax1.set_title("A. Controlled Comparison Across 100 Scenarios (#28)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="upper right")

    # Factorial Interaction Plot (#29 Leedy & Ormrod)
    styles = {
        "Pre-Planned": ("#e74c3c", "o-", "Pre-Planned (No Adaptation)"),
        "Ground-Delayed (7d)": ("#f39c12", "s--", "Ground-in-the-Loop (7d Delay)"),
        "Onboard (IF + Pilot)": ("#2ecc71", "^-", "Autonomous Onboard (IF + Pilot)"),
    }
    for ctrl, (col, fmt, lbl) in styles.items():
        sub = df_fact[df_fact["Controller"] == ctrl]
        ax2.plot(sub["Degradation_Pct"], sub["Mean_Miss_km"] / 1e6, fmt,
                 color=col, linewidth=2.2, markersize=7, label=lbl)

    ax2.axhline(P.CAPTURE_RADIUS_KM / 1e6, color="red", linestyle=":", label="Capture Bound (0.577M km)")
    ax2.set_xlabel("Thruster Degradation Severity (%)")
    ax2.set_ylabel("Mean Miss Distance (Million km)")
    ax2.set_xticks([10, 15, 20, 25])
    ax2.set_title("B. Factorial Interaction: Controller × Degradation (#29)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="upper left", fontsize=8.5)

    plt.tight_layout()
    plt.savefig("artifacts/figures/gap_experiment_comparison.png", dpi=300)
    plt.close()
    print("Saved plot to artifacts/figures/gap_experiment_comparison.png\n")


if __name__ == "__main__":
    run()
