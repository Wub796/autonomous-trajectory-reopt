"""
Experiment 2: Onboard Anomaly Detection Benchmark
Tests Chandola et al. (2009) [#31], Fu et al. (2022) [#20], and Liu et al. (2008) [#26]:
  1. Point-Only Threshold (w=1): Looks at single hourly readings in isolation (#31).
  2. Physics-Informed Z-Score (w=24): Rolling-window statistical residual filter.
  3. Collective Isolation Forest (w=24): Rolling-window feature extraction (#20) +
     unsupervised random binary decision trees (#26).
Evaluates True/False Positives, Precision, Recall, F1, and Detection Latency.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mission import thruster as T
from mission import detector as D

os.makedirs("artifacts/figures", exist_ok=True)


def evaluate_detector(name, deg_alarm, nom_alarm, onset_hr):
    N = len(onset_hr)
    delays = deg_alarm - onset_hr
    tp = int(np.sum(~np.isinf(deg_alarm) & (deg_alarm >= onset_hr)))
    fn = int(N - tp)
    fp = int(np.sum(~np.isinf(nom_alarm)))
    tn = int(N - fp)

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    valid_delays = delays[~np.isinf(delays) & (delays >= 0)]
    mean_delay = float(np.mean(valid_delays)) if len(valid_delays) > 0 else np.nan

    return {
        "Model": name, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": prec, "Recall": rec, "F1": f1, "Mean_Delay_Hours": mean_delay
    }


def run():
    print("=== Experiment 2: Anomaly Detection Benchmark (#20, #26, #31) ===")
    rng = np.random.default_rng(42)

    # 1. Synthesize 50 nominal runs and 50 degraded runs (#15 Ballentine et al., 2026)
    N = 50
    onset_hr = rng.uniform(40 * 24, 150 * 24, N)
    sev = rng.uniform(0.10, 0.25, N)
    ramp_hr = rng.uniform(24, 120, N)

    tel_deg = T.make_telemetry(rng, onset_hr, sev, ramp_hr, hours=T.MISSION_HOURS)
    tel_nom = T.make_telemetry(rng, np.inf, 0.0, 0.0, hours=T.MISSION_HOURS)

    # 2. Benchmark Point-Only Isolation Forest (w=1, no window features - #31 Chandola et al.)
    pt_det = D.PointIsolationForestDetector(contamination=0.01, baseline_hours=240, persist=3, seed=42)
    deg_alarm_pt, _, _ = pt_det.run(tel_deg)
    nom_alarm_pt, _, _ = pt_det.run(tel_nom)

    # 3. Benchmark Physics-Informed Z-Score Detector (w=24)
    z_det = D.ZScoreDetector(window=24, threshold=3.5, persist=3, baseline_hours=240)
    deg_alarm_z, _, _ = z_det.run(tel_deg["thrust_mN"])
    nom_alarm_z, _, _ = z_det.run(tel_nom["thrust_mN"])

    # 4. Benchmark Collective-Window Isolation Forest (w=24 - #20 Fu et al., #26 Liu et al.)
    if_det = D.IsolationForestDetector(window=24, contamination=0.005, baseline_hours=240, persist=4, seed=42)
    deg_alarm_if, _, _ = if_det.run(tel_deg)
    nom_alarm_if, _, _ = if_det.run(tel_nom)

    results = [
        evaluate_detector("1. Point-Only Isolation Forest (w=1)", deg_alarm_pt, nom_alarm_pt, onset_hr),
        evaluate_detector("2. Physics-Informed Z-Score (w=24)", deg_alarm_z, nom_alarm_z, onset_hr),
        evaluate_detector("3. Collective Isolation Forest (w=24)", deg_alarm_if, nom_alarm_if, onset_hr),
    ]
    df_res = pd.DataFrame(results)
    df_res.to_csv("artifacts/anomaly_metrics.csv", index=False)
    print("\nAnomaly Detection Results:")
    print(df_res.to_string(index=False))

    # 5. Visualization: Single Scenario Telemetry & Detector Response
    single_tel = T.make_telemetry(rng, 1000.0, 0.18, 48.0, hours=3000)
    single_alarm_z, _, believed_z = z_det.run(single_tel["thrust_mN"])
    single_alarm_if, _, believed_if = if_det.run(single_tel)

    hours_plt = np.arange(3000)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top: Telemetry stream + health estimate
    ax1.plot(hours_plt, single_tel["thrust_mN"][0], color="lightgray", label="Noisy Sensor Telemetry (1% noise)")
    ax1.plot(hours_plt, single_tel["true_frac"][0] * T.T0 * 1000, "r-", label="True Thrust (18% Gradual Decay at h=1000)")
    ax1.plot(hours_plt, believed_if[0] * T.T0 * 1000, "b--", label="Onboard Believed Thrust (Post-Alarm)")
    ax1.axvline(1000.0, color="orange", linestyle=":", linewidth=1.8, label="True Degradation Onset (h=1000)")
    if not np.isinf(single_alarm_if[0]):
        ax1.axvline(single_alarm_if[0], color="green", linestyle="--", linewidth=1.8,
                    label=f"Isolation Forest Alarm (h={single_alarm_if[0]:.0f})")
    if not np.isinf(single_alarm_z[0]):
        ax1.axvline(single_alarm_z[0], color="purple", linestyle="-.", linewidth=1.5,
                    label=f"Z-Score Alarm (h={single_alarm_z[0]:.0f})")
    ax1.set_ylabel("Thrust (mN)")
    ax1.set_title("Collective Anomaly Detection on Gradual Thruster Decay (#20, #26, #31)")
    ax1.legend(loc="lower left", fontsize=8.5)
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Bottom: Rolling Z-Score & Collective Drift Profile
    z_scores = z_det.scores(single_tel["thrust_mN"])[0]
    ax2.plot(hours_plt, z_scores, color="indigo", label="24-Hour Rolling Z-Score")
    ax2.axhline(-3.5, color="red", linestyle="--", label="Alarm Threshold = -3.5")
    ax2.set_xlabel("Mission Elapsed Time (hours)")
    ax2.set_ylabel("Collective Drift (Std Devs)")
    ax2.set_ylim(-15, 5)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig("artifacts/figures/anomaly_detection_analysis.png", dpi=300)
    plt.close()
    print("Saved plot to artifacts/figures/anomaly_detection_analysis.png\n")


if __name__ == "__main__":
    run()
