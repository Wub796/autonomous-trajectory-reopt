"""
anomaly_detection_analysis.py — Formal anomaly-detection metrics for the onboard
Isolation Forest thruster-health monitor used in the curriculum env
(src.env.spacecraft_env / src.env.robust_spacecraft_env).

Generates the data needed for the paper's Section 3.3 (Data Collection) and
Section 4 (Results):

  1. Labeled confusion-matrix evaluation (TP / FP / TN / FN), precision, recall,
     F1-score for the Isolation Forest on synthetic-but-physically-consistent
     telemetry ([Isp, solar temperature] features, as consumed by the env).
  2. ROC curve data (TPR / FPR / AUC) via score_samples / decision_function.
  3. Detection-threshold analysis (F1 / FPR / detection-delay vs. threshold).
  4. Comparison against alternative one-class detectors
     (One-Class SVM, Local Outlier Factor, Elliptic Envelope).
  5. Full-mission detection profile reproducing the paper's "no false positives
     during the first N hours, detection at hour M" analysis (configured onset /
     degradation rate), including the detection-delay and false-positive counts.

Outputs (written to artifacts/anomaly_detection/):
  anomaly_metrics.csv, anomaly_roc_data.csv, anomaly_threshold_analysis.csv,
  anomaly_mission_detection.csv, anomaly_detection_report.md,
  figures/anomaly_roc.png, figures/anomaly_threshold_analysis.png,
  figures/anomaly_mission_profile.png

Usage:
    PYTHONPATH=. .venv/bin/python scripts/anomaly_detection_analysis.py
    PYTHONPATH=. .venv/bin/python scripts/anomaly_detection_analysis.py --quick-test
"""
import os
import argparse
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_fscore_support

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")
_OUT_DIR = os.path.join(_ARTIFACTS_DIR, "anomaly_detection")
_FIG_DIR = os.path.join(_OUT_DIR, "figures")

# ---------------------------------------------------------------------------
# Physics-consistent telemetry synthesis
# ---------------------------------------------------------------------------

# Nominal thruster operating point (SPT-140, Hargus & Fife; env default)
ISP_NOMINAL = 1782.0       # s
ISP_FAILED = 1514.7        # s (env Isp_failed)
SOLAR_TEMP_NOMINAL = 45.0  # arbitrary units matching env anomaly injection
SOLAR_TEMP_SIGMA = 0.5

# Log-drift degradation model (Section 2.3 of the paper):  Isp(h) = Isp0 - k * ln(1 + h/tau)
DEGRADATION_K = 18.0       # s per natural-log unit of operating time
DEGRADATION_TAU = 400.0    # characteristic time (h)


def sample_isp_degraded(hours_since_onset: np.ndarray,
                        k: float = DEGRADATION_K,
                        tau: float = DEGRADATION_TAU,
                        rng: np.random.Generator | None = None) -> np.ndarray:
    """Returns degraded Isp (s) for a number of hours since anomaly onset."""
    rng = rng if rng is not None else np.random.default_rng()
    hours = np.maximum(hours_since_onset, 0.0)
    isp = ISP_NOMINAL - k * np.log(1.0 + hours / tau)
    isp = np.clip(isp, ISP_FAILED, ISP_NOMINAL)
    # measurement jitter (instrument noise)
    isp = isp + rng.normal(0.0, 1.2, size=isp.shape)
    return isp


def make_labeled_dataset(n_nominal: int = 6000,
                         n_anomalous: int = 2000,
                         seed: int = 7) -> pd.DataFrame:
    """
    Synthesizes labeled telemetry:  features = [Isp (s), solar temperature].
    Nominal:  Isp ~ N(1782, 2), temp ~ N(45, 0.5).
    Anomalous: Isp sampled from the log-degradation curve at random operating
    ages (occasionally compounded by a thermal excursion), temp ~ N(45, 0.5).
    """
    rng = np.random.default_rng(seed)

    nominal_isp = rng.normal(ISP_NOMINAL, 2.0, size=n_nominal)
    nominal_temp = rng.normal(SOLAR_TEMP_NOMINAL, SOLAR_TEMP_SIGMA, size=n_nominal)

    hours = rng.uniform(0.0, 16000.0, size=n_anomalous)
    anom_isp = sample_isp_degraded(hours, rng=rng)
    # ~10% of anomalies include a solar thermal excursion
    thermal = rng.random(n_anomalous) < 0.10
    anom_temp = rng.normal(SOLAR_TEMP_NOMINAL, SOLAR_TEMP_SIGMA, size=n_anomalous)
    anom_temp[thermal] += rng.normal(8.0, 2.0, size=int(thermal.sum()))

    X = np.vstack([
        np.column_stack([nominal_isp, nominal_temp]),
        np.column_stack([anom_isp, anom_temp]),
    ])
    y = np.concatenate([np.zeros(n_nominal, dtype=int), np.ones(n_anomalous, dtype=int)])
    return pd.DataFrame(X, columns=["isp_s", "solar_temp"]).assign(label=y)


def _split(df: pd.DataFrame, seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    n_train = int(0.7 * len(df))
    return df.iloc[idx[:n_train]], df.iloc[idx[n_train:]]


# ---------------------------------------------------------------------------
# Detector zoo
# ---------------------------------------------------------------------------

def fit_detectors(X_train: np.ndarray) -> dict:
    """Trains the Isolation Forest plus three comparison detectors."""
    models = {}
    models["Isolation Forest"] = IsolationForest(
        n_estimators=200, contamination=0.10, random_state=0, n_jobs=-1
    ).fit(X_train)
    # One-Class SVM is trained on a subsample for tractability
    n_sub = min(2500, len(X_train))
    sub = X_train[np.random.default_rng(1).choice(len(X_train), n_sub, replace=False)]
    models["One-Class SVM"] = OneClassSVM(nu=0.10, kernel="rbf", gamma="scale").fit(sub)
    models["Local Outlier Factor"] = LocalOutlierFactor(
        novelty=True, contamination=0.10, n_neighbors=40
    ).fit(X_train)
    models["Elliptic Envelope"] = EllipticEnvelope(contamination=0.10, random_state=0).fit(X_train)
    return models


def anomaly_score(model, X: np.ndarray) -> np.ndarray:
    """Higher = more anomalous. Uses sklearn's built-in scoring interfaces."""
    if hasattr(model, "score_samples"):
        return -model.score_samples(X)
    return -model.decision_function(X)


def evaluate_detector(model, X_test: np.ndarray, y_test: np.ndarray,
                      threshold: float | None = None,
                      pos_label: int = 1) -> dict:
    """
    Evaluates a detector at a given anomaly-score threshold.
    Defaults to the sklearn decision threshold (predict == -1 means anomaly).
    """
    scores = anomaly_score(model, X_test)
    if threshold is None:
        pred = np.where(model.predict(X_test) == -1, 1, 0)
    else:
        pred = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, pred, labels=[1],
                                                       zero_division=0)
    return {
        "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn),
        "precision": float(prec[0]), "recall": float(rec[0]), "f1": float(f1[0]),
        "threshold": threshold,
    }


# ---------------------------------------------------------------------------
# Mission-level detection profile
# ---------------------------------------------------------------------------

def run_mission_detection_profile(model, onset_hour: float = 1000.0,
                                  mission_hours: int = 11040,
                                  seed: int = 3) -> pd.DataFrame:
    """
    Simulates the onboard health monitor sampling [Isp, solar temp] once per
    hour over a full 460-day mission. Anomaly onset occurs at `onset_hour`;
    before onset the thruster is nominal. Reports the class, detector score and
    flag for every hour so false positives (pre-onset) and detection delay
    (post-onset) can be computed from raw data.
    """
    rng = np.random.default_rng(seed)
    hours = np.arange(mission_hours)

    isp = rng.normal(ISP_NOMINAL, 1.2, size=mission_hours)
    degraded = hours >= onset_hour
    isp[degraded] = sample_isp_degraded(hours[degraded] - onset_hour, rng=rng)
    temp = rng.normal(SOLAR_TEMP_NOMINAL, SOLAR_TEMP_SIGMA, size=mission_hours)

    X = np.column_stack([isp, temp])
    scores = anomaly_score(model, X)
    flags = model.predict(X) == -1

    return pd.DataFrame({
        "hour": hours,
        "isp_s": isp,
        "solar_temp": temp,
        "anomaly_present": degraded.astype(int),
        "anomaly_score": scores,
        "detector_flag": flags.astype(int),
    })


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def plot_roc(roc_data: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 7))
    for method, grp in roc_data.groupby("method"):
        ax.plot(grp["fpr"], grp["tpr"], label=f"{method} (AUC={grp['auc'].iloc[0]:.3f})", lw=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Anomaly Detector ROC Curves — Thruster Health Monitor")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_threshold_analysis(threshold_df: pd.DataFrame, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    t = threshold_df["threshold"]
    axes[0].plot(t, threshold_df["precision"], label="Precision", color="#d62728")
    axes[0].plot(t, threshold_df["recall"], label="Recall", color="#1f77b4")
    axes[0].plot(t, threshold_df["f1"], label="F1", color="#2ca02c", lw=2.5)
    axes[0].set_xlabel("Anomaly-score threshold")
    axes[0].set_ylabel("Score")
    axes[0].set_title("Precision / Recall / F1 vs. Threshold")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(t, threshold_df["fpr"] * 100.0, color="#9467bd", lw=2)
    axes[1].set_xlabel("Anomaly-score threshold")
    axes[1].set_ylabel("False Positive Rate (%)")
    axes[1].set_title("FPR vs. Threshold")
    axes[1].grid(alpha=0.3)

    axes[2].plot(t, threshold_df["detection_delay_h"], color="#ff7f0e", lw=2)
    axes[2].set_xlabel("Anomaly-score threshold")
    axes[2].set_ylabel("Detection delay (h)")
    axes[2].set_title("Mission Detection Delay vs. Threshold")
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_mission_profile(profile: pd.DataFrame, metrics: dict, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    onset = metrics["onset_hour"]

    axes[0].plot(profile["hour"], profile["isp_s"], lw=0.8, color="#1f77b4",
                 label="Measured Isp")
    axes[0].axhline(ISP_NOMINAL, color="k", ls="--", alpha=0.6, label="Nominal Isp")
    axes[0].axhline(ISP_FAILED, color="k", ls=":", alpha=0.6, label="Failed-state floor")
    axes[0].axvline(onset, color="#d62728", ls="--", alpha=0.8, label=f"Degradation onset (h={onset:.0f})")
    if metrics["detection_hour"] is not None:
        axes[0].axvline(metrics["detection_hour"], color="#2ca02c", ls="-", lw=2,
                        label=f"Detector alarm (h={metrics['detection_hour']:.0f})")
    axes[0].set_ylabel("Specific impulse (s)")
    axes[0].set_title(f"Mission Health-Monitor Profile — {metrics['false_positives']} FP pre-onset, "
                      f"detection delay {metrics['detection_delay_h']:.0f} h")
    axes[0].legend(loc="lower left", fontsize=9)
    axes[0].grid(alpha=0.3)

    flag = profile["detector_flag"].to_numpy()
    axes[1].fill_between(profile["hour"], 0, flag, step="mid", color="#2ca02c", alpha=0.8,
                         label="Detector flag (anomaly)")
    axes[1].fill_between(profile["hour"], 0, profile["anomaly_present"], step="mid",
                         color="#d62728", alpha=0.35, label="True anomaly present")
    axes[1].axvline(onset, color="#d62728", ls="--", alpha=0.8)
    axes[1].set_xlabel("Mission hour")
    axes[1].set_ylabel("Flag state")
    axes[1].set_ylim(-0.1, 1.1)
    axes[1].legend(loc="upper left", fontsize=9)
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_anomaly_analysis(quick_test: bool = False, onset_hour: float = 1000.0,
                         n_nominal: int = 6000, n_anomalous: int = 2000) -> None:
    print("=" * 70)
    print(" Anomaly Detection Analysis — Isolation Forest Health Monitor")
    print(" (TP/FP/TN/FN, precision/recall/F1, ROC, threshold & mission profile)")
    print("=" * 70)

    os.makedirs(_FIG_DIR, exist_ok=True)

    if quick_test:
        n_nominal, n_anomalous = 1000, 300
        onset_hour = 1000.0

    # --- 1. Build labeled dataset & partition -------------------------------
    df = make_labeled_dataset(n_nominal=n_nominal, n_anomalous=n_anomalous)
    train, test = _split(df)
    print(f"\n[1/6] Labeled telemetry: {len(train):,} train / {len(test):,} test "
          f"({int(test['label'].sum())} anomalies)")

    X_train, y_train = train[["isp_s", "solar_temp"]].to_numpy(), train["label"].to_numpy()
    X_test, y_test = test[["isp_s", "solar_temp"]].to_numpy(), test["label"].to_numpy()

    # --- 2. Train / load detectors ------------------------------------------
    print("[2/6] Fitting Isolation Forest + comparison detectors...")
    models = fit_detectors(X_train)

    # --- 3. Confusion-matrix metrics at default thresholds -------------------
    print("[3/6] Evaluating confusion matrices & P/R/F1 at default thresholds...")
    metric_records = []
    for name, m in models.items():
        rec = evaluate_detector(m, X_test, y_test)
        rec["method"] = name
        metric_records.append(rec)
        print(f"      {name:22s} TP={rec['tp']:5d} FP={rec['fp']:4d} TN={rec['tn']:5d} FN={rec['fn']:4d} "
              f"| P={rec['precision']:.3f} R={rec['recall']:.3f} F1={rec['f1']:.3f}")
    df_metrics = pd.DataFrame(metric_records)[
        ["method", "tp", "fp", "tn", "fn", "precision", "recall", "f1", "threshold"]]

    # --- 4. ROC curve data ---------------------------------------------------
    print("[4/6] Computing ROC curve data + AUC...")
    roc_records = []
    for name, m in models.items():
        scores = anomaly_score(m, X_test)
        fpr, tpr, _th = roc_curve(y_test, scores)
        auc_val = auc(fpr, tpr)
        for f, t in zip(fpr, tpr):
            roc_records.append({"method": name, "fpr": f, "tpr": t, "auc": auc_val})
    df_roc = pd.DataFrame(roc_records)

    # --- 5. Detection-threshold analysis (Isolation Forest) ------------------
    print("[5/6] Sweeping detection threshold (precision/recall/F1/FPR/detection delay)...")
    if_model = models["Isolation Forest"]
    all_scores = np.sort(anomaly_score(if_model, X_test))
    th_grid = np.quantile(all_scores, np.linspace(0.02, 0.98, 30))
    th_records = []
    for th in th_grid:
        ev = evaluate_detector(if_model, X_test, y_test, threshold=th)
        # detection delay: run a mission profile and find first post-onset alarm
        prof = run_mission_detection_profile(if_model, onset_hour=onset_hour)
        post = prof[prof["anomaly_present"] == 1]
        first_alarm = post[post["detector_flag"] == 1]
        delay = float(first_alarm["hour"].iloc[0] - onset_hour) if len(first_alarm) > 0 else np.nan
        th_records.append({
            "threshold": th,
            "precision": ev["precision"],
            "recall": ev["recall"],
            "f1": ev["f1"],
            "fpr": ev["fp"] / max(ev["fp"] + ev["tn"], 1),
            "detection_delay_h": delay,
        })
    df_threshold = pd.DataFrame(th_records)

    # --- 6. Full-mission detection profile -----------------------------------
    print(f"[6/6] Full-mission detection profile (onset at hour {onset_hour:.0f})...")
    mission_df = run_mission_detection_profile(if_model, onset_hour=onset_hour)
    pre = mission_df[mission_df["anomaly_present"] == 0]
    post = mission_df[mission_df["anomaly_present"] == 1]
    first_alarm = post[post["detector_flag"] == 1]
    mission_metrics = {
        "onset_hour": onset_hour,
        "false_positives": int(pre["detector_flag"].sum()),
        "true_positives": int(post["detector_flag"].sum()),
        "detection_hour": float(first_alarm["hour"].iloc[0]) if len(first_alarm) > 0 else None,
        "detection_delay_h": float(first_alarm["hour"].iloc[0] - onset_hour) if len(first_alarm) > 0 else np.nan,
        "hours_flagged": int(mission_df["detector_flag"].sum()),
    }
    print(f"      False positives before onset (h<{onset_hour:.0f}): {mission_metrics['false_positives']}")
    if mission_metrics["detection_hour"] is not None:
        print(f"      First alarm at hour {mission_metrics['detection_hour']:.0f} "
              f"(delay {mission_metrics['detection_delay_h']:.0f} h after onset)")
    else:
        print("      No alarm raised after onset — detector failed to trigger")

    # --- Export --------------------------------------------------------------
    df_metrics.to_csv(os.path.join(_OUT_DIR, "anomaly_metrics.csv"), index=False)
    df_roc.to_csv(os.path.join(_OUT_DIR, "anomaly_roc_data.csv"), index=False)
    df_threshold.to_csv(os.path.join(_OUT_DIR, "anomaly_threshold_analysis.csv"), index=False)
    mission_df.to_csv(os.path.join(_OUT_DIR, "anomaly_mission_detection.csv"), index=False)

    plot_roc(df_roc, os.path.join(_FIG_DIR, "anomaly_roc.png"))
    plot_threshold_analysis(df_threshold, os.path.join(_FIG_DIR, "anomaly_threshold_analysis.png"))
    plot_mission_profile(mission_df, mission_metrics, os.path.join(_FIG_DIR, "anomaly_mission_profile.png"))

    # --- Markdown report -----------------------------------------------------
    report_path = os.path.join(_OUT_DIR, "anomaly_detection_report.md")
    with open(report_path, "w") as f:
        f.write("# Onboard Anomaly Detection Analysis Report (Section 3.3 / Section 4)\n\n")
        f.write("Evaluation of the Isolation Forest thruster-health monitor consumed by the RL environment "
                "(`src.env.spacecraft_env` / `src.env.robust_spacecraft_env`). Features: `[Isp (s), solar temperature]`, "
                "sampled once per mission hour; the detector flags `Isp` degradation and thermal excursions.\n\n")
        f.write(f"## 1. Confusion Matrix & P/R/F1 (default decision thresholds)\n\n")
        f.write("| Method | TP | FP | TN | FN | Precision | Recall | F1 |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for _, r in df_metrics.iterrows():
            f.write(f"| {r['method']} | {r['tp']} | {r['fp']} | {r['tn']} | {r['fn']} | "
                    f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |\n")
        f.write("\n## 2. ROC / AUC\n\n")
        for name, grp in df_roc.groupby("method"):
            f.write(f"- **{name}**: AUC = `{grp['auc'].iloc[0]:.4f}`\n")
        f.write(f"\nFull TPR/FPR curves exported to `anomaly_roc_data.csv`.\n\n")
        f.write("## 3. Detection Threshold Analysis (Isolation Forest)\n\n")
        best = df_threshold.loc[df_threshold["f1"].idxmax()]
        f.write(f"- Grid: {len(df_threshold)} thresholds across the anomaly-score quantile range.\n")
        f.write(f"- Max-F1 operating point: threshold = `{best['threshold']:.4f}`, "
                f"F1 = `{best['f1']:.3f}`, precision = `{best['precision']:.3f}`, "
                f"recall = `{best['recall']:.3f}`, FPR = `{best['fpr']*100:.2f}%`, "
                f"detection delay = `{best['detection_delay_h']:.0f} h`.\n")
        f.write("\n## 4. Full-Mission Detection Profile\n\n")
        f.write(f"- Mission length: `{len(mission_df):,} h`; degradation onset at hour `{mission_metrics['onset_hour']:.0f}`.\n")
        f.write(f"- **False positives during nominal phase**: `{mission_metrics['false_positives']}`\n")
        f.write(f"- **First alarm**: hour `{mission_metrics['detection_hour']:.0f}` "
                f"(detection delay `{mission_metrics['detection_delay_h']:.0f} h`)\n"
                if mission_metrics["detection_hour"] is not None else "- **First alarm**: never raised\n")
        f.write(f"- **True positives (post-onset flags)**: `{mission_metrics['true_positives']}` hours\n\n")
        f.write("## 5. Methodology Notes\n\n")
        f.write("- The labeled test set is synthesized from the same physics used by the environment "
                "(nominal `Isp = 1782 s` vs. degraded `Isp -> 1514.7 s` along the log-drift curve "
                "$Isp(h) = Isp_0 - k\\,\\ln(1 + h/\\tau)$, plus occasional thermal excursions).\n")
        f.write("- Replace `make_labeled_dataset()` / `run_mission_detection_profile()` data with flight "
                "or lab telemetry to re-run the same analysis on measured data.\n")
    print(f"\nAnomaly detection analysis complete.")
    print(f"  Metrics CSV : {os.path.join(_OUT_DIR, 'anomaly_metrics.csv')}")
    print(f"  ROC CSV     : {os.path.join(_OUT_DIR, 'anomaly_roc_data.csv')}")
    print(f"  Threshold   : {os.path.join(_OUT_DIR, 'anomaly_threshold_analysis.csv')}")
    print(f"  Mission CSV : {os.path.join(_OUT_DIR, 'anomaly_mission_detection.csv')}")
    print(f"  Report      : {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Anomaly detection metrics for the Isolation Forest health monitor")
    parser.add_argument("--quick-test", action="store_true", help="Small dataset for smoke testing")
    parser.add_argument("--onset-hour", type=float, default=1000.0, help="Degradation onset hour for mission profile")
    parser.add_argument("--n-nominal", type=int, default=6000, help="Nominal samples in labeled set")
    parser.add_argument("--n-anomalous", type=int, default=2000, help="Anomalous samples in labeled set")
    args = parser.parse_args()

    run_anomaly_analysis(quick_test=args.quick_test, onset_hour=args.onset_hour,
                         n_nominal=args.n_nominal, n_anomalous=args.n_anomalous)