"""
Experiment 7: how much of the controlled comparison is the training draw?

Experiment 4 compares four controllers on 100 fixed fault scenarios, but the
guidance policy and the nominal plan it records are each a single training
draw.  This script re-trains the policy from four further seeds, then flies
every policy against every recorded plan on the same 100 scenarios (25 cells)
and writes the table cited in Appendix A:

    artifacts/training_draw_sensitivity.csv

The headline run (seed 42) is read from artifacts/trained_pilot.npy and
artifacts/nominal_plan.npz, so this script never rewrites the committed
headline artifacts and reproduces their row exactly.

Seed schedule for the extra draws, matching the sweep described in the paper:
policy initialisation default_rng(s), Stage 1 ES seed 1000+s, Stage 2 ES seed
2000+s, and the fixed ten-scenario validation set default_rng(999) from
Experiment 3.  The recipe itself is mission.trainer.train_policy, shared with
Experiment 3 so the sweep cannot drift from the headline run.

Runtime is about six minutes on a laptop CPU (four trainings, ~80 s each).
Trained draws are cached in --draws so a re-run only re-flies the table.

Usage:
    python experiments/07_training_draw_sensitivity.py
    python experiments/07_training_draw_sensitivity.py --draws /tmp/draws
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np
import pandas as pd

from mission import thruster as T
from mission import detector as D
from mission import simulate as S
from mission import trainer as ES

HEADLINE_SEED = 42
EXTRA_SEEDS = (43, 44, 45, 46)


def train_draw(seed, outdir):
    """Train one draw with the shared two-stage recipe and cache it.

    Only the seed schedule differs from the headline run in Experiment 3.
    """
    theta, plan, _, _ = ES.train_policy(
        seed=seed, stage1_seed=1000 + seed, stage2_seed=2000 + seed)
    np.save(os.path.join(outdir, f"trained_pilot_{seed}.npy"), theta)
    np.savez(os.path.join(outdir, f"nominal_plan_{seed}.npz"),
             throttle=plan["throttle"], direction=plan["direction"])


def load_draws(draws_dir):
    """Return {seed: (theta, plan)} for the headline run and the extra draws."""
    draws = {}
    draws[HEADLINE_SEED] = (
        np.load(os.path.join(ROOT, "artifacts", "trained_pilot.npy")),
        dict(np.load(os.path.join(ROOT, "artifacts", "nominal_plan.npz"))))
    for seed in EXTRA_SEEDS:
        theta_path = os.path.join(draws_dir, f"trained_pilot_{seed}.npy")
        plan_path = os.path.join(draws_dir, f"nominal_plan_{seed}.npz")
        if not (os.path.exists(theta_path) and os.path.exists(plan_path)):
            print(f"seed {seed}: training (about 80 s)")
            train_draw(seed, draws_dir)
        else:
            print(f"seed {seed}: reusing trained draw in {draws_dir}")
        z = np.load(plan_path)
        draws[seed] = (np.load(theta_path),
                       {"throttle": z["throttle"], "direction": z["direction"]})
    return draws


def crossed_table(draws):
    """Fly every policy against every plan on Experiment 4's 100 scenarios."""
    rng = np.random.default_rng(2026)
    N = 100
    onset_hr = rng.uniform(30 * 24, 120 * 24, N)
    severity = rng.uniform(0.10, 0.25, N)
    ramp_hr = rng.uniform(24, 96, N)
    tel = T.make_telemetry(rng, onset_hr, severity, ramp_hr, hours=T.MISSION_HOURS)

    z_det = D.ZScoreDetector(window=24, threshold=3.5, persist=3, baseline_hours=240)
    alarm_z, _, believed_z = z_det.run(tel["thrust_mN"])
    alarm_z = np.where(np.isinf(alarm_z), T.MISSION_HOURS, alarm_z)

    if_det = D.IsolationForestDetector(window=24, contamination=0.005,
                                       baseline_hours=240, persist=4, seed=42)
    alarm_if, _, believed_if = if_det.run(tel)
    alarm_if = np.where(np.isinf(alarm_if), T.MISSION_HOURS, alarm_if)

    seeds = [HEADLINE_SEED] + list(EXTRA_SEEDS)
    rows = []
    for pi in seeds:
        plan = draws[pi][1]
        pre = S.fly(draws[pi][0], S.Scenario(
            onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
            believed=np.ones_like(believed_if), takeover_hour=np.full(N, np.inf)),
            plan=plan, dt_hours=S.DT_HOURS)
        for pj in seeds:
            theta = draws[pj][0]
            gnd = S.fly(theta, S.Scenario(
                onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
                believed=believed_if, takeover_hour=alarm_if + 7 * 24.0),
                plan=plan, dt_hours=S.DT_HOURS)
            onb = S.fly(theta, S.Scenario(
                onset_hour=onset_hr, severity=severity, ramp_hours=ramp_hr,
                believed=believed_if, takeover_hour=alarm_if),
                plan=plan, dt_hours=S.DT_HOURS)
            rows.append({
                "plan_seed": pi,
                "policy_seed": pj,
                "matched": pi == pj,
                "pre_planned_median_km": int(round(np.median(pre["min_dist_km"]))),
                "ground_median_km": int(round(np.median(gnd["min_dist_km"]))),
                "onboard_median_km": int(round(np.median(onb["min_dist_km"]))),
                "onboard_minus_ground_paired_median_km":
                    int(round(np.median(onb["min_dist_km"] - gnd["min_dist_km"]))),
                "ground_better_pct":
                    round(float(np.mean(gnd["min_dist_km"] < onb["min_dist_km"]) * 100), 1),
                "ground_capture_pct": round(float(np.mean(gnd["captured"]) * 100), 1),
                "onboard_capture_pct": round(float(np.mean(onb["captured"]) * 100), 1),
            })
            print(f"  plan {pi} / policy {pj}: pre {rows[-1]['pre_planned_median_km']/1e6:.3f}  "
                  f"ground {rows[-1]['ground_median_km']/1e6:.3f}  "
                  f"onboard {rows[-1]['onboard_median_km']/1e6:.3f} M km", flush=True)
    return pd.DataFrame(rows)


def variance_shares(df, col):
    """Two-way sums of squares over the 5x5 layout, one observation per cell."""
    y = df[col].values.reshape(5, 5).astype(float)
    grand = y.mean()
    ss_plan = 5 * np.sum((y.mean(axis=1) - grand) ** 2)
    ss_policy = 5 * np.sum((y.mean(axis=0) - grand) ** 2)
    ss_total = np.sum((y - grand) ** 2)
    return (100 * ss_plan / ss_total, 100 * ss_policy / ss_total,
            100 * (ss_total - ss_plan - ss_policy) / ss_total)


def run(draws_dir, out_path):
    print("=== Experiment 7: Training-Draw Sensitivity of the Controller Comparison ===")
    draws = load_draws(draws_dir)
    df = crossed_table(draws)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df)} plan x policy cells)")

    matched = df[df.matched]
    print("\nMatched draws (plan seed = policy seed):")
    for _, r in matched.iterrows():
        red = 100 * (1 - r.onboard_median_km / r.pre_planned_median_km)
        print(f"  {int(r.plan_seed)}: pre {r.pre_planned_median_km/1e6:.3f}  "
              f"ground {r.ground_median_km/1e6:.3f}  onboard {r.onboard_median_km/1e6:.3f} M km "
              f"| onboard reduction {red:.1f}%")

    print("\nGround ahead in %d of %d cells" % (int((df.onboard_minus_ground_paired_median_km > 0).sum()), len(df)))
    for col, label in [("ground_median_km", "ground level"),
                       ("onboard_median_km", "onboard level"),
                       ("onboard_minus_ground_paired_median_km", "ground-onboard gap")]:
        plan, pol, res = variance_shares(df, col)
        print(f"Variance of {label}: plan {plan:.1f}%, policy {pol:.1f}%, residual {res:.1f}%")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--draws", default=os.path.join(ROOT, "artifacts", "training_draws"),
                    help="where trained draws are cached (default: artifacts/training_draws)")
    ap.add_argument("--out", default=os.path.join(ROOT, "artifacts", "training_draw_sensitivity.csv"),
                    help="output table (default: artifacts/training_draw_sensitivity.csv)")
    args = ap.parse_args()
    os.makedirs(args.draws, exist_ok=True)
    run(args.draws, args.out)
