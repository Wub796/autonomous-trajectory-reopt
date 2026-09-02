"""
plotting.py — Publication-grade visualization tools for astrodynamics and Monte Carlo dispersion analysis.

Following the presentation standards of:
Capra, Brandonisio, and Lavagna (2022, 2025) and Zavoli & Federici (2021).
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any

# Set publication style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300,
})


def plot_monte_carlo_dispersions_2d(
    recorded_trajectories: list[list[dict]],
    output_path: str,
) -> None:
    """
    Plots the 2D heliocentric X-Y trajectory dispersion cone.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 8))

    # Sun at origin
    ax.plot(0, 0, marker="o", color="#FFA500", markersize=14, label="Sun (Heliocenter)", zorder=10)

    # Earth orbit circle approx (~1.0 AU = 1.496e8 km)
    theta = np.linspace(0, 2 * np.pi, 200)
    r_earth = 1.496e8
    r_mars = 2.279e8
    ax.plot(r_earth * np.cos(theta) / 1e6, r_earth * np.sin(theta) / 1e6, "k--", alpha=0.3, label="Earth Orbit (1.0 AU)")
    ax.plot(r_mars * np.cos(theta) / 1e6, r_mars * np.sin(theta) / 1e6, "r--", alpha=0.3, label="Mars Orbit (1.52 AU)")

    # Plot dispersed trajectories
    for i, traj in enumerate(recorded_trajectories):
        xs = [pt["x_km"] / 1e6 for pt in traj]
        ys = [pt["y_km"] / 1e6 for pt in traj]
        label = "Dispersed Trajectories" if i == 0 else None
        ax.plot(xs, ys, color="#1f77b4", alpha=0.4, linewidth=1.2, label=label)

    # Final arrival points
    if len(recorded_trajectories) > 0:
        final_xs = [traj[-1]["x_km"] / 1e6 for traj in recorded_trajectories]
        final_ys = [traj[-1]["y_km"] / 1e6 for traj in recorded_trajectories]
        ax.scatter(final_xs, final_ys, color="#d62728", s=25, zorder=5, label="Arrival Endpoints")

    ax.set_xlabel("Heliocentric X ($10^6$ km)")
    ax.set_ylabel("Heliocentric Y ($10^6$ km)")
    ax.set_title("Monte Carlo Trajectory Dispersion Envelopes (Zavoli & Federici 2021 Disturbances)")
    ax.legend(loc="upper right", frameon=True)
    ax.set_aspect("equal")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_monte_carlo_histograms(
    df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plots multi-panel Monte Carlo statistical distribution histograms.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 1. Final Miss Distance
    miss_km_m = df["final_mars_dist_km"] / 1e6
    mean_miss = np.mean(miss_km_m)
    p95_miss = np.percentile(miss_km_m, 95)

    axes[0].hist(miss_km_m, bins=25, color="#2b5c8f", edgecolor="black", alpha=0.8)
    axes[0].axvline(mean_miss, color="red", linestyle="--", linewidth=2, label=f"Mean: {mean_miss:.2f}M km")
    axes[0].axvline(p95_miss, color="orange", linestyle=":", linewidth=2, label=f"95th %ile: {p95_miss:.2f}M km")
    axes[0].set_xlabel("Final Mars Distance ($10^6$ km)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Terminal Miss Distance Distribution")
    axes[0].legend()

    # 2. Propellant Consumed
    fuels = df["fuel_consumed_kg"]
    mean_fuel = np.mean(fuels)
    axes[1].hist(fuels, bins=25, color="#2ca02c", edgecolor="black", alpha=0.8)
    axes[1].axvline(mean_fuel, color="black", linestyle="--", linewidth=2, label=f"Mean: {mean_fuel:.1f} kg")
    axes[1].set_xlabel("Propellant Consumed (kg)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Fuel Expenditure Distribution")
    axes[1].legend()

    # 3. Terminal Relative Velocity
    rel_vel = df["final_rel_vel_km_s"]
    mean_vel = np.mean(rel_vel)
    axes[2].hist(rel_vel, bins=25, color="#9467bd", edgecolor="black", alpha=0.8)
    axes[2].axvline(mean_vel, color="black", linestyle="--", linewidth=2, label=f"Mean: {mean_vel:.2f} km/s")
    axes[2].set_xlabel("Terminal Relative Velocity (km/s)")
    axes[2].set_ylabel("Frequency")
    axes[2].set_title("Arrival Relative Velocity Dispersion")
    axes[2].legend()

    plt.suptitle("Monte Carlo Robustness Analysis — Statistical Metric Distributions", fontsize=15, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_sensitivity_tornado(
    df_sweep: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Plots a tornado sensitivity chart comparing the impact of various disturbance types.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Calculate max deviation per category
    cat_summary = (
        df_sweep.groupby("parameter_category")["delta_from_baseline_km"]
        .apply(lambda s: float(np.max(np.abs(s))) / 1e6)
        .reset_index()
    )
    cat_summary = cat_summary.sort_values(by="delta_from_baseline_km", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        cat_summary["parameter_category"],
        cat_summary["delta_from_baseline_km"],
        color="#d95f02",
        edgecolor="black",
        alpha=0.85,
        height=0.55,
    )

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height() / 2, f"{w:.2f}M km", va="center", ha="left", fontsize=10)

    ax.set_xlabel("Max Increase in Terminal Miss Distance ($10^6$ km)")
    ax.set_title("Guidance Sensitivity Tornado Chart (Capra et al. 2022 Perturbations)")
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_scenario_comparison(
    df_scenario: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Violin of reward & fuel dispersion and time-to-closest-approach across
    uncertainty scenarios (deterministic / mild / zavoli / severe) — the
    distribution data called out in Section 5.4.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scenarios = df_scenario["scenario"].unique().tolist()

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    panels = [
        (0, "fuel_consumed_kg", "Propellant Consumption", "Fuel (kg)", "#2ca02c"),
        (1, "total_reward", "Episode Reward Distribution", "Cumulative reward", "#1f77b4"),
    ]
    for i, col, title, ylab, color in panels:
        data = [df_scenario[df_scenario["scenario"] == s][col].values for s in scenarios]
        parts = axes[i].violinplot(data, showmeans=True, showmedians=True)
        for body in parts["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.5)
        axes[i].set_xticks(range(1, len(scenarios) + 1))
        axes[i].set_xticklabels(scenarios, rotation=20, ha="right")
        axes[i].set_ylabel(ylab)
        axes[i].set_title(title)
        axes[i].grid(alpha=0.3, axis="y")

    # 3. Time-to-closest-approach (convergence) box plot
    conv_data = [df_scenario[df_scenario["scenario"] == s]["convergence_step"].values for s in scenarios]
    axes[2].boxplot(conv_data, showmeans=True)
    axes[2].set_xticks(range(1, len(scenarios) + 1))
    axes[2].set_xticklabels(scenarios, rotation=20, ha="right")
    axes[2].set_ylabel("Time to closest approach (h)")
    axes[2].set_title("Time-to-Convergence per Scenario")
    axes[2].grid(alpha=0.3, axis="y")

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_failure_mode_breakdown(
    df_scenario: pd.DataFrame,
    output_path: str,
) -> None:
    """Stacked-bar chart of intercept / fuel-exhaustion / time-expiry outcomes."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    scenarios = df_scenario["scenario"].unique().tolist()
    reasons = sorted(df_scenario["failure_reason"].unique().tolist())
    counts = np.zeros((len(scenarios), len(reasons)))
    for i, s in enumerate(scenarios):
        sub = df_scenario[df_scenario["scenario"] == s]
        for j, r in enumerate(reasons):
            counts[i, j] = int((sub["failure_reason"] == r).sum())

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bottom = np.zeros(len(scenarios))
    colors = ["#2ca02c", "#d62728", "#ff7f0e"]
    for j, r in enumerate(reasons):
        ax.bar(scenarios, counts[:, j], bottom=bottom, label=r,
               color=colors[j % len(colors)], edgecolor="black", alpha=0.85)
        bottom += counts[:, j]
    intercept_idx = reasons.index("intercept achieved") if "intercept achieved" in reasons else 0
    for i, s in enumerate(scenarios):
        total = int(bottom[i])
        n_success = int(counts[i, intercept_idx])
        ax.text(i, total + 0.4, f"{n_success}/{total} intercepts", ha="center", fontsize=10)
    ax.set_ylabel("Number of Monte Carlo runs")
    ax.set_title("Failure-Mode Analysis per Uncertainty Scenario")
    ax.legend()
    ax.set_ylim(0, bottom.max() * 1.15 + 1)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_sensitivity_curves(
    df_sweep: pd.DataFrame,
    output_path: str,
) -> None:
    """Line plots of terminal miss distance vs. disturbance magnitude per category."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cats = df_sweep["parameter_category"].unique().tolist()
    n = len(cats)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 5 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, cat in zip(axes, cats):
        grp = df_sweep[df_sweep["parameter_category"] == cat].sort_values("numerical_val")
        ax.errorbar(grp["numerical_val"], grp["mean_miss_km"] / 1e6, yerr=grp["std_miss_km"] / 1e6,
                    fmt="-o", color="#1f77b4", capsize=4, lw=1.8)
        ax.set_xlabel("Perturbation magnitude")
        ax.set_ylabel("Terminal miss distance ($10^6$ km)")
        ax.set_title(cat)
        ax.grid(alpha=0.3)
    for ax in axes[len(cats):]:
        ax.set_visible(False)

    fig.suptitle("Parametric Sensitivity — Miss Distance vs. Disturbance Level", y=1.01)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_nominal_deviation(
    nominal_traj: list[dict],
    dispersed_trajs: list[list[dict]],
    output_path: str,
) -> None:
    """
    Comparisons against the nominal (deterministic) reference trajectory:
    heliocentric position deviation profiles of dispersed runs vs. baseline.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    if len(nominal_traj) == 0:
        plt.close(fig)
        return

    nom_steps = np.array([pt["step"] for pt in nominal_traj])
    nom_xyz = np.column_stack([
        [pt["x_km"] for pt in nominal_traj],
        [pt["y_km"] for pt in nominal_traj],
        [pt["z_km"] for pt in nominal_traj],
    ])

    max_dev_profile = np.zeros_like(nom_steps, dtype=float)
    rms_dev_profile = np.zeros_like(nom_steps, dtype=float)
    count = np.zeros_like(nom_steps, dtype=int)

    # Plot a few dispersed trajectories vs. nominal in the left panel
    axes[0].plot(nom_steps, np.linalg.norm(nom_xyz, axis=1) / 1e6, color="k", lw=2.5,
                 label="Nominal (deterministic)")
    for traj in dispersed_trajs[:10]:
        steps = np.array([pt["step"] for pt in traj])
        xyz = np.column_stack([[pt["x_km"] for pt in traj], [pt["y_km"] for pt in traj], [pt["z_km"] for pt in traj]])
        axes[0].plot(steps, np.linalg.norm(xyz, axis=1) / 1e6, color="#1f77b4", alpha=0.35, lw=1.0)
        # align to nominal step grid
        for st, val in zip(steps, xyz):
            idx = np.searchsorted(nom_steps, st)
            if idx < len(nom_steps):
                dev = np.linalg.norm(val - nom_xyz[idx])
                max_dev_profile[idx] = max(max_dev_profile[idx], dev)
                rms_dev_profile[idx] += dev**2
                count[idx] += 1
    axes[0].set_xlabel("Flight time (h)")
    axes[0].set_ylabel("Heliocentric radius ($10^6$ km)")
    axes[0].set_title("Nominal vs. Dispersed Trajectories")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    rms_dev_profile = np.sqrt(np.divide(rms_dev_profile, np.maximum(count, 1)))
    valid = count > 0
    if valid.any():
        axes[1].fill_between(nom_steps[valid], 0, max_dev_profile[valid] / 1e6,
                             color="#d62728", alpha=0.3, label="Max deviation envelope")
        axes[1].plot(nom_steps[valid], rms_dev_profile[valid] / 1e6, color="#1f77b4", lw=2,
                     label="RMS deviation")
    axes[1].set_xlabel("Flight time (h)")
    axes[1].set_ylabel("Deviation from nominal ($10^6$ km)")
    axes[1].set_title("Position Error vs. Nominal Reference")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
