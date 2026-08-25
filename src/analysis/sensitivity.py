"""
sensitivity.py — Parametric sensitivity and perturbation analysis.

Following the methodology of:
Capra, Brandonisio, and Lavagna (2022),
"Network architecture and action space analysis for deep reinforcement learning towards spacecraft autonomous guidance",
Advances in Space Research.
"""
from typing import Any
import numpy as np
import pandas as pd
from src.env.uncertainty import UncertaintyConfig
from src.analysis.monte_carlo import run_monte_carlo_suite


def run_sensitivity_sweep(
    model: Any,
    episodes_per_level: int = 20,
    base_seed: int = 100,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Executes one-at-a-time sensitivity sweeps across 5 disturbance dimensions:
    1. Launch Injection Velocity Error (m/s)
    2. Navigation Observation Noise (km)
    3. Thrust Execution Error (%)
    4. Thruster Outage Rate (%)
    5. Pointing Jitter (deg)
    """
    sweep_records: list[dict[str, Any]] = []

    # Baseline nominal
    base_cfg = UncertaintyConfig.deterministic()
    _, base_stats, _ = run_monte_carlo_suite(
        model, n_episodes=episodes_per_level, uncertainty_config=base_cfg, base_seed=base_seed
    )
    base_miss_km = base_stats["miss_dist_mean_km"]

    # 1. Launch Injection Velocity Error Sweep
    for vel_m_s in [0.5, 1.0, 5.0, 10.0, 20.0]:
        cfg = UncertaintyConfig.deterministic()
        cfg.enabled = True
        cfg.sigma_init_vel_km_s = vel_m_s / 1000.0
        _, stats, _ = run_monte_carlo_suite(
            model, n_episodes=episodes_per_level, uncertainty_config=cfg, base_seed=base_seed
        )
        sweep_records.append({
            "parameter_category": "Launch Velocity Error",
            "parameter_value": f"{vel_m_s:.1f} m/s",
            "numerical_val": vel_m_s,
            "mean_miss_km": stats["miss_dist_mean_km"],
            "std_miss_km": stats["miss_dist_std_km"],
            "delta_from_baseline_km": stats["miss_dist_mean_km"] - base_miss_km,
            "mean_fuel_kg": stats["fuel_mean_kg"],
        })

    # 2. Navigation Observation Noise Sweep
    for obs_noise_km in [10.0, 50.0, 100.0, 200.0, 300.0]:
        cfg = UncertaintyConfig.deterministic()
        cfg.enabled = True
        cfg.sigma_obs_sc_pos_km = obs_noise_km
        cfg.sigma_obs_rel_pos_km = obs_noise_km * 1.5
        _, stats, _ = run_monte_carlo_suite(
            model, n_episodes=episodes_per_level, uncertainty_config=cfg, base_seed=base_seed
        )
        sweep_records.append({
            "parameter_category": "Observation Noise",
            "parameter_value": f"{obs_noise_km:.0f} km",
            "numerical_val": obs_noise_km,
            "mean_miss_km": stats["miss_dist_mean_km"],
            "std_miss_km": stats["miss_dist_std_km"],
            "delta_from_baseline_km": stats["miss_dist_mean_km"] - base_miss_km,
            "mean_fuel_kg": stats["fuel_mean_kg"],
        })

    # 3. Thrust Execution Error Sweep
    for thrust_err_pct in [0.5, 1.0, 2.5, 5.0, 10.0]:
        cfg = UncertaintyConfig.deterministic()
        cfg.enabled = True
        cfg.sigma_thrust_magnitude_pct = thrust_err_pct / 100.0
        _, stats, _ = run_monte_carlo_suite(
            model, n_episodes=episodes_per_level, uncertainty_config=cfg, base_seed=base_seed
        )
        sweep_records.append({
            "parameter_category": "Thrust Magnitude Error",
            "parameter_value": f"{thrust_err_pct:.1f}%",
            "numerical_val": thrust_err_pct,
            "mean_miss_km": stats["miss_dist_mean_km"],
            "std_miss_km": stats["miss_dist_std_km"],
            "delta_from_baseline_km": stats["miss_dist_mean_km"] - base_miss_km,
            "mean_fuel_kg": stats["fuel_mean_kg"],
        })

    # 4. Thruster Outage Rate Sweep
    for outage_pct in [0.1, 0.5, 1.0, 2.0, 5.0]:
        cfg = UncertaintyConfig.deterministic()
        cfg.enabled = True
        cfg.p_missed_thrust_step = outage_pct / 100.0
        _, stats, _ = run_monte_carlo_suite(
            model, n_episodes=episodes_per_level, uncertainty_config=cfg, base_seed=base_seed
        )
        sweep_records.append({
            "parameter_category": "Thruster Outage Rate",
            "parameter_value": f"{outage_pct:.1f}% / hr",
            "numerical_val": outage_pct,
            "mean_miss_km": stats["miss_dist_mean_km"],
            "std_miss_km": stats["miss_dist_std_km"],
            "delta_from_baseline_km": stats["miss_dist_mean_km"] - base_miss_km,
            "mean_fuel_kg": stats["fuel_mean_kg"],
        })

    # 5. Pointing Jitter Sweep
    for pointing_deg in [0.1, 0.25, 0.5, 1.0, 2.0]:
        cfg = UncertaintyConfig.deterministic()
        cfg.enabled = True
        cfg.sigma_pointing_jitter_rad = float(np.deg2rad(pointing_deg))
        _, stats, _ = run_monte_carlo_suite(
            model, n_episodes=episodes_per_level, uncertainty_config=cfg, base_seed=base_seed
        )
        sweep_records.append({
            "parameter_category": "Pointing Alignment Jitter",
            "parameter_value": f"{pointing_deg:.2f}°",
            "numerical_val": pointing_deg,
            "mean_miss_km": stats["miss_dist_mean_km"],
            "std_miss_km": stats["miss_dist_std_km"],
            "delta_from_baseline_km": stats["miss_dist_mean_km"] - base_miss_km,
            "mean_fuel_kg": stats["fuel_mean_kg"],
        })

    df_sweep = pd.DataFrame(sweep_records)
    return df_sweep, sweep_records
