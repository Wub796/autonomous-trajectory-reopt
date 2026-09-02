"""
monte_carlo.py — High-fidelity Monte Carlo trajectory dispersion and robustness analysis.

Following the methodology of:
Capra, Brandonisio, and Lavagna (2022),
"Network architecture and action space analysis for deep reinforcement learning towards spacecraft autonomous guidance",
Advances in Space Research.
"""
import numpy as np
import pandas as pd
from typing import Any
from src.env.robust_spacecraft_env import RobustSpacecraftEnv
from src.env.uncertainty import UncertaintyConfig
from src.models.architectures import predict_action


def run_single_dispersed_trajectory(
    model: Any,
    seed: int,
    uncertainty_config: UncertaintyConfig,
    record_trajectory: bool = False,
) -> dict[str, Any]:
    """
    Executes a single dispersed trajectory simulation.
    """
    env = RobustSpacecraftEnv(uncertainty_config=uncertainty_config, seed=seed)
    obs, info = env.reset()

    traj_points: list[dict] = []
    done = False
    step = 0
    total_reward = 0.0
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)

    # Initial state conditions
    init_pos = env.state[0:3].copy()
    init_vel = env.vel.copy()
    init_mass = float(env.state[9])

    min_dist_to_mars = float(np.linalg.norm(env.state[3:6]))
    min_dist_step = 0

    while not done and step < env.t_max:
        # Predict action
        action, lstm_states = predict_action(
            model,
            obs,
            state=lstm_states,
            episode_start=episode_start,
            deterministic=True,
        )
        episode_start = np.zeros((1,), dtype=bool)
        cmd_action = action[0] if action.ndim > 1 else action

        if record_trajectory and (step % 50 == 0 or step == env.t_max - 1):
            traj_points.append({
                "step": step,
                "x_km": env.state[0],
                "y_km": env.state[1],
                "z_km": env.state[2],
                "vx_km_s": env.vel[0],
                "vy_km_s": env.vel[1],
                "vz_km_s": env.vel[2],
                "mars_dist_km": float(np.linalg.norm(env.state[3:6])),
                "mass_kg": float(env.state[9]),
                "thrust_N": float(cmd_action[0]),
            })

        obs, reward, terminated, truncated, info = env.step(cmd_action)
        total_reward += reward
        step += 1
        done = terminated or truncated

        current_dist = info["mars_dist_km"]
        if current_dist < min_dist_to_mars:
            min_dist_to_mars = current_dist
            min_dist_step = step

    final_dist = info["mars_dist_km"]
    final_mass = info["mass_kg"]
    fuel_consumed = init_mass - final_mass
    final_rel_vel = float(np.linalg.norm(env.state[6:9]))
    is_intercepted = final_dist < env.capture_radius_km

    # Failure-mode classification (why did this run end as it did?)
    fuel_exhausted = final_mass <= env.obs_min[9] + 1e-6
    if is_intercepted:
        failure_reason = "intercept achieved"
    elif fuel_exhausted:
        failure_reason = "fuel exhausted"
    else:
        failure_reason = "time expired without intercept"

    return {
        "seed": seed,
        "total_steps": step,
        "final_mars_dist_km": final_dist,
        "min_mars_dist_km": min_dist_to_mars,
        "min_dist_step": min_dist_step,
        "final_mass_kg": final_mass,
        "fuel_consumed_kg": fuel_consumed,
        "final_rel_vel_km_s": final_rel_vel,
        "total_reward": total_reward,
        "missed_thrust_steps": info["total_missed_steps"],
        "is_intercepted": is_intercepted,
        "failure_reason": failure_reason,
        "convergence_step": min_dist_step,  # time-to-closest-approach (h)
        "init_pos_x": init_pos[0],
        "init_pos_y": init_pos[1],
        "init_pos_z": init_pos[2],
        "init_vel_x": init_vel[0],
        "init_vel_y": init_vel[1],
        "init_vel_z": init_vel[2],
        "trajectory_points": traj_points if record_trajectory else [],
    }


def run_monte_carlo_suite(
    model: Any,
    n_episodes: int = 100,
    uncertainty_config: UncertaintyConfig | None = None,
    base_seed: int = 42,
    record_sample_trajectories: int = 20,
) -> tuple[pd.DataFrame, dict[str, Any], list[list[dict]]]:
    """
    Runs a full Monte Carlo ensemble over N dispersed trajectory runs.
    """
    if uncertainty_config is None:
        uncertainty_config = UncertaintyConfig.zavoli_federici_2021()

    results = []
    recorded_trajectories = []

    for i in range(n_episodes):
        seed = base_seed + i
        should_record = i < record_sample_trajectories
        res = run_single_dispersed_trajectory(
            model,
            seed=seed,
            uncertainty_config=uncertainty_config,
            record_trajectory=should_record,
        )
        if should_record and len(res["trajectory_points"]) > 0:
            recorded_trajectories.append(res["trajectory_points"])
        del res["trajectory_points"]
        results.append(res)

    df = pd.DataFrame(results)

    # Compute Statistical Metrics & Dispersion Envelopes
    miss_dists = df["final_mars_dist_km"].values
    fuels = df["fuel_consumed_kg"].values
    rel_vels = df["final_rel_vel_km_s"].values
    rewards = df["total_reward"].values
    conv_steps = df["convergence_step"].values

    # Distribution of success / failure cases
    failure_counts = df["failure_reason"].value_counts().to_dict()

    stats = {
        "n_episodes": n_episodes,
        "capture_rate_pct": float(np.mean(df["is_intercepted"]) * 100.0),
        "miss_dist_mean_km": float(np.mean(miss_dists)),
        "miss_dist_std_km": float(np.std(miss_dists)),
        "miss_dist_median_km": float(np.median(miss_dists)),
        "miss_dist_min_km": float(np.min(miss_dists)),
        "miss_dist_max_km": float(np.max(miss_dists)),
        "miss_dist_1sigma_km": float(np.percentile(miss_dists, 68.27)),
        "miss_dist_2sigma_km": float(np.percentile(miss_dists, 95.45)),
        "miss_dist_3sigma_km": float(np.percentile(miss_dists, 99.73)),
        "fuel_mean_kg": float(np.mean(fuels)),
        "fuel_std_kg": float(np.std(fuels)),
        "fuel_median_kg": float(np.median(fuels)),
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_median": float(np.median(rewards)),
        "reward_min": float(np.min(rewards)),
        "reward_max": float(np.max(rewards)),
        "convergence_step_mean_h": float(np.mean(conv_steps)),
        "convergence_step_std_h": float(np.std(conv_steps)),
        "final_mass_mean_kg": float(np.mean(df["final_mass_kg"])),
        "rel_vel_mean_km_s": float(np.mean(rel_vels)),
        "rel_vel_std_km_s": float(np.std(rel_vels)),
        "mean_missed_thrust_steps": float(np.mean(df["missed_thrust_steps"])),
        "failure_mode_counts": failure_counts,
    }

    return df, stats, recorded_trajectories
