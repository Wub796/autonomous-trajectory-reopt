"""
compare_architectures.py — Comparison of Feed-Forward vs. Recurrent Neural Network Architectures
for autonomous spacecraft trajectory optimization under partial observability and noise.

Following the methodology of:
Capra, Brandonisio, and Lavagna (2022),
"Network architecture and action space analysis for deep reinforcement learning towards spacecraft autonomous guidance",
Advances in Space Research.

In addition to the summary benchmark table (Table 3 equivalent), this script now also
produces the Section 5.5 data products:

  * Full training dynamics curves (reward, value loss, explained variance, entropy,
    approx_kl over training steps) exported to CSVs and plotted.
  * Training stability metrics (convergence step, oscillation index, final/peak
    value loss, explained variance, entropy, KL) per architecture.
  * Explicit hyperparameter records (net_arch, LSTM hidden size/layers, lr, ent_coef,
    n_steps, batch_size, clip_range) for the report.
  * Single-threaded ONNX inference-time profiling on the current host (laptop).
    Re-run the identical command on a Raspberry Pi 4 to populate the embedded row.

Usage:
    PYTHONPATH=. python scripts/compare_architectures.py --timesteps 50000 --eval-episodes 10
    PYTHONPATH=. python scripts/compare_architectures.py --quick-test
    PYTHONPATH=. python scripts/compare_architectures.py --timesteps 20000 --inference-trials 1000
"""
import os
import time
import argparse
import platform
import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback

from src.env.robust_spacecraft_env import RobustSpacecraftEnv
from src.env.uncertainty import UncertaintyConfig
from src.models.architectures import (
    create_feedforward_ppo,
    create_recurrent_lstm_ppo,
    predict_action,
)
from src.deployment.exporter import export_to_onnx
from src.deployment.pil_runner import EmbeddedGNCInferenceEngine, profile_inference_latency

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ARTIFACTS_DIR = os.path.join(_PROJECT_ROOT, "artifacts")
_FIGURES_DIR = os.path.join(_ARTIFACTS_DIR, "figures")

# ---------------------------------------------------------------------------
# Training-dynamics logging
# ---------------------------------------------------------------------------

# SB3 logger keys captured at each rollout end
_METRIC_KEYS = [
    "train/value_loss",
    "train/explained_variance",
    "train/entropy_loss",
    "train/approx_kl",
    "train/std",
    "train/policy_gradient_loss",
    "train/loss",
    "train/clip_fraction",
    "rollout/ep_rew_mean",
    "rollout/ep_len_mean",
    "time/fps",
]


class TrainingDynamicsLogger(BaseCallback):
    """
    Captures the per-rollout training dynamics (reward, value loss, explained
    variance, entropy, KL) and periodically evaluates the capture success rate
    so training curves + success-rate curves can be exported for the paper.
    """

    def __init__(self, model_name: str, make_eval_env, uncertainty_config,
                 eval_every_n_rollouts: int = 0, n_eval_episodes: int = 2,
                 verbose: int = 0):
        super().__init__(verbose)
        self.model_name = model_name
        self.make_eval_env = make_eval_env
        self.uncertainty_config = uncertainty_config
        self.eval_every_n_rollouts = eval_every_n_rollouts
        self.n_eval_episodes = n_eval_episodes
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> bool:
        step = int(self.num_timesteps)
        log = dict(self.model.logger.name_to_value)  # latest scalar values
        # Rollout mean reward: use the Monitor ep_info_buffer when episodes
        # complete within the rollout, otherwise the mean of the just-collected
        # rollout buffer rewards (episodes are 11,040 steps long, i.e. several
        # rollouts, so ep_info_buffer is often empty at _on_rollout_end).
        ep_infos = list(self.model.ep_info_buffer)
        if ep_infos and all(isinstance(ep, dict) and "r" in ep for ep in ep_infos):
            mean_reward = float(np.mean([ep["r"] for ep in ep_infos]))
            ep_len = float(np.mean([ep.get("l", np.nan) for ep in ep_infos]))
        else:
            try:
                rewards = getattr(self.model, "rollout_buffer", None).rewards
                mean_reward = float(np.mean(rewards)) if rewards is not None else np.nan
            except Exception:
                mean_reward = float(log.get("rollout/ep_rew_mean", np.nan))
            ep_len = float(log.get("rollout/ep_len_mean", np.nan))
        record = {
            "model": self.model_name,
            "step": step,
            "mean_reward": mean_reward,
            "value_loss": float(log.get("train/value_loss", np.nan)),
            "explained_variance": float(log.get("train/explained_variance", np.nan)),
            "entropy_loss": float(log.get("train/entropy_loss", np.nan)),
            "approx_kl": float(log.get("train/approx_kl", np.nan)),
            "std": float(log.get("train/std", np.nan)),
            "ep_len_mean": ep_len,
            "fps": float(log.get("time/fps", np.nan)),
            "capture_rate_pct": np.nan,
        }
        # Periodic in-training capture-rate evaluation (success-rate curve)
        if self.eval_every_n_rollouts > 0 and self.n_calls % self.eval_every_n_rollouts == 0:
            record["capture_rate_pct"] = self._eval_capture_rate()
        self.history.append(record)
        return True

    def _eval_capture_rate(self) -> float:
        env = self.make_eval_env()
        captures = 0
        for ep in range(self.n_eval_episodes):
            obs, _ = env.reset(seed=10000 + ep)
            done = False
            step = 0
            lstm_states = None
            episode_start = np.ones((1,), dtype=bool)
            while not done and step < env.t_max:
                norm_obs = obs
                norm_obs_batch = np.expand_dims(norm_obs, axis=0)
                if isinstance(self.model, type(None)):
                    break
                if hasattr(self.model, "policy") and hasattr(self.model.policy, "lstm_actor"):
                    action, lstm_states = self.model.predict(
                        norm_obs_batch, state=lstm_states, episode_start=episode_start, deterministic=True)
                    episode_start = np.zeros((1,), dtype=bool)
                else:
                    action, _ = self.model.predict(norm_obs_batch, deterministic=True)
                obs, _, terminated, truncated, info = env.step(action[0])
                step += 1
                done = terminated or truncated
            if info["mars_dist_km"] < env.capture_radius_km:
                captures += 1
        return float(captures / max(self.n_eval_episodes, 1) * 100.0)


# ---------------------------------------------------------------------------
# Stability metrics
# ---------------------------------------------------------------------------

def compute_stability_metrics(curve: pd.DataFrame) -> dict:
    """
    Derives training-stability statistics from the per-rollout reward series:
      - convergence_step: first step where the smoothed reward settles within 2%
        of its final value (a proxy for time-to-convergence).
      - oscillation_index: std of first-differences of the smoothed reward,
        normalised by the mean absolute reward (higher = more oscillatory).
    """
    s = curve.sort_values("step").reset_index(drop=True)
    # Forward-fill metric columns so lead/trail NaN records (logger not yet
    # populated on the first rollouts) don't corrupt the statistics.
    for col in ["mean_reward", "value_loss", "explained_variance",
                "entropy_loss", "approx_kl", "std"]:
        s[col] = s[col].ffill()
    if len(s) < 3:
        return {"convergence_step": None, "oscillation_index": None,
                "final_mean_reward": None, "peak_mean_reward": None}
    reward = s["mean_reward"].rolling(5, min_periods=1).mean()
    final_r = float(reward.iloc[-1])
    span = max(float(reward.max() - reward.min()), 1e-9)
    settled = np.abs(reward.to_numpy() - final_r) <= 0.02 * span + 1e-9
    conv_idx = int(np.argmax(settled)) if settled.any() else len(s) - 1
    diffs = np.diff(reward.to_numpy())
    osc = float(np.std(diffs) / (np.mean(np.abs(reward.to_numpy())) + 1e-9))
    return {
        "convergence_step": int(s["step"].iloc[conv_idx]),
        "oscillation_index": osc,
        "final_mean_reward": final_r,
        "peak_mean_reward": float(reward.max()),
    }


def plot_training_curves(curves: dict[str, pd.DataFrame], output_path: str) -> None:
    """2x2 panel of reward, value loss, explained variance, entropy over steps."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    panels = [
        (axes[0, 0], "mean_reward", "Cumulative reward (ep_rew_mean)", "Reward"),
        (axes[0, 1], "value_loss", "Value loss convergence", "Value loss"),
        (axes[1, 0], "explained_variance", "Explained variance (critic accuracy)", "Explained variance"),
        (axes[1, 1], "entropy_loss", "Policy entropy (exploration)", "Entropy loss"),
    ]
    colors = {"Feed-Forward MLP": "#1f77b4", "Recurrent LSTM": "#d62728"}
    for name, df in curves.items():
        d = df.sort_values("step")
        for ax, col, title, ylab in panels:
            ax.plot(d["step"], d[col], lw=1.6, color=colors.get(name, None), label=name)
            ax.set_title(title)
            ax.set_xlabel("Training timesteps")
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
            ax.legend()
    plt.suptitle("Neural Architecture Training Dynamics (Capra et al. 2022)", y=1.0)
    plt.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_architecture(
    model,
    is_recurrent: bool,
    uncertainty_config: UncertaintyConfig,
    n_episodes: int = 10,
    vec_normalize: VecNormalize | None = None,
) -> dict:
    """
    Evaluates an agent across n stochastic test episodes.
    (Reuses a single env instance across episodes to avoid repeated ephemeris
    pre-computation.)
    """
    test_env = RobustSpacecraftEnv(uncertainty_config=uncertainty_config, seed=1000)
    miss_distances = []
    propellant_used = []
    step_counts = []
    rewards = []
    capture_successes = []

    for ep in range(n_episodes):
        obs, info = test_env.reset(seed=1000 + ep)
        done = False
        step = 0
        ep_reward = 0.0
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while not done and step < test_env.t_max:
            if vec_normalize is not None:
                norm_obs = vec_normalize.normalize_obs(obs)
            else:
                norm_obs = obs

            norm_obs_batch = np.expand_dims(norm_obs, axis=0)

            if is_recurrent:
                action, lstm_states = model.predict(
                    norm_obs_batch,
                    state=lstm_states,
                    episode_start=episode_start,
                    deterministic=True,
                )
                episode_start = np.zeros((1,), dtype=bool)
            else:
                action, _ = model.predict(norm_obs_batch, deterministic=True)

            cmd_action = action[0]
            obs, reward, terminated, truncated, info = test_env.step(cmd_action)
            ep_reward += reward
            step += 1
            done = terminated or truncated

        final_dist = info["mars_dist_km"]
        final_mass = info["mass_kg"]
        used_fuel = 2747.0 - final_mass
        is_captured = final_dist < test_env.capture_radius_km

        miss_distances.append(final_dist)
        propellant_used.append(used_fuel)
        step_counts.append(step)
        rewards.append(ep_reward)
        capture_successes.append(is_captured)

    return {
        "mean_miss_dist_km": float(np.mean(miss_distances)),
        "std_miss_dist_km": float(np.std(miss_distances)),
        "min_miss_dist_km": float(np.min(miss_distances)),
        "max_miss_dist_km": float(np.max(miss_distances)),
        "mean_fuel_kg": float(np.mean(propellant_used)),
        "mean_reward": float(np.mean(rewards)),
        "capture_rate_pct": float(np.mean(capture_successes) * 100.0),
        "mean_steps": float(np.mean(step_counts)),
    }


def profile_host_inference(model_path_onnx: str, n_trials: int = 500) -> dict:
    """
    Single-threaded ONNX latency profile on the current host. Run the same call
    on a Raspberry Pi 4 to obtain the embedded-hardware row for the report.
    """
    engine = EmbeddedGNCInferenceEngine(model_path_onnx, runtime="onnx", single_threaded=True)
    perf = profile_inference_latency(engine, n_trials=n_trials)
    return {
        "inference_mean_us": perf["mean_us"],
        "inference_median_us": perf["median_us"],
        "inference_p99_us": perf["p99_us"],
        "inference_throughput_hz": perf["throughput_hz"],
        "inference_hardware": f"{platform.system()} ({platform.machine()})",
        "inference_n_trials": n_trials,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_comparison(
    timesteps: int = 50000,
    eval_episodes: int = 10,
    uncertainty_level: str = "zavoli",
    quick_test: bool = False,
    inference_trials: int = 500,
    training_evals: int = 0,
):
    if quick_test:
        timesteps = 5000
        eval_episodes = 3
        inference_trials = 100

    print(f"=================================================================")
    print(f" Neural Network Architecture Benchmark (Capra et al. 2022)")
    print(f" Timesteps per model: {timesteps:,} | Eval Episodes: {eval_episodes}")
    print(f" Uncertainty Profile: {uncertainty_level.upper()}")
    print(f"=================================================================")

    # Select uncertainty configuration
    if uncertainty_level == "mild":
        u_config = UncertaintyConfig.mild()
    elif uncertainty_level == "severe":
        u_config = UncertaintyConfig.severe()
    elif uncertainty_level == "deterministic":
        u_config = UncertaintyConfig.deterministic()
    else:
        u_config = UncertaintyConfig.zavoli_federici_2021()

    # 1. Environment creation helper
    def make_train_env():
        return Monitor(RobustSpacecraftEnv(uncertainty_config=u_config))

    def make_eval_env():
        return RobustSpacecraftEnv(uncertainty_config=u_config, seed=9999)

    results = []
    curves = {}
    hyperparameter_records = []

    # -------------------------------------------------------------------------
    # A. Train & Evaluate Feed-Forward (MLP) Agent
    # -------------------------------------------------------------------------
    print("\n[1/2] Training Feed-Forward MLP Agent (Baseline)...")
    mlp_vec_env = DummyVecEnv([make_train_env])
    mlp_vec_norm = VecNormalize(mlp_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    mlp_hyper = dict(
        architecture="Feed-Forward MLP (PPO Baseline)",
        policy="MlpPolicy", net_arch="pi=[256, 256], vf=[256, 256]",
        activation_fn="Tanh", lstm_hidden_size="n/a", n_lstm_layers="n/a",
        learning_rate="3e-4 (linear decay)", n_steps=1024 if quick_test else 2760,
        batch_size=256 if quick_test else 460, ent_coef=0.01, clip_range=0.2,
    )
    hyperparameter_records.append(mlp_hyper)

    eval_every = max(1, (timesteps // (1024 if quick_test else 2760)) // max(training_evals, 1))
    mlp_logger = TrainingDynamicsLogger(
        "Feed-Forward MLP", make_eval_env, u_config,
        eval_every_n_rollouts=eval_every if training_evals > 0 else 0,
        n_eval_episodes=max(1, eval_episodes // 3),
    )
    mlp_model = create_feedforward_ppo(
        mlp_vec_norm,
        n_steps=1024 if quick_test else 2760,
        batch_size=256 if quick_test else 460,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=0,
    )

    t0_mlp = time.time()
    mlp_model.learn(total_timesteps=timesteps, callback=mlp_logger)
    mlp_train_time = time.time() - t0_mlp
    print(f"   MLP Training complete in {mlp_train_time:.1f} s.")

    mlp_curve = pd.DataFrame(mlp_logger.history)
    curves["Feed-Forward MLP"] = mlp_curve

    print("   Evaluating Feed-Forward MLP under POMDP noise...")
    mlp_eval = evaluate_architecture(
        mlp_model, is_recurrent=False, uncertainty_config=u_config,
        n_episodes=eval_episodes, vec_normalize=mlp_vec_norm,
    )
    mlp_eval["architecture"] = mlp_hyper["architecture"]
    mlp_eval["train_time_sec"] = mlp_train_time
    mlp_eval.update(compute_stability_metrics(mlp_curve))

    print("   Profiling single-threaded ONNX inference on host...")
    mlp_onnx = os.path.join(_ARTIFACTS_DIR, "deployment", "mlp_comparison_policy.onnx")
    export_to_onnx(mlp_model, mlp_onnx)
    mlp_eval.update(profile_host_inference(mlp_onnx, n_trials=inference_trials))
    results.append(mlp_eval)

    # -------------------------------------------------------------------------
    # B. Train & Evaluate Recurrent (LSTM) Agent
    # -------------------------------------------------------------------------
    print("\n[2/2] Training Recurrent LSTM Agent (RecurrentPPO / POMDP)...")
    lstm_vec_env = DummyVecEnv([make_train_env])
    lstm_vec_norm = VecNormalize(lstm_vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0)

    lstm_hyper = dict(
        architecture="Recurrent LSTM (RecurrentPPO / POMDP)",
        policy="MlpLstmPolicy", net_arch="LSTM(128, 1 layer) + MLP [256, 256]",
        activation_fn="Tanh", lstm_hidden_size=128, n_lstm_layers=1,
        learning_rate="3e-4 (linear decay)", n_steps=1024 if quick_test else 2760,
        batch_size=256 if quick_test else 460, ent_coef=0.01, clip_range=0.2,
    )
    hyperparameter_records.append(lstm_hyper)

    lstm_logger = TrainingDynamicsLogger(
        "Recurrent LSTM", make_eval_env, u_config,
        eval_every_n_rollouts=eval_every if training_evals > 0 else 0,
        n_eval_episodes=max(1, eval_episodes // 3),
    )
    lstm_model = create_recurrent_lstm_ppo(
        lstm_vec_norm,
        n_steps=1024 if quick_test else 2760,
        batch_size=256 if quick_test else 460,
        lstm_hidden_size=128,
        n_lstm_layers=1,
        learning_rate=3e-4,
        ent_coef=0.01,
        verbose=0,
    )

    t0_lstm = time.time()
    lstm_model.learn(total_timesteps=timesteps, callback=lstm_logger)
    lstm_train_time = time.time() - t0_lstm
    print(f"   LSTM Training complete in {lstm_train_time:.1f} s.")

    lstm_curve = pd.DataFrame(lstm_logger.history)
    curves["Recurrent LSTM"] = lstm_curve

    print("   Evaluating Recurrent LSTM under POMDP noise...")
    lstm_eval = evaluate_architecture(
        lstm_model, is_recurrent=True, uncertainty_config=u_config,
        n_episodes=eval_episodes, vec_normalize=lstm_vec_norm,
    )
    lstm_eval["architecture"] = lstm_hyper["architecture"]
    lstm_eval["train_time_sec"] = lstm_train_time
    lstm_eval.update(compute_stability_metrics(lstm_curve))

    print("   Profiling single-threaded ONNX inference on host...")
    lstm_onnx = os.path.join(_ARTIFACTS_DIR, "deployment", "lstm_comparison_policy.onnx")
    export_to_onnx(lstm_model, lstm_onnx)
    lstm_eval.update(profile_host_inference(lstm_onnx, n_trials=inference_trials))
    results.append(lstm_eval)

    # -------------------------------------------------------------------------
    # Summary, CSV & figure export
    # -------------------------------------------------------------------------
    df_results = pd.DataFrame(results)
    os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
    df_results.to_csv(os.path.join(_ARTIFACTS_DIR, "architecture_comparison_results.csv"), index=False)

    curve_df = pd.concat(curves.values(), ignore_index=True)
    curve_df.to_csv(os.path.join(_ARTIFACTS_DIR, "architecture_training_curves.csv"), index=False)
    for name, df in curves.items():
        safe = name.lower().replace(" ", "_").replace("/", "_")
        df.to_csv(os.path.join(_ARTIFACTS_DIR, f"training_curves_{safe}.csv"), index=False)

    pd.DataFrame(hyperparameter_records).to_csv(
        os.path.join(_ARTIFACTS_DIR, "architecture_hyperparameters.csv"), index=False)

    fig_path = os.path.join(_FIGURES_DIR, "architecture_training_curves.png")
    plot_training_curves(curves, fig_path)

    # Add per-model final-value summaries onto the results table columns
    for r in results:
        name = r["architecture"]
        c = curves["Feed-Forward MLP"] if "Feed-Forward" in name else curves["Recurrent LSTM"]
        c = c.sort_values("step").ffill()
        r["final_value_loss"] = float(c["value_loss"].iloc[-1])
        r["peak_value_loss"] = float(c["value_loss"].max())
        r["final_explained_variance"] = float(c["explained_variance"].iloc[-1])
        r["final_entropy"] = float(c["entropy_loss"].iloc[-1])
        r["final_approx_kl"] = float(c["approx_kl"].iloc[-1])
        r["final_std"] = float(c["std"].iloc[-1])
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(_ARTIFACTS_DIR, "architecture_comparison_results.csv"), index=False)

    print("\n=================================================================")
    print(" Architecture Comparison Summary Table")
    print("=================================================================")
    for r in results:
        print(f"\nModel: {r['architecture']}")
        print(f"  Capture Rate:        {r['capture_rate_pct']:.1f}%")
        print(f"  Mean Miss Distance:  {r['mean_miss_dist_km']:,.1f} ± {r['std_miss_dist_km']:,.1f} km")
        print(f"  Mean Propellant Used:{r['mean_fuel_kg']:.2f} kg")
        print(f"  Mean Episodic Reward:{r['mean_reward']:,.1f}")
        print(f"  Final value_loss:    {r.get('final_value_loss', np.nan):.2e}")
        print(f"  Final explained_var: {r.get('final_explained_variance', np.nan):.3f}")
        print(f"  Convergence step:    {r.get('convergence_step')}")
        osc = r.get('oscillation_index')
        print(f"  Oscillation index:   {osc:.4f}" if osc is not None else "  Oscillation index:   n/a")
        print(f"  Host inference:      {r['inference_mean_us']:.2f} μs (median {r['inference_median_us']:.2f} μs)")
        print(f"  Training Time:       {r['train_time_sec']:.2f} s")

    # Generate Markdown Report
    report_path = os.path.join(_ARTIFACTS_DIR, "architecture_comparison_report.md")
    with open(report_path, "w") as f:
        f.write("# Neural Network Architecture Comparison Report\n\n")
        f.write("Following Capra, Brandonisio, and Lavagna (2022), this benchmark compares standard feed-forward (MLP) "
                "policies against recurrent (LSTM) policies in partially observable, stochastic trajectory environments.\n\n")
        f.write("## Benchmark Results Table\n\n")
        f.write("| Architecture | Capture Rate (%) | Mean Miss Distance (km) | Min Miss Dist (km) | Mean Propellant (kg) | Mean Reward | Training Time (s) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| **{r['architecture']}** | {r['capture_rate_pct']:.1f}% | {r['mean_miss_dist_km']:,.1f} ± {r['std_miss_dist_km']:,.1f} | {r['min_miss_dist_km']:,.1f} | {r['mean_fuel_kg']:.1f} | {r['mean_reward']:,.1f} | {r['train_time_sec']:.1f} |\n")

        f.write("\n## Training Dynamics (Section 5.5)\n\n")
        f.write("| Metric | Feed-Forward MLP | Recurrent LSTM |\n")
        f.write("| :--- | :--- | :--- |\n")
        for key, label in [
            ("final_value_loss", "Final value loss"),
            ("peak_value_loss", "Peak value loss"),
            ("final_explained_variance", "Final explained variance"),
            ("final_entropy", "Final entropy loss"),
            ("final_approx_kl", "Final approx_kl"),
            ("final_std", "Final policy std (action noise)"),
            ("convergence_step", "Convergence step (reward within 2% of final)"),
            ("oscillation_index", "Oscillation index (std of smoothed Δreward)"),
        ]:
            mlp_r = results[0]
            lstm_r = results[1]
            mv = mlp_r.get(key)
            lv = lstm_r.get(key)
            ms = f"{mv:,.4g}" if isinstance(mv, float) else (str(mv) if mv is not None else "n/a")
            ls = f"{lv:,.4g}" if isinstance(lv, float) else (str(lv) if lv is not None else "n/a")
            f.write(f"| **{label}** | {ms} | {ls} |\n")
        f.write(f"\nFull per-rollout curves exported to `architecture_training_curves.csv` / "
                f"`training_curves_feed-forward_mlp.csv` / `training_curves_recurrent_lstm.csv`; "
                f"plot saved to `figures/architecture_training_curves.png`.\n")

        f.write("\n## Hyperparameters\n\n")
        f.write("| Hyperparameter | Feed-Forward MLP | Recurrent LSTM |\n")
        f.write("| :--- | :--- | :--- |\n")
        for key, label in [("policy", "Policy"), ("net_arch", "Network architecture"),
                           ("activation_fn", "Activation"), ("lstm_hidden_size", "LSTM hidden size"),
                           ("n_lstm_layers", "LSTM layers"), ("learning_rate", "Learning rate"),
                           ("n_steps", "Rollout n_steps"), ("batch_size", "Batch size"),
                           ("ent_coef", "Entropy coef"), ("clip_range", "Clip range")]:
            f.write(f"| **{label}** | {mlp_hyper.get(key, 'n/a')} | {lstm_hyper.get(key, 'n/a')} |\n")

        f.write("\n## Inference Time (single-thread CPU)\n\n")
        f.write("| Architecture | Hardware | Trials | Mean (μs) | Median (μs) | P99 (μs) | Throughput (Hz) |\n")
        f.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in results:
            f.write(f"| **{r['architecture']}** | {r['inference_hardware']} | {r['inference_n_trials']} | "
                    f"{r['inference_mean_us']:.2f} | {r['inference_median_us']:.2f} | {r['inference_p99_us']:.2f} | "
                    f"{r['inference_throughput_hz']:,.0f} |\n")
        f.write("\n> To obtain the Raspberry Pi 4 row, copy the repo to the Pi and re-run the identical command "
                "(`PYTHONPATH=. python scripts/compare_architectures.py`); the script profiles the exported ONNX "
                "binaries single-threaded on whichever host it runs on.\n")

        f.write("\n## Findings & Astrodynamics Insights\n")
        f.write("1. **Temporal Filtering**: Recurrent architectures maintain hidden memory states that act as an implicit state observer (similar to an onboard Extended Kalman Filter), filtering navigation sensor noise.\n")
        f.write("2. **Robustness to Missed Thrust**: Under stochastic thruster outages, LSTM networks retain historical actuation commands, allowing quicker compensatory burns once propulsion is restored.\n")

    print(f"\nResults saved to {os.path.join(_ARTIFACTS_DIR, 'architecture_comparison_results.csv')} and {report_path}")
    print(f"Training curves saved to {fig_path}")
    return df_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Neural Architecture Comparison")
    parser.add_argument("--timesteps", type=int, default=50000, help="Training timesteps per architecture")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--uncertainty", type=str, default="zavoli", choices=["deterministic", "mild", "zavoli", "severe"])
    parser.add_argument("--inference-trials", type=int, default=500, help="Forward-pass trials for host inference profiling")
    parser.add_argument("--training-evals", type=int, default=0, help="In-training capture-rate evaluations (0 = disabled)")
    parser.add_argument("--quick-test", action="store_true", help="Run rapid smoke test")
    args = parser.parse_args()

    run_comparison(
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        uncertainty_level=args.uncertainty,
        quick_test=args.quick_test,
        inference_trials=args.inference_trials,
        training_evals=args.training_evals,
    )