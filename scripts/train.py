"""
train.py — PPO training entrypoint for the Mars trajectory RL agent.

Run from the project root:
    python scripts/train.py
"""
import os
import numpy as np
from collections import deque

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from src.env.spacecraft_env import SpacecraftEnv

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


# ---------------------------------------------------------------------------
# Custom Callbacks
# ---------------------------------------------------------------------------

class CurriculumCallback(BaseCallback):
    """Activates anomaly injection once the agent reaches a reward threshold."""

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold
        self.activated = False

    def _on_step(self) -> bool:
        if self.activated:
            return True
            
        buf = self.model.ep_info_buffer
        if isinstance(buf, deque) and len(buf) > 0:
            mean_reward = sum(ep["r"] for ep in buf) / len(buf)
            if mean_reward >= self.reward_threshold:
                self.training_env.set_attr('enable_anomalies', True)
                print(
                    f"[CURRICULUM] Anomaly phase activated at timestep "
                    f"{self.num_timesteps} | ep_rew_mean: {mean_reward:.1f}"
                )
                self.activated = True
        return True


class SaveVecNormalizeCallback(BaseCallback):
    """
    Saves the VecNormalize statistics at the same frequency as CheckpointCallback.
    """

    def __init__(self, save_freq: int, save_path: str,
                 name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.pkl"
            )
            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(path)
        return True


class StopOnInterceptCallback(BaseCallback):
    """
    Evaluates the current model every eval_freq steps in the raw environment.
    Stops training early if the spacecraft successfully intercepts Mars (distance < 577,000 km).
    """

    def __init__(self, eval_freq: int, verbose: int = 0):
        super().__init__(verbose)
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            from src.env.spacecraft_env import SpacecraftEnv
            test_env = SpacecraftEnv()
            obs, _ = test_env.reset()
            done = False
            step = 0
            while not done and step < 11040:
                if isinstance(self.training_env, VecNormalize):
                    norm_obs = self.training_env.normalize_obs(obs)
                else:
                    norm_obs = obs
                action, _ = self.model.predict(norm_obs, deterministic=True)
                obs, reward, done, truncated, info = test_env.step(action)
                step += 1
                
            final_dist = np.linalg.norm(test_env.state[3:6])
            print(f"[EVALUATION] Step {self.num_timesteps} | Final distance to Mars: {final_dist:,.2f} km")
            
            if final_dist < 577000:
                print(f"🎯 SUCCESS! Target intercept achieved at step {self.num_timesteps} | Dist: {final_dist:,.2f} km < 577,000 km.")
                os.makedirs(_ARTIFACTS_DIR, exist_ok=True)
                self.model.save(os.path.join(_ARTIFACTS_DIR, "ppo_spacecraft_phase5_final"))
                if isinstance(self.training_env, VecNormalize):
                    self.training_env.save(os.path.join(_ARTIFACTS_DIR, "vec_normalize_phase5_final.pkl"))
                print("[EVALUATION] Saved successful model and normalization statistics.")
                return False  # Stops training!
        return True


# ---------------------------------------------------------------------------
# Environment Setup
# ---------------------------------------------------------------------------

def make_env():
    return Monitor(SpacecraftEnv())


train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env,
    norm_obs=True,
    norm_reward=False,
    clip_obs=10.0,
)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env,
    norm_obs=True,
    norm_reward=False,
    training=False,
    clip_obs=10.0,
)

# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------

checkpoint_freq = 11040 * 25   # every 25 episodes
checkpoint_dir  = "./logs/checkpoints/"

checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint_freq,
    save_path=checkpoint_dir,
    name_prefix="ppo_spacecraft",
)

vec_normalize_callback = SaveVecNormalizeCallback(
    save_freq=checkpoint_freq,
    save_path=checkpoint_dir,
    name_prefix="vec_normalize",
)

eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/best_model',
    log_path='./logs/results',
    eval_freq=11040 * 2,
    deterministic=True,
    render=False,
)

curriculum_callback = CurriculumCallback(reward_threshold=500.0)

stop_on_intercept_callback = StopOnInterceptCallback(eval_freq=11040 * 2)

callback_list = CallbackList([
    eval_callback,
    curriculum_callback,
    checkpoint_callback,
    vec_normalize_callback,
    stop_on_intercept_callback,
])

# ---------------------------------------------------------------------------
# Model Definition & Training
# ---------------------------------------------------------------------------

model = PPO(
    "MlpPolicy",
    train_env,
    n_steps=2760,
    batch_size=460,
    ent_coef=0.01,
    learning_rate=lambda progress_remaining: progress_remaining * 3e-4,
    clip_range=0.2,
    clip_range_vf=0.2,
    verbose=1,
    tensorboard_log="./ppo_mars_logs/",
)

print("Ignition... Phase 5 Stabilized Training started.")
model.learn(total_timesteps=5_520_000, callback=callback_list)

# Final State Preservation
final_model_path = os.path.join(_ARTIFACTS_DIR, "ppo_spacecraft_phase5_final")
final_vec_path   = os.path.join(_ARTIFACTS_DIR, "vec_normalize_phase5_final.pkl")
model.save(final_model_path)
train_env.save(final_vec_path)
print("Mission complete. Model and environment statistics archived.")
