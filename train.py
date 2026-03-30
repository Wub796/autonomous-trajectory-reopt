import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from spacecraft_env import SpacecraftEnv

class CurriculumCallback(BaseCallback):
    def __init__(self, reward_threshold: float, verbose=0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        if len(self.model.ep_info_buffer) > 0:
            mean_reward = sum(ep["r"] for ep in self.model.ep_info_buffer) / len(self.model.ep_info_buffer)
            if mean_reward >= self.reward_threshold:
                self.training_env.set_attr('enable_anomalies', True)
                print(f"[CURRICULUM] Anomaly phase activated at timestep {self.num_timesteps} | ep_rew_mean: {mean_reward:.1f}")
        return True

class SaveVecNormalizeCallback(BaseCallback):
    """
    Saves the VecNormalize statistics at the exact same frequency as the model CheckpointCallback.
    """
    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "vec_normalize", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.pkl")
            self.training_env.save(path)
        return True

# 1. Environment Wrapper Setup
def make_env():
    return Monitor(SpacecraftEnv())

train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env, 
    norm_obs=True, 
    norm_reward=True, 
    clip_obs=10.0, 
    clip_reward=10.0
)

eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env, 
    norm_obs=True, 
    norm_reward=False, 
    training=False, 
    clip_obs=10.0
)

# 2. Callback Instantiation
checkpoint_freq = 11040 * 25 # Save every 10 episodes
save_path = "./logs/checkpoints/"

checkpoint_callback = CheckpointCallback(
    save_freq=checkpoint_freq,
    save_path=save_path,
    name_prefix="ppo_spacecraft"
)

vec_normalize_callback = SaveVecNormalizeCallback(
    save_freq=checkpoint_freq,
    save_path=save_path,
    name_prefix="vec_normalize"
)

eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./logs/best_model',
    log_path='./logs/results', 
    eval_freq=11040 * 2, 
    deterministic=True, 
    render=False
)

curriculum_callback = CurriculumCallback(reward_threshold=500.0)

# Bundle callbacks
callback_list = CallbackList([
    eval_callback, 
    curriculum_callback, 
    checkpoint_callback, 
    vec_normalize_callback
])

# 3. PPO Model Definition
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
    tensorboard_log="./ppo_mars_logs/"
)

# 4. Execution
print("Ignition... Phase 5 Stabilized Training started.")
model.learn(total_timesteps=5520000, callback=callback_list)

# 5. Final State Preservation
model.save("ppo_spacecraft_phase5_final")
train_env.save("vec_normalize_phase5_final.pkl")
print("Mission complete. Model and environment statistics archived.")