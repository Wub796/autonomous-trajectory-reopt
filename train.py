from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
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
                # Trigger anomaly phase across all vectorized environments
                self.training_env.set_attr('enable_anomalies', True)
        return True

# 1. Environment Wrapper Setup
def make_env():
    return Monitor(SpacecraftEnv())

# Training Environment (Normalizes Observations AND Rewards)
train_env = DummyVecEnv([make_env])
train_env = VecNormalize(
    train_env, 
    norm_obs=True, 
    norm_reward=True, 
    clip_obs=10.0, 
    clip_reward=10.0
)

# Evaluation Environment (Normalizes Observations, but NEVER Rewards. Does not update stats.)
eval_env = DummyVecEnv([make_env])
eval_env = VecNormalize(
    eval_env, 
    norm_obs=True, 
    norm_reward=False, 
    training=False, # CRITICAL FIX: Freezes running statistics during evaluation
    clip_obs=10.0
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

# 2. PPO Model Definition with Stability Fixes
model = PPO(
    "MlpPolicy", 
    train_env, 
    n_steps=2760,      
    batch_size=460,    
    ent_coef=0.01,       
    learning_rate=lambda progress_remaining: progress_remaining * 3e-4, # Linear Decay
    clip_range=0.2,
    clip_range_vf=0.2, # Value Function Clipping
    verbose=1, 
    tensorboard_log="./ppo_mars_logs/"
)

# 3. Execution
print("Ignition... Phase 4 Training started.")
model.learn(total_timesteps=5520000, callback=[eval_callback, curriculum_callback])

# 4. State Preservation (Model + Normalization Stats)
model.save("ppo_spacecraft_phase4")
train_env.save("vec_normalize_phase4.pkl")
print("Mission complete. Model and environment statistics archived.")