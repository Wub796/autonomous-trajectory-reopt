from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from spacecraft_env import SpacecraftEnv
from stable_baselines3.common.monitor import Monitor

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

train_env = Monitor(SpacecraftEnv())
# Create a separate evaluation environment
eval_env = Monitor(SpacecraftEnv())

# Stop training once a certain reward threshold is met or just save the best
eval_callback = EvalCallback(
    eval_env, 
    best_model_save_path='./logs/best_model',
    log_path='./logs/results', 
    eval_freq=11040 * 2, 
    deterministic=True, 
    render=False
)

curriculum_callback = CurriculumCallback(reward_threshold=500.0)

model = PPO(
    "MlpPolicy", 
    train_env, 
    n_steps=2760,      
    batch_size=460,    
    ent_coef=0.01,       # Forced exploration
    learning_rate=1e-3,  # Accelerated early gradient descent
    verbose=1, 
    tensorboard_log="./ppo_mars_logs/"
)

model.learn(total_timesteps=5520000, callback=[eval_callback, curriculum_callback])

model.save("ppo_spacecraft")
print("Training complete")