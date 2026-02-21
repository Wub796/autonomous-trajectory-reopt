# Spacecraft PPO Training Infrastructure

## Resuming Training from Checkpoints
To resume training from a specific step, the model weights and the exact normalization statistics must be loaded synchronously to prevent catastrophic forgetting.

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

# Define target checkpoint
step_count = "276000"

# 1. Load exact environment normalization statistics
train_env = VecNormalize.load(f"logs/checkpoints/vec_normalize_{step_count}_steps.pkl", train_env)
# 2. Sync all VecNormalize statistics from train to eval
eval_env.obs_rms = train_env.obs_rms
eval_env.ret_rms = train_env.ret_rms

# 3. Load model weights coupled to the environment
model = PPO.load(f"logs/checkpoints/ppo_spacecraft_{step_count}_steps", env=train_env)

# 4. Resume execution (reset_num_timesteps=False maintains TensorBoard alignment)
model.learn(total_timesteps=remaining_timesteps, callback=callback_list, reset_num_timesteps=False)