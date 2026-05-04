import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
from spacecraft_env import SpacecraftEnv

# 1. Recreate the precise evaluation environment
def make_env():
    return SpacecraftEnv()

eval_env = DummyVecEnv([make_env])
# CRITICAL: Load the exact normalization statistics used during training
eval_env = VecNormalize.load("vec_normalize_phase5_final.pkl", eval_env)
eval_env.training = False # Ensure stats are frozen
eval_env.norm_reward = False

# 2. Load the optimized model
model = PPO.load("ppo_spacecraft_phase5_final")

# 3. Trajectory Generation Loop
obs = eval_env.reset()
terminated = False
telemetry = []

print("Simulating optimal trajectory...")

# We know the episode length is exactly 11,040 steps
for step in range(11040):
    # deterministic=True disables exploration noise for the true flight path
    action, _states = model.predict(obs, deterministic=True)
    
    # Access the underlying un-normalized environment to get absolute physics
    raw_env = eval_env.envs[0]
    
    # Record physical state BEFORE taking the action
    telemetry.append({
        "time_step_hr": step,
        "sc_x_km": raw_env.state[0],
        "sc_y_km": raw_env.state[1],
        "sc_z_km": raw_env.state[2],
        "sc_vx_km_s": raw_env.vel[0],
        "sc_vy_km_s": raw_env.vel[1],
        "sc_vz_km_s": raw_env.vel[2],
        "mars_dist_km": np.linalg.norm(raw_env.state[3:6]),
        "mass_kg": raw_env.state[9],
        "thrust_cmd_N": action[0][0], # action is returned as a 2D array by VecEnv
        "thrust_theta_rad": action[0][1],
        "thrust_phi_rad": action[0][2],
        "anomaly_active": raw_env.state[11] < 1782 # True if Isp dropped below baseline
    })
    
    obs, reward, done, info = eval_env.step(action)
    
    # VecEnv returns done as a boolean array
    if done[0]:
        print(f"Simulation terminated at step {step}")
        break

# 4. Export to CSV for Dashboarding
df = pd.DataFrame(telemetry)
export_path = "optimal_mars_trajectory.csv"
df.to_csv(export_path, index=False)
print(f"Trajectory data exported successfully to {export_path}")
print(f"Final distance to Mars: {telemetry[-1]['mars_dist_km']:,.2f} km")