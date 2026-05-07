# Autonomous Trajectory Re-optimization

Reinforcement learning agent (PPO) that learns to fly a spacecraft from Earth to Mars using a realistic physics simulation and a multi-objective reward.

---

## Project Structure

```
research/
├── src/
│   ├── env/spacecraft_env.py   # Gymnasium physics environment
│   └── utils/ephemeris.py      # Astropy planetary state-vector helper
├── scripts/
│   ├── train.py                # Training entrypoint
│   └── trajectory.py           # Evaluation & CSV export
├── artifacts/                  # Model weights, normalizers, outputs (gitignored)
├── logs/                       # Checkpoints & eval results (gitignored)
└── ppo_mars_logs/              # TensorBoard event files (gitignored)
```

---

## Setup

### 1. Create & activate the virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies
```bash
pip install stable-baselines3[extra] gymnasium astropy scikit-learn pandas numpy tensorboard
```

---

## Running

> **All commands must be run from the project root** (`research/`) so that `src.*` imports resolve correctly.

### Train the agent
```bash
source .venv/bin/activate
python scripts/train.py
```
- Trains for **5.52M timesteps** (~460 episodes of 11,040 steps each)
- Saves checkpoints every 25 episodes to `logs/checkpoints/`
- Saves the best model to `logs/best_model/`
- Writes TensorBoard logs to `ppo_mars_logs/`
- Final model saved to `artifacts/ppo_spacecraft_phase5_final.zip`

### Generate the optimal trajectory CSV
```bash
source .venv/bin/activate
python scripts/trajectory.py
```
- Loads `artifacts/ppo_spacecraft_phase5_final.zip` + `vec_normalize_phase5_final.pkl`
- Runs one deterministic episode
- Exports `artifacts/optimal_mars_trajectory.csv` (11,040 rows × 12 columns)

### Query planetary state vectors
```bash
source .venv/bin/activate
python -m src.utils.ephemeris              # defaults to 2027-02-19
python -m src.utils.ephemeris 2027-06-01   # custom date
```

### Monitor training (TensorBoard)
```bash
source .venv/bin/activate
tensorboard --logdir ppo_mars_logs/
# then open http://localhost:6006
```

---

## Resuming from a Checkpoint

Model weights and normalization stats must always be loaded together to prevent catastrophic forgetting:

```python
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

step_count = "276000"  # pick any checkpoint from logs/checkpoints/

train_env = VecNormalize.load(f"logs/checkpoints/vec_normalize_{step_count}_steps.pkl", train_env)
eval_env.obs_rms = train_env.obs_rms
eval_env.ret_rms = train_env.ret_rms

model = PPO.load(f"logs/checkpoints/ppo_spacecraft_{step_count}_steps", env=train_env)
model.learn(total_timesteps=remaining_timesteps, callback=callback_list, reset_num_timesteps=False)
```