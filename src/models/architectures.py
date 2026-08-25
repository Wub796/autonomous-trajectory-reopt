"""
architectures.py — Model architectures, policy builders, and robust model loading for spacecraft trajectory optimization.

Following Capra, Brandonisio, and Lavagna (2022),
"Network architecture and action space analysis for deep reinforcement learning towards spacecraft autonomous guidance",
Advances in Space Research.
"""
import io
import zipfile
from typing import Any, Union
import numpy as np
import torch
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecEnv


def create_feedforward_ppo(
    env: Union[VecEnv, Any],
    learning_rate: float = 3e-4,
    net_arch: list[int] | dict[str, list[int]] | None = None,
    activation_fn: type[nn.Module] = nn.Tanh,
    tensorboard_log: str | None = "./ppo_mars_logs/mlp",
    verbose: int = 1,
    **kwargs: Any,
) -> PPO:
    """
    Creates a Feed-Forward (MLP) PPO agent (Baseline).
    """
    if net_arch is None:
        net_arch = dict(pi=[256, 256], vf=[256, 256])

    policy_kwargs = {
        "net_arch": net_arch,
        "activation_fn": activation_fn,
    }

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **kwargs,
    )
    return model


def create_recurrent_lstm_ppo(
    env: Union[VecEnv, Any],
    learning_rate: float = 3e-4,
    lstm_hidden_size: int = 128,
    n_lstm_layers: int = 1,
    shared_lstm: bool = False,
    enable_critic_lstm: bool = True,
    activation_fn: type[nn.Module] = nn.Tanh,
    tensorboard_log: str | None = "./ppo_mars_logs/lstm",
    verbose: int = 1,
    **kwargs: Any,
) -> RecurrentPPO:
    """
    Creates a Recurrent Neural Network (LSTM) PPO agent for POMDPs.
    Retains temporal memory across steps to filter sensor noise and track thruster outages.
    """
    policy_kwargs = {
        "lstm_hidden_size": lstm_hidden_size,
        "n_lstm_layers": n_lstm_layers,
        "shared_lstm": shared_lstm,
        "enable_critic_lstm": enable_critic_lstm,
        "activation_fn": activation_fn,
    }

    model = RecurrentPPO(
        policy="MlpLstmPolicy",
        env=env,
        learning_rate=learning_rate,
        policy_kwargs=policy_kwargs,
        tensorboard_log=tensorboard_log,
        verbose=verbose,
        **kwargs,
    )
    return model


def load_policy_from_zip(
    zip_path: str,
    obs_dim: int = 12,
    action_dim: int = 3,
    t_max_n: float = 0.289,
) -> ActorCriticPolicy:
    """
    Instantly loads an ActorCriticPolicy from an SB3 checkpoint zip file,
    bypassing numpy/cloudpickle version incompatibilities.
    """
    obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
    act_space = gym.spaces.Box(
        low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
        high=np.array([t_max_n, 2 * np.pi, np.pi], dtype=np.float32),
        dtype=np.float32,
    )

    policy = ActorCriticPolicy(
        observation_space=obs_space,
        action_space=act_space,
        lr_schedule=lambda _: 0.0,
    )

    if not zip_path.endswith(".zip"):
        zip_path += ".zip"

    with zipfile.ZipFile(zip_path, "r") as z:
        policy_bytes = io.BytesIO(z.read("policy.pth"))
        state_dict = torch.load(policy_bytes, map_location="cpu", weights_only=False)
        policy.load_state_dict(state_dict)

    policy.eval()
    return policy


def predict_action(
    model: Union[PPO, RecurrentPPO, ActorCriticPolicy],
    obs: np.ndarray,
    state: tuple[np.ndarray, ...] | None = None,
    episode_start: np.ndarray | None = None,
    deterministic: bool = True,
) -> tuple[np.ndarray, tuple[np.ndarray, ...] | None]:
    """
    Unified prediction interface handling standard Feed-Forward PPO,
    ActorCriticPolicy instances, and Recurrent LSTM PPO policies with internal hidden states.
    """
    if isinstance(model, RecurrentPPO):
        action, new_state = model.predict(
            obs,
            state=state,
            episode_start=episode_start,
            deterministic=deterministic,
        )
        return action, new_state
    elif isinstance(model, ActorCriticPolicy):
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
        tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
        with torch.no_grad():
            action_dist = model.get_distribution(tensor_obs)
            actions = action_dist.get_actions(deterministic=deterministic).cpu().numpy()
        return actions, None
    else:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action, None
