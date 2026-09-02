"""
exporter.py — Model export utilities for ONNX and TorchScript flight software integration.

Following Capra, Brandonisio, and Lavagna (2025),
"Reinforced Model Predictive Guidance and Control for Spacecraft Proximity Operations", Aerospace.
"""
import os
import torch
import torch.nn as nn
from typing import Any, Union
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from sb3_contrib import RecurrentPPO


class OnnxMlpActorWrapper(nn.Module):
    """Wraps an SB3 MLP Actor for standard ONNX/TorchScript export."""

    def __init__(self, policy: Any):
        super().__init__()
        # SB3 actor network: mlp_extractor.policy_net + action_net
        self.mlp_extractor: nn.Module = policy.mlp_extractor
        self.action_net: nn.Module = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Forward pass through policy trunk
        latent_pi: torch.Tensor = getattr(self.mlp_extractor, "forward_actor")(obs)
        # Deterministic action mean
        action_mean: torch.Tensor = self.action_net(latent_pi)
        return action_mean


class OnnxLstmActorWrapper(nn.Module):
    """
    Wraps the actor path of an SB3-Contrib RecurrentPPO (MlpLstmPolicy) policy
    for single-step ONNX/TorchScript export:
        obs -> LSTM (zero-init hidden) -> mlp_extractor (actor trunk) -> action_net
    The zero-initialised hidden state corresponds to episode start, which is the
    worst-case (cold-cache) forward path profiled in the PIL benchmark.
    """

    def __init__(self, policy: Any):
        super().__init__()
        self.lstm_actor: nn.Module = policy.lstm_actor
        self.mlp_extractor: nn.Module = policy.mlp_extractor
        self.action_net: nn.Module = policy.action_net

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        batch = obs.shape[0]
        n_layers = self.lstm_actor.num_layers
        hidden_size = self.lstm_actor.hidden_size
        # sb3-contrib feeds the LSTM sequence-first: (seq_len, batch, features)
        features_seq = obs.unsqueeze(0)  # (1, batch, features)
        h0 = torch.zeros(n_layers, batch, hidden_size, device=obs.device)
        c0 = torch.zeros(n_layers, batch, hidden_size, device=obs.device)
        latent, _ = self.lstm_actor(features_seq, (h0, c0))  # (1, batch, hidden)
        latent = latent.squeeze(0)  # (batch, hidden)
        latent_pi = getattr(self.mlp_extractor, "forward_actor")(latent)
        return self.action_net(latent_pi)


def count_parameters(model: nn.Module) -> int:
    """Returns total trainable parameter count."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_mlp_flops(obs_dim: int, net_arch: list[int], action_dim: int) -> int:
    """Estimates FLOPs (Multiply-Accumulate * 2) for a standard feed-forward MLP."""
    flops = 0
    in_dim = obs_dim
    for hidden_dim in net_arch:
        flops += 2 * (in_dim * hidden_dim) + hidden_dim  # Linear layer + activation
        in_dim = hidden_dim
    flops += 2 * (in_dim * action_dim)  # Output layer
    return flops


def estimate_lstm_flops(obs_dim: int, lstm_hidden_size: int, n_lstm_layers: int,
                        net_arch: list[int], action_dim: int) -> int:
    """
    Estimates FLOPs per timestep for an LSTM policy:
    LSTM cell (4 gates * MACs) + MLP trunk + action head.
    """
    lstm_per_layer = 4 * 2 * lstm_hidden_size * (obs_dim + lstm_hidden_size)
    lstm = lstm_per_layer * n_lstm_layers
    mlp = estimate_mlp_flops(lstm_hidden_size, net_arch, action_dim)
    return lstm + mlp


def _extract_policy_module(model: Union[PPO, RecurrentPPO, ActorCriticPolicy, nn.Module]) -> nn.Module:
    """Extracts the underlying PyTorch nn.Module policy."""
    if hasattr(model, "policy") and isinstance(getattr(model, "policy"), nn.Module):
        return getattr(model, "policy")
    elif isinstance(model, nn.Module):
        return model
    else:
        raise TypeError(f"Expected nn.Module or SB3 model, got {type(model)}")


def export_to_onnx(
    model: Union[PPO, RecurrentPPO, ActorCriticPolicy, nn.Module],
    output_path: str,
    obs_dim: int = 12,
    opset_version: int = 18,
) -> dict[str, Any]:
    """
    Exports a trained policy to ONNX format and computes flight model metadata.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cpu")
    
    policy_module = _extract_policy_module(model)
    policy_module.to(device)
    policy_module.eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)

    # Wrap actor for clean ONNX export (LSTM policies get a recurrent wrapper)
    if hasattr(policy_module, "lstm_actor"):
        actor_wrapper = OnnxLstmActorWrapper(policy_module)
    else:
        actor_wrapper = OnnxMlpActorWrapper(policy_module)
    actor_wrapper.eval()

    torch.onnx.export(
        actor_wrapper,
        (dummy_input,),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["observation"],
        output_names=["action_mean"],
        dynamic_axes={"observation": {0: "batch_size"}, "action_mean": {0: "batch_size"}},
    )

    file_size_kb = os.path.getsize(output_path) / 1024.0
    num_params = count_parameters(actor_wrapper)
    if hasattr(policy_module, "lstm_actor"):
        net_arch = [256, 256]
        hidden = policy_module.lstm_actor.hidden_size
        layers = policy_module.lstm_actor.num_layers
        estimated_flops = estimate_lstm_flops(obs_dim, hidden, layers, net_arch, 3)
    else:
        estimated_flops = estimate_mlp_flops(obs_dim, [256, 256], 3)

    return {
        "onnx_path": output_path,
        "file_size_kb": file_size_kb,
        "num_params": num_params,
        "estimated_flops": estimated_flops,
    }


def export_to_torchscript(
    model: Union[PPO, RecurrentPPO, ActorCriticPolicy, nn.Module],
    output_path: str,
    obs_dim: int = 12,
) -> dict[str, Any]:
    """
    Exports a trained policy to TorchScript (.pt) for C++ / embedded libtorch deployment.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    device = torch.device("cpu")
    
    policy_module = _extract_policy_module(model)
    policy_module.to(device)
    policy_module.eval()

    if hasattr(policy_module, "lstm_actor"):
        actor_wrapper = OnnxLstmActorWrapper(policy_module)
    else:
        actor_wrapper = OnnxMlpActorWrapper(policy_module)
    actor_wrapper.eval()

    dummy_input = torch.zeros(1, obs_dim, dtype=torch.float32, device=device)
    traced_model: Any = torch.jit.trace(actor_wrapper, (dummy_input,))
    traced_model.save(output_path)

    file_size_kb = os.path.getsize(output_path) / 1024.0
    num_params = count_parameters(actor_wrapper)

    return {
        "torchscript_path": output_path,
        "file_size_kb": file_size_kb,
        "num_params": num_params,
    }
