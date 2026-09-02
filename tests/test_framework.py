"""
test_framework.py — Comprehensive unit and integration test suite for the research extensions.
"""
import unittest
import numpy as np
import torch
import os

from src.env.uncertainty import UncertaintyConfig, DisturbanceModel
from src.env.robust_spacecraft_env import RobustSpacecraftEnv
from src.models.architectures import (
    create_feedforward_ppo,
    create_recurrent_lstm_ppo,
    load_policy_from_zip,
    predict_action,
)
from src.deployment.exporter import export_to_onnx, export_to_torchscript, count_parameters
from src.deployment.pil_runner import (
    EmbeddedGNCInferenceEngine,
    profile_inference_latency,
    run_closed_loop_pil_simulation,
)
from src.analysis.monte_carlo import run_single_dispersed_trajectory, run_monte_carlo_suite


class TestUncertaintyModeling(unittest.TestCase):
    """Verifies Zavoli & Federici (2021) disturbance models."""

    def test_deterministic_mode(self):
        cfg = UncertaintyConfig.deterministic()
        dist = DisturbanceModel(cfg)
        d_pos, d_vel, d_mass = dist.sample_initial_state_perturbation()
        np.testing.assert_array_equal(d_pos, np.zeros(3))
        np.testing.assert_array_equal(d_vel, np.zeros(3))
        self.assertEqual(d_mass, 0.0)

        t_act, th_act, ph_act, missed = dist.apply_control_uncertainty(0.289, 1.0, 1.5, 0.289)
        self.assertEqual(t_act, 0.289)
        self.assertFalse(missed)

    def test_zavoli_federici_stochasticity(self):
        cfg = UncertaintyConfig.zavoli_federici_2021()
        dist = DisturbanceModel(cfg, np.random.default_rng(42))
        
        # Initial dispersion
        d_pos, d_vel, d_mass = dist.sample_initial_state_perturbation()
        self.assertNotEqual(np.linalg.norm(d_pos), 0.0)
        self.assertNotEqual(np.linalg.norm(d_vel), 0.0)

        # Process noise
        w_pos, w_vel = dist.sample_process_noise()
        self.assertEqual(w_pos.shape, (3,))
        self.assertEqual(w_vel.shape, (3,))

        # Observation noise
        raw_state = np.ones(12) * 100.0
        noisy_state = dist.apply_observation_noise(raw_state)
        self.assertFalse(np.array_equal(raw_state, noisy_state))


class TestRobustSpacecraftEnv(unittest.TestCase):
    """Verifies Gymnasium environment integration under noise."""

    def test_env_step_and_reset(self):
        env = RobustSpacecraftEnv(uncertainty_config=UncertaintyConfig.mild(), seed=42)
        obs, info = env.reset()
        self.assertEqual(obs.shape, (12,))
        self.assertTrue(np.all(obs >= 0.0) and np.all(obs <= 1.0))

        action = np.array([0.289, np.pi, np.pi / 2], dtype=np.float32)
        obs, reward, term, trunc, info = env.step(action)
        self.assertIn("true_state", info)
        self.assertIn("noisy_state", info)
        self.assertIn("mars_dist_km", info)
        self.assertFalse(term)


class TestModelArchitectures(unittest.TestCase):
    """Verifies Capra et al. (2022) model builders and loaders."""

    def test_policy_loading_and_prediction(self):
        model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "ppo_spacecraft_phase5_final.zip")
        if os.path.exists(model_path):
            policy = load_policy_from_zip(model_path)
            self.assertIsNotNone(policy)
            obs = np.random.uniform(0.0, 1.0, size=12).astype(np.float32)
            action, state = predict_action(policy, obs, deterministic=True)
            self.assertEqual(action.shape[-1], 3)
            self.assertIsNone(state)


class TestDeploymentAndPIL(unittest.TestCase):
    """Verifies Capra et al. (2025) ONNX/TorchScript export and PIL benchmarking."""

    def setUp(self):
        self.test_deploy_dir = os.path.join(os.path.dirname(__file__), "..", "artifacts", "test_deploy")
        os.makedirs(self.test_deploy_dir, exist_ok=True)
        model_path = os.path.join(os.path.dirname(__file__), "..", "artifacts", "ppo_spacecraft_phase5_final.zip")
        self.policy = load_policy_from_zip(model_path)

    def test_onnx_and_torchscript_export(self):
        onnx_path = os.path.join(self.test_deploy_dir, "test_policy.onnx")
        ts_path = os.path.join(self.test_deploy_dir, "test_policy.pt")

        onnx_meta = export_to_onnx(self.policy, onnx_path)
        ts_meta = export_to_torchscript(self.policy, ts_path)

        self.assertTrue(os.path.exists(onnx_path))
        self.assertTrue(os.path.exists(ts_path))
        self.assertGreater(onnx_meta["num_params"], 0)
        self.assertGreater(ts_meta["num_params"], 0)

        # Profile latency
        engine = EmbeddedGNCInferenceEngine(onnx_path, runtime="onnx", single_threaded=True)
        perf = profile_inference_latency(engine, n_trials=50)
        self.assertLess(perf["mean_us"], 1000.0)  # Sub-millisecond latency requirement


if __name__ == "__main__":
    unittest.main()
