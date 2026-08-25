"""
pil_runner.py — Processor-in-the-Loop (PIL) execution harness and embedded latency profiler.

Following Capra, Brandonisio, and Lavagna (2025),
"Reinforced Model Predictive Guidance and Control for Spacecraft Proximity Operations", Aerospace.
"""
import os
import json
import time
import platform
import numpy as np
import onnxruntime as ort
import torch
from typing import Any
from src.env.robust_spacecraft_env import RobustSpacecraftEnv


# ---------------------------------------------------------------------------
# Layer-level & node-level profiling helpers
# ---------------------------------------------------------------------------

def count_module_params(module: Any) -> int:
    """Total trainable parameters of a torch module."""
    return int(sum(p.numel() for p in module.parameters() if p.requires_grad))


def _time_module_call(fn, *args, repeat: int, warmup: int = 20) -> float:
    """Returns mean single-call duration in microseconds."""
    for _ in range(warmup):
        fn(*args)
    t0 = time.perf_counter_ns()
    for _ in range(repeat):
        fn(*args)
    t1 = time.perf_counter_ns()
    return (t1 - t0) / repeat / 1000.0


def profile_policy_layer_breakdown(
    policy: Any,
    obs_dim: int = 12,
    n_trials: int = 500,
) -> list[dict[str, Any]]:
    """
    Times each internal layer of the deployed GNC actor (LSTM if present,
    policy trunk layers, action head) on the current host. This is the
    per-component timing breakdown requested for Section 5.6.
    """
    policy.eval()
    obs = torch.rand(1, obs_dim, dtype=torch.float32)
    records: list[dict[str, Any]] = []
    x = obs

    # 1. Recurrent stage (LSTM policies only)
    if hasattr(policy, "lstm_actor"):
        lstm = policy.lstm_actor
        h0 = torch.zeros(lstm.num_layers, 1, lstm.hidden_size)
        c0 = torch.zeros(lstm.num_layers, 1, lstm.hidden_size)
        seq = obs.unsqueeze(0)
        mean_us = _time_module_call(lambda: lstm(seq, (h0, c0)), repeat=n_trials)
        latent, _ = lstm(seq, (h0, c0))
        x = latent.squeeze(0)
        records.append({
            "component": "LSTM recurrent stage",
            "layer_type": "nn.LSTM",
            "params": count_module_params(lstm),
            "mean_us": mean_us,
        })

    # 2. Policy trunk (mlp_extractor.policy_net)
    net = getattr(policy, "mlp_extractor", None)
    if net is not None and hasattr(net, "policy_net"):
        trunk = net.policy_net
        for i, layer in enumerate(trunk):
            mean_us = _time_module_call(layer, x, repeat=n_trials)
            x = layer(x)  # advance activation through the trunk copy
            records.append({
                "component": f"policy trunk layer {i}",
                "layer_type": type(layer).__name__,
                "params": count_module_params(layer),
                "mean_us": mean_us,
            })

    # 3. Action head
    act = getattr(policy, "action_net", None)
    if act is not None:
        mean_us = _time_module_call(act, x, repeat=n_trials)
        records.append({
            "component": "action head (mean)",
            "layer_type": type(act).__name__,
            "params": count_module_params(act),
            "mean_us": mean_us,
        })

    total = sum(r["mean_us"] for r in records) or 1e-9
    for r in records:
        r["share_of_actor_pct"] = r["mean_us"] / total * 100.0
        r["n_trials"] = n_trials
        r["hardware"] = f"{platform.system()} ({platform.machine()})"
    return records


def profile_onnx_node_breakdown(
    engine: "EmbeddedGNCInferenceEngine",
    obs_dim: int = 12,
    n_trials: int = 500,
) -> tuple[list[dict[str, Any]] | None, str | None]:
    """
    ONNX Runtime node-level profile (per-graph-node CPU time) aggregated over
    n_trials. Returns (records, error) — records is None when the runtime
    profiler is unavailable.
    """
    try:
        # Profiling must be enabled at session-creation time in onnxruntime >= 1.18
        prof_opts = ort.SessionOptions()
        prof_opts.intra_op_num_threads = 1
        prof_opts.inter_op_num_threads = 1
        prof_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        prof_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        prof_opts.enable_profiling = True
        session = ort.InferenceSession(engine.model_path, sess_options=prof_opts,
                                       providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        obs = np.random.uniform(0.0, 1.0, size=(1, obs_dim)).astype(np.float32)
        for _ in range(n_trials):
            session.run(None, {input_name: obs})
        prof_file = session.end_profiling()
        with open(prof_file, "r") as f:
            trace = json.load(f)
        os.remove(prof_file)

        node_times: dict[str, float] = {}
        node_counts: dict[str, int] = {}
        for ev in trace:
            if ev.get("cat") == "Node" and "dur" in ev and ev.get("dur") is not None:
                name = str(ev.get("name", "?"))
                node_times[name] = node_times.get(name, 0.0) + float(ev["dur"])
                node_counts[name] = node_counts.get(name, 0) + 1
        total = sum(node_times.values()) or 1e-9
        records = [{
            "onnx_node": name,
            "n_invocations": node_counts[name],
            "total_us": node_times[name],
            "mean_us": node_times[name] / max(node_counts[name], 1),
            "share_of_graph_pct": node_times[name] / total * 100.0,
        } for name in sorted(node_times, key=lambda k: -node_times[k])]
        return records, None
    except Exception as exc:  # pragma: no cover - runtime-dependent
        return None, str(exc)


def profile_torchscript_op_breakdown(
    engine: "EmbeddedGNCInferenceEngine",
    obs_dim: int = 12,
    n_trials: int = 50,
) -> list[dict[str, Any]]:
    """
    torch.profiler op-level breakdown for the TorchScript engine (single-thread
    CPU activities aggregated per operator).
    """
    obs = torch.rand(1, obs_dim, dtype=torch.float32)
    for _ in range(10):
        engine.infer(obs.numpy())

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        for _ in range(n_trials):
            engine.infer(obs.numpy())

    events = prof.key_averages()
    total = sum(e.self_cpu_time_total for e in events) or 1e-9
    records = [{
        "operator": e.key,
        "n_calls": e.count,
        "self_cpu_us": e.self_cpu_time_total / 1000.0,
        "share_of_graph_pct": e.self_cpu_time_total / total * 100.0,
    } for e in sorted(events, key=lambda e: -e.self_cpu_time_total) if e.self_cpu_time_total > 0]
    return records


# ---------------------------------------------------------------------------
# Memory & power tracking
# ---------------------------------------------------------------------------

def measure_process_memory_mb() -> dict[str, float]:
    """Current RSS and peak RSS of this process, in MB."""
    rss_mb = float("nan")
    peak_mb = float("nan")
    try:
        import psutil
        rss_mb = psutil.Process(os.getpid()).memory_info().rss / 1e6
    except Exception:
        pass
    try:
        import resource
        ru = resource.getrusage(resource.RUSAGE_SELF)
        # macOS reports ru_maxrss in bytes; Linux in kilobytes
        peak_mb = ru.ru_maxrss / 1e6 if platform.system() == "Darwin" else ru.ru_maxrss / 1024.0
    except Exception:
        pass
    return {"rss_mb": rss_mb, "peak_rss_mb": peak_mb}


def estimate_raspberry_pi_power(
    mean_latency_us: float,
    n_gnc_steps: int = 11040,
    active_power_w: float = 3.4,
    idle_power_w: float = 2.7,
) -> dict[str, float]:
    """
    Engineering estimate of energy consumption for the GNC workload on a
    Raspberry Pi 4 Model B (documented idle ~2.7 W, single-core active ~3.4 W;
    ../../paper refs. official power-measurement data). Measuring real power
    requires a USB-C power meter — see methodology note in the report.
    """
    per_inference_s = mean_latency_us * 1e-6
    energy_per_inference_j = active_power_w * per_inference_s
    mission_energy_j = energy_per_inference_j * n_gnc_steps
    mission_baseline_j = idle_power_w * n_gnc_steps  # hypothetical always-on baseline
    return {
        "assumed_active_power_w": active_power_w,
        "assumed_idle_power_w": idle_power_w,
        "energy_per_inference_uJ": energy_per_inference_j * 1e6,
        "mission_energy_j": mission_energy_j,
        "mission_baseline_idle_j": mission_baseline_j,
        "n_gnc_steps": n_gnc_steps,
    }


class EmbeddedGNCInferenceEngine:
    """
    Flight-representative GNC inference engine running via ONNX Runtime or TorchScript
    with constrained single-core CPU execution.
    """

    def __init__(self, model_path: str, runtime: str = "onnx", single_threaded: bool = True):
        self.model_path = model_path
        self.runtime = runtime.lower()
        self.single_threaded = single_threaded

        if self.runtime == "onnx":
            opts = ort.SessionOptions()
            if single_threaded:
                opts.intra_op_num_threads = 1
                opts.inter_op_num_threads = 1
                opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
            self.input_name = self.session.get_inputs()[0].name
        elif self.runtime == "torchscript":
            if single_threaded:
                torch.set_num_threads(1)
            self.session = torch.jit.load(model_path)
            self.session.eval()
        else:
            raise ValueError(f"Unsupported runtime: {runtime}. Choose 'onnx' or 'torchscript'.")

    def infer(self, obs: np.ndarray) -> np.ndarray:
        """
        Executes a single forward pass, returning the deterministic control action.
        """
        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0).astype(np.float32)

        if self.runtime == "onnx":
            outputs = self.session.run(None, {self.input_name: obs})
            return outputs[0][0]
        else:
            with torch.no_grad():
                tensor_obs = torch.from_numpy(obs)
                output = self.session(tensor_obs)
                return output.numpy()[0]


def profile_inference_latency(
    engine: EmbeddedGNCInferenceEngine,
    obs_dim: int = 12,
    n_warmup: int = 500,
    n_trials: int = 5000,
) -> dict[str, Any]:
    """
    Measures microsecond-level latency distributions across thousands of consecutive forward passes.
    """
    dummy_obs = np.random.uniform(0.0, 1.0, size=obs_dim).astype(np.float32)

    # 1. Warm-up
    for _ in range(n_warmup):
        _ = engine.infer(dummy_obs)

    # 2. Benchmark trials with high-resolution clock
    latencies_us: list[float] = []
    for _ in range(n_trials):
        t0 = time.perf_counter_ns()
        _ = engine.infer(dummy_obs)
        t1 = time.perf_counter_ns()
        latencies_us.append((t1 - t0) / 1000.0)  # nanoseconds -> microseconds

    latencies = np.array(latencies_us)

    mean_us = float(np.mean(latencies))
    median_us = float(np.median(latencies))
    std_us = float(np.std(latencies))
    min_us = float(np.min(latencies))
    max_us = float(np.max(latencies))
    p90_us = float(np.percentile(latencies, 90))
    p95_us = float(np.percentile(latencies, 95))
    p99_us = float(np.percentile(latencies, 99))
    p999_us = float(np.percentile(latencies, 99.9))

    # Real-time execution margin for a 1-hour GNC step budget (3600 seconds)
    # and a 1-Hz high-rate attitude control loop (1.0 second)
    gnc_step_budget_s = 3600.0
    margin_1hr = (1.0 - (max_us * 1e-6 / gnc_step_budget_s)) * 100.0

    high_rate_budget_s = 1.0
    margin_1hz = (1.0 - (max_us * 1e-6 / high_rate_budget_s)) * 100.0

    return {
        "runtime": engine.runtime,
        "single_threaded": engine.single_threaded,
        "n_trials": n_trials,
        "mean_us": mean_us,
        "median_us": median_us,
        "std_us": std_us,
        "min_us": min_us,
        "max_us": max_us,
        "p90_us": p90_us,
        "p95_us": p95_us,
        "p99_us": p99_us,
        "p999_us": p999_us,
        "throughput_hz": 1e6 / mean_us,
        "realtime_margin_1hr_pct": margin_1hr,
        "realtime_margin_1hz_pct": margin_1hz,
    }


def run_closed_loop_pil_simulation(
    engine: EmbeddedGNCInferenceEngine,
    env: RobustSpacecraftEnv,
    max_steps: int = 11040,
) -> dict[str, Any]:
    """
    Executes a complete closed-loop flight simulation with the embedded inference engine.
    """
    obs, info = env.reset()
    step_latencies_us: list[float] = []
    telemetry: list[dict] = []

    done = False
    step = 0
    total_reward = 0.0

    while not done and step < max_steps:
        # Measure onboard GNC guidance latency
        t0 = time.perf_counter_ns()
        action = engine.infer(obs)
        t1 = time.perf_counter_ns()
        latency_us = (t1 - t0) / 1000.0
        step_latencies_us.append(latency_us)

        # Scale/clip action to physical bounds
        action_clipped = np.clip(action, env.action_space.low, env.action_space.high)

        # Step orbital environment
        obs, reward, terminated, truncated, info = env.step(action_clipped)
        total_reward += reward
        step += 1
        done = terminated or truncated

        if step % 1000 == 0:
            telemetry.append({
                "step": step,
                "mars_dist_km": info["mars_dist_km"],
                "mass_kg": info["mass_kg"],
                "latency_us": latency_us,
            })

    final_dist = info["mars_dist_km"]
    final_mass = info["mass_kg"]

    return {
        "total_steps": step,
        "final_mars_dist_km": final_dist,
        "final_mass_kg": final_mass,
        "total_reward": total_reward,
        "mean_latency_us": float(np.mean(step_latencies_us)),
        "max_latency_us": float(np.max(step_latencies_us)),
        "p99_latency_us": float(np.percentile(step_latencies_us, 99)),
        "is_intercepted": final_dist < env.capture_radius_km,
        "telemetry_checkpoints": telemetry,
    }
