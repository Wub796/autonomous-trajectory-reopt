"""
Experiment 6: Onboard Flight Computer Execution Timing
Benchmarks single-step decision latency and memory footprint of the neural pilot.
Demonstrates feasibility for low-power, radiation-tolerant spacecraft processors.
"""
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mission import pilot as AI

os.makedirs("artifacts/figures", exist_ok=True)

def run():
    print("=== Experiment 6: Onboard Execution Profiling & Real-Time Margin ===")

    if not os.path.exists("artifacts/trained_pilot.npy"):
        raise FileNotFoundError("Run experiments/03_train_pilot.py first!")

    theta = np.load("artifacts/trained_pilot.npy")
    obs_sample = np.ones(AI.N_IN)

    # 1. Warm-up
    for _ in range(100):
        _ = AI.act_single(theta, obs_sample)

    # 2. Benchmark N=10,000 single-step guidance calls
    N_TRIALS = 10_000
    latencies_us = np.zeros(N_TRIALS)

    for i in range(N_TRIALS):
        t0 = time.perf_counter()
        _ = AI.act_single(theta, obs_sample)
        t1 = time.perf_counter()
        latencies_us[i] = (t1 - t0) * 1e6

    mean_us = np.mean(latencies_us)
    median_us = np.median(latencies_us)
    p99_us = np.percentile(latencies_us, 99.0)
    max_us = np.max(latencies_us)

    # Guidance cycle is dt = 12 hours = 43,200 seconds
    CYCLE_BUDGET_SEC = 12 * 3600.0
    margin_pct = (1.0 - (mean_us * 1e-6) / CYCLE_BUDGET_SEC) * 100.0

    # Memory calculation
    param_count = len(theta)
    param_bytes = theta.nbytes  # 243 * 8 bytes = 1944 bytes ~ 1.9 KB

    print(f"Parameters:              {param_count} float64 weights ({param_bytes} bytes / {param_bytes/1024:.2f} KB)")
    print(f"Mean Decision Latency:   {mean_us:.2f} μs")
    print(f"Median Decision Latency: {median_us:.2f} μs")
    print(f"99th Percentile Latency: {p99_us:.2f} μs")
    print(f"Max Latency:             {max_us:.2f} μs")
    print(f"Available Step Budget:   {CYCLE_BUDGET_SEC:,.0f} seconds")
    print(f"Real-Time Flight Margin: {margin_pct:.6f}%")

    df_timing = pd.DataFrame([{
        "Parameter_Count": param_count,
        "Memory_Bytes": param_bytes,
        "Mean_Latency_us": mean_us,
        "Median_Latency_us": median_us,
        "P99_Latency_us": p99_us,
        "Max_Latency_us": max_us,
        "Step_Budget_sec": CYCLE_BUDGET_SEC,
        "Real_Time_Margin_Pct": margin_pct
    }])
    df_timing.to_csv("artifacts/onboard_timing_results.csv", index=False)

    # Plot Latency Distribution
    plt.figure(figsize=(7, 4.5))
    plt.hist(latencies_us, bins=40, color="navy", edgecolor="black", alpha=0.7)
    plt.axvline(median_us, color="orange", linestyle="--", linewidth=2, label=f"Median ({median_us:.2f} μs)")
    plt.axvline(p99_us, color="red", linestyle=":", linewidth=2, label=f"99th Percentile ({p99_us:.2f} μs)")
    plt.xlabel("Guidance Inference Latency (microseconds)")
    plt.ylabel("Execution Count (out of 10,000)")
    plt.title("Onboard Neural Guidance Latency Distribution")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()

    plt.tight_layout()
    plt.savefig("artifacts/figures/onboard_latency_distribution.png", dpi=300)
    plt.close()
    print("Saved plot to artifacts/figures/onboard_latency_distribution.png\n")

if __name__ == "__main__":
    run()
