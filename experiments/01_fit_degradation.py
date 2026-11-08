"""
Experiment 1: Thruster Degradation Model Fitting
Fits the logarithmic wear model T(h) = P0 - k * ln(1 + h / tau) to SPT-140 lab data.
Uses least squares (Calc AB optimization) and generates publication plots.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mission import thruster as T

os.makedirs("artifacts/figures", exist_ok=True)

def run():
    print("=== Experiment 1: Thruster Degradation Fitting ===")
    hours = T.WEAR_HOURS
    thrust = T.WEAR_THRUST_MN
    isp = T.WEAR_ISP_S

    # Fit thrust wear
    fit_thrust = T.fit_wear_model(hours, thrust)
    print(f"Thrust Fit: P0 = {fit_thrust['P0']:.2f} mN, k = {fit_thrust['k']:.4f}, tau = {fit_thrust['tau']:.1f} h, R^2 = {fit_thrust['r2']:.4f}")

    # Fit Isp wear
    fit_isp = T.fit_wear_model(hours, isp)
    print(f"Isp Fit:    P0 = {fit_isp['P0']:.2f} s,  k = {fit_isp['k']:.4f}, tau = {fit_isp['tau']:.1f} h, R^2 = {fit_isp['r2']:.4f}")

    # Generate curves over full 11,040 hour mission
    sim_hours = np.linspace(0, T.MISSION_HOURS, 500)
    pred_thrust = T.wear_curve(sim_hours, fit_thrust['P0'], fit_thrust['k'], fit_thrust['tau'])
    pred_isp = T.wear_curve(sim_hours, fit_isp['P0'], fit_isp['k'], fit_isp['tau'])

    # Save summary table
    df_fit = pd.DataFrame([
        {"Metric": "Thrust (mN)", "P0": fit_thrust['P0'], "k": fit_thrust['k'], "tau_h": fit_thrust['tau'], "R2": fit_thrust['r2'], "SSE": fit_thrust['sse']},
        {"Metric": "Isp (s)", "P0": fit_isp['P0'], "k": fit_isp['k'], "tau_h": fit_isp['tau'], "R2": fit_isp['r2'], "SSE": fit_isp['sse']}
    ])
    df_fit.to_csv("artifacts/degradation_fit_results.csv", index=False)
    print("Saved fit results to artifacts/degradation_fit_results.csv")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: Empirical data + fitted curve
    ax1.scatter(hours, thrust, color="crimson", zorder=5, label="Kamhawi et al. (2014) Data")
    ax1.plot(sim_hours[:100], pred_thrust[:100], "b--", label=f"Fit (R² = {fit_thrust['r2']:.4f})")
    ax1.plot(sim_hours, pred_thrust, "b-", alpha=0.7, label="Mission Extrapolation (460 d)")
    ax1.set_xlabel("Operating Time (hours)")
    ax1.set_ylabel("Thrust (mN)")
    ax1.set_title("SPT-140 Thruster Degradation Model")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # Right: SSE vs tau parameter landscape
    ax2.semilogx(fit_thrust["tau_grid"], fit_thrust["sse_curve"], color="purple")
    ax2.axvline(fit_thrust["tau"], color="green", linestyle="--", label=f"Optimal tau = {fit_thrust['tau']:.0f} h")
    ax2.set_xlabel("Time Constant tau (hours, log scale)")
    ax2.set_ylabel("Sum of Squared Errors (SSE)")
    ax2.set_title("Least-Squares Optimization Landscape")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.savefig("artifacts/figures/thruster_degradation_fit.png", dpi=300)
    plt.close()
    print("Saved plot to artifacts/figures/thruster_degradation_fit.png\n")

if __name__ == "__main__":
    run()
