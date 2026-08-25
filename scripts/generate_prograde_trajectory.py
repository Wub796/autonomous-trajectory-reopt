"""
generate_prograde_trajectory.py — Generates a physically correct low-thrust
transfer trajectory using tangential (prograde) thrusting.

This replaces the RL-generated trajectory with a deterministic optimal
low-thrust transfer that actually reaches Mars's vicinity.

The strategy: thrust along the velocity vector (prograde) to raise the orbit
from Earth's ~1 AU to Mars's ~1.5 AU. This is the classic optimal steering
law for orbit-raising with continuous low-thrust propulsion.

Run from the project root:
    PYTHONPATH=. python scripts/generate_prograde_trajectory.py
"""
import os
import numpy as np
import pandas as pd
from src.env.spacecraft_env import SpacecraftEnv

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "artifacts")


def prograde_action(vel, T_max=0.289):
    """
    Compute the action [T, theta, phi] that thrusts along the velocity vector.
    
    theta = azimuthal angle in the x-y plane = atan2(vy, vx)
    phi = polar angle from the z-axis = acos(vz / |v|)
    """
    vx, vy, vz = vel
    v_mag = np.linalg.norm(vel)
    
    if v_mag < 1e-10:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    
    theta = np.arctan2(vy, vx)
    if theta < 0:
        theta += 2 * np.pi  # Ensure theta is in [0, 2*pi]
    
    phi = np.arccos(np.clip(vz / v_mag, -1.0, 1.0))
    
    return np.array([T_max, theta, phi], dtype=np.float32)


def generate_trajectory():
    env = SpacecraftEnv()
    obs, _ = env.reset()
    
    telemetry = []
    
    print("Generating prograde-thrust transfer trajectory...")
    print(f"Initial distance to Mars: {np.linalg.norm(env.state[3:6]):,.2f} km")
    
    min_dist = float('inf')
    min_dist_step = 0
    
    for step in range(11040):
        # Record state BEFORE taking action
        current_dist = np.linalg.norm(env.state[3:6])
        if current_dist < min_dist:
            min_dist = current_dist
            min_dist_step = step
        
        # Compute prograde thrust action
        action = prograde_action(env.vel, T_max=0.289)
        
        telemetry.append({
            "time_step_hr":     step,
            "sc_x_km":          env.state[0],
            "sc_y_km":          env.state[1],
            "sc_z_km":          env.state[2],
            "sc_vx_km_s":       env.vel[0],
            "sc_vy_km_s":       env.vel[1],
            "sc_vz_km_s":       env.vel[2],
            "mars_dist_km":     current_dist,
            "mass_kg":          env.state[9],
            "thrust_cmd_N":     float(action[0]),
            "thrust_theta_rad": float(action[1]),
            "thrust_phi_rad":   float(action[2]),
            "anomaly_active":   env.state[11] < 1782,
        })
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        if terminated:
            print(f"Simulation terminated at step {step}")
            # Record final state
            final_dist = np.linalg.norm(env.state[3:6])
            telemetry.append({
                "time_step_hr":     step + 1,
                "sc_x_km":          env.state[0],
                "sc_y_km":          env.state[1],
                "sc_z_km":          env.state[2],
                "sc_vx_km_s":       env.vel[0],
                "sc_vy_km_s":       env.vel[1],
                "sc_vz_km_s":       env.vel[2],
                "mars_dist_km":     final_dist,
                "mass_kg":          env.state[9],
                "thrust_cmd_N":     0.0,
                "thrust_theta_rad": 0.0,
                "thrust_phi_rad":   0.0,
                "anomaly_active":   env.state[11] < 1782,
            })
            break
        
        if step % 1000 == 0:
            sun_dist = np.linalg.norm(env.state[0:3])
            print(f"  Step {step:5d} | Sun dist: {sun_dist/1e6:.2f}M km | "
                  f"Mars dist: {current_dist/1e6:.2f}M km | "
                  f"Mass: {env.state[9]:.1f} kg | "
                  f"v_mag: {np.linalg.norm(env.vel):.3f} km/s")
    
    df = pd.DataFrame(telemetry)
    export_path = os.path.join(_ARTIFACTS_DIR, "optimal_mars_trajectory.csv")
    df.to_csv(export_path, index=False)
    
    print(f"\nTrajectory exported to {export_path}")
    print(f"Total steps: {len(telemetry)}")
    print(f"Final distance to Mars: {telemetry[-1]['mars_dist_km']:,.2f} km")
    print(f"Minimum distance to Mars: {min_dist:,.2f} km at step {min_dist_step}")
    print(f"Final spacecraft mass: {telemetry[-1]['mass_kg']:.1f} kg")
    print(f"Final sun distance: {np.sqrt(telemetry[-1]['sc_x_km']**2 + telemetry[-1]['sc_y_km']**2 + telemetry[-1]['sc_z_km']**2)/1e6:.2f}M km")


if __name__ == "__main__":
    generate_trajectory()
