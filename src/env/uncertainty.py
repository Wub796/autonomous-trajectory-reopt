"""
uncertainty.py — Stochastic disturbance and uncertainty modeling for spacecraft trajectory re-optimization.

Following the methodology of:
Alessandro Zavoli & Lorenzo Federici (2021),
"Reinforcement Learning for Robust Trajectory Design of Interplanetary Missions",
Journal of Guidance, Control, and Dynamics, Vol. 44, No. 8, pp. 1440–1453.
"""
from dataclasses import dataclass
import numpy as np


@dataclass
class UncertaintyConfig:
    """
    Configuration parameters for stochastic disturbances in orbital dynamics,
    sensor observations, and propulsion actuation.
    """
    # -------------------------------------------------------------------------
    # Process / State Dynamics Noise (Additive Gaussian on state propagation)
    # -------------------------------------------------------------------------
    # Position noise 1-sigma per step (km)
    sigma_pos_process_km: float = 0.0
    # Velocity noise 1-sigma per step (km/s) - e.g. 0.0001 km/s = 0.1 m/s
    sigma_vel_process_km_s: float = 0.0

    # -------------------------------------------------------------------------
    # Observation / Navigation Sensor Noise (Additive Gaussian on observed features)
    # -------------------------------------------------------------------------
    # Spacecraft heliocentric position observation noise 1-sigma (km)
    sigma_obs_sc_pos_km: float = 0.0
    # Target (Mars) relative position observation noise 1-sigma (km)
    sigma_obs_rel_pos_km: float = 0.0
    # Target relative velocity observation noise 1-sigma (km/s)
    sigma_obs_rel_vel_km_s: float = 0.0
    # Mass estimation uncertainty 1-sigma (kg)
    sigma_obs_mass_kg: float = 0.0

    # -------------------------------------------------------------------------
    # Control / Actuation Execution Errors
    # -------------------------------------------------------------------------
    # Thrust magnitude proportional error 1-sigma: T_act = T_cmd * (1 + delta_T)
    sigma_thrust_magnitude_pct: float = 0.0  # e.g., 0.02 for 2% 1-sigma error
    # Thrust pointing / attitude jitter 1-sigma (radians)
    sigma_pointing_jitter_rad: float = 0.0  # e.g., 0.0087 rad (~0.5 deg)

    # -------------------------------------------------------------------------
    # Missed Thrust Events (Thruster Failures / Safe-Mode Outages)
    # -------------------------------------------------------------------------
    # Probability per step of thruster temporary outage (Bernoulli trial)
    p_missed_thrust_step: float = 0.0
    # Duration range of safe-mode outage in hours (min, max)
    outage_duration_range_hr: tuple[int, int] = (1, 12)

    # -------------------------------------------------------------------------
    # Initial State Dispersion (Launch Injection 3-sigma Errors)
    # -------------------------------------------------------------------------
    # Heliocentric position injection dispersion 1-sigma (km)
    sigma_init_pos_km: float = 0.0
    # Heliocentric velocity injection dispersion 1-sigma (km/s)
    sigma_init_vel_km_s: float = 0.0
    # Initial mass uncertainty 1-sigma (kg)
    sigma_init_mass_kg: float = 0.0

    # Master enable toggle
    enabled: bool = True

    @classmethod
    def deterministic(cls) -> "UncertaintyConfig":
        """Deterministic baseline environment (zero noise)."""
        return cls(enabled=False)

    @classmethod
    def mild(cls) -> "UncertaintyConfig":
        """
        Mild disturbance profile:
        - Low navigation noise (10 km, 0.5 m/s)
        - Minor thrust magnitude error (1%)
        - Minor pointing jitter (0.25 deg)
        - Rare missed thrust (0.1% chance per step)
        """
        return cls(
            sigma_pos_process_km=5.0,
            sigma_vel_process_km_s=0.00005,  # 0.05 m/s
            sigma_obs_sc_pos_km=10.0,
            sigma_obs_rel_pos_km=15.0,
            sigma_obs_rel_vel_km_s=0.0005,   # 0.5 m/s
            sigma_obs_mass_kg=0.5,
            sigma_thrust_magnitude_pct=0.01,
            sigma_pointing_jitter_rad=float(np.deg2rad(0.25)),
            p_missed_thrust_step=0.001,
            outage_duration_range_hr=(1, 6),
            sigma_init_pos_km=50.0,
            sigma_init_vel_km_s=0.001,       # 1.0 m/s
            sigma_init_mass_kg=2.0,
            enabled=True,
        )

    @classmethod
    def zavoli_federici_2021(cls) -> "UncertaintyConfig":
        """
        Realistic uncertainty profile matching Zavoli & Federici (2021):
        - Process noise on orbital dynamics
        - DSN / OpNav measurement noise
        - 2.5% thrust magnitude execution error
        - 0.5 degree pointing dispersion
        - Stochastic thruster outages (0.5% chance per step)
        - Realistic launch C3 injection dispersion
        """
        return cls(
            sigma_pos_process_km=25.0,
            sigma_vel_process_km_s=0.0002,   # 0.2 m/s
            sigma_obs_sc_pos_km=50.0,
            sigma_obs_rel_pos_km=75.0,
            sigma_obs_rel_vel_km_s=0.002,    # 2 m/s
            sigma_obs_mass_kg=1.5,
            sigma_thrust_magnitude_pct=0.025,# 2.5%
            sigma_pointing_jitter_rad=float(np.deg2rad(0.5)), # 0.5 deg
            p_missed_thrust_step=0.005,      # 0.5% per hour
            outage_duration_range_hr=(2, 24),# 2 to 24 hr outages
            sigma_init_pos_km=250.0,         # Launch injection dispersion
            sigma_init_vel_km_s=0.005,       # 5.0 m/s launch dispersion
            sigma_init_mass_kg=5.0,
            enabled=True,
        )

    @classmethod
    def severe(cls) -> "UncertaintyConfig":
        """
        Severe stress-test profile:
        - High navigation noise (200 km, 10 m/s)
        - 5% thrust execution error
        - 1.5 degree pointing jitter
        - Frequent thruster outages (2% per step, up to 48 hours)
        - Substantial launch injection dispersions
        """
        return cls(
            sigma_pos_process_km=100.0,
            sigma_vel_process_km_s=0.001,    # 1.0 m/s
            sigma_obs_sc_pos_km=200.0,
            sigma_obs_rel_pos_km=300.0,
            sigma_obs_rel_vel_km_s=0.010,    # 10 m/s
            sigma_obs_mass_kg=5.0,
            sigma_thrust_magnitude_pct=0.05, # 5%
            sigma_pointing_jitter_rad=float(np.deg2rad(1.5)), # 1.5 deg
            p_missed_thrust_step=0.02,
            outage_duration_range_hr=(6, 48),
            sigma_init_pos_km=1000.0,
            sigma_init_vel_km_s=0.020,       # 20 m/s
            sigma_init_mass_kg=15.0,
            enabled=True,
        )


class DisturbanceModel:
    """
    Executes stochastic perturbations for state propagation, observation noise,
    actuation errors, and thruster outage tracking.
    """

    def __init__(self, config: UncertaintyConfig, rng: np.random.Generator | None = None):
        self.config = config
        self.rng = rng if rng is not None else np.random.default_rng()
        self.active_outage_steps_remaining: int = 0
        self.total_missed_steps: int = 0

    def reset(self, rng: np.random.Generator | None = None) -> None:
        """Reset internal outage trackers and optional random generator."""
        if rng is not None:
            self.rng = rng
        self.active_outage_steps_remaining = 0
        self.total_missed_steps = 0

    def sample_initial_state_perturbation(self) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Sample launch injection dispersion: (delta_pos, delta_vel, delta_mass).
        """
        if not self.config.enabled:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64), 0.0

        d_pos = self.rng.normal(0.0, self.config.sigma_init_pos_km, size=3)
        d_vel = self.rng.normal(0.0, self.config.sigma_init_vel_km_s, size=3)
        d_mass = float(self.rng.normal(0.0, self.config.sigma_init_mass_kg))
        return d_pos, d_vel, d_mass

    def apply_control_uncertainty(
        self,
        thrust_cmd: float,
        theta_cmd: float,
        phi_cmd: float,
        t_max: float,
    ) -> tuple[float, float, float, bool]:
        """
        Apply thrust execution errors, pointing jitter, and missed thrust events.
        
        Returns:
            (actual_thrust, actual_theta, actual_phi, is_missed_event)
        """
        if not self.config.enabled:
            return thrust_cmd, theta_cmd, phi_cmd, False

        # Check ongoing outage or trigger new outage
        if self.active_outage_steps_remaining > 0:
            self.active_outage_steps_remaining -= 1
            self.total_missed_steps += 1
            return 0.0, theta_cmd, phi_cmd, True
        elif self.config.p_missed_thrust_step > 0.0:
            if self.rng.random() < self.config.p_missed_thrust_step:
                dur_min, dur_max = self.config.outage_duration_range_hr
                self.active_outage_steps_remaining = int(self.rng.integers(dur_min, dur_max + 1)) - 1
                self.total_missed_steps += 1
                return 0.0, theta_cmd, phi_cmd, True

        # Thrust magnitude error: T_act = T_cmd * (1 + delta_T)
        if self.config.sigma_thrust_magnitude_pct > 0.0 and thrust_cmd > 1e-6:
            scale_error = self.rng.normal(0.0, self.config.sigma_thrust_magnitude_pct)
            actual_thrust = float(np.clip(thrust_cmd * (1.0 + scale_error), 0.0, t_max))
        else:
            actual_thrust = thrust_cmd

        # Pointing jitter: theta, phi perturbations
        if self.config.sigma_pointing_jitter_rad > 0.0:
            d_theta = float(self.rng.normal(0.0, self.config.sigma_pointing_jitter_rad))
            d_phi = float(self.rng.normal(0.0, self.config.sigma_pointing_jitter_rad))
            actual_theta = float((theta_cmd + d_theta) % (2.0 * np.pi))
            actual_phi = float(np.clip(phi_cmd + d_phi, 0.0, np.pi))
        else:
            actual_theta = theta_cmd
            actual_phi = phi_cmd

        return actual_thrust, actual_theta, actual_phi, False

    def sample_process_noise(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Sample additive state propagation process noise (pos_noise, vel_noise).
        """
        if not self.config.enabled:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        w_pos = self.rng.normal(0.0, self.config.sigma_pos_process_km, size=3)
        w_vel = self.rng.normal(0.0, self.config.sigma_vel_process_km_s, size=3)
        return w_pos, w_vel

    def apply_observation_noise(self, raw_state: np.ndarray) -> np.ndarray:
        """
        Inject navigation sensor measurement noise into the raw physical state vector
        before normalization.
        
        raw_state layout:
          0..2:   sc_pos (x, y, z) [km]
          3..5:   rel_pos to Mars (rx, ry, rz) [km]
          6..8:   rel_vel to Mars (vx, vy, vz) [km/s]
          9:      mass [kg]
          10:     time remaining [hr]
          11:     Isp [s]
        """
        if not self.config.enabled:
            return raw_state.copy()

        noisy_state = raw_state.copy()
        
        # Sc pos noise
        if self.config.sigma_obs_sc_pos_km > 0.0:
            noisy_state[0:3] += self.rng.normal(0.0, self.config.sigma_obs_sc_pos_km, size=3)

        # Rel pos noise
        if self.config.sigma_obs_rel_pos_km > 0.0:
            noisy_state[3:6] += self.rng.normal(0.0, self.config.sigma_obs_rel_pos_km, size=3)

        # Rel vel noise
        if self.config.sigma_obs_rel_vel_km_s > 0.0:
            noisy_state[6:9] += self.rng.normal(0.0, self.config.sigma_obs_rel_vel_km_s, size=3)

        # Mass noise
        if self.config.sigma_obs_mass_kg > 0.0:
            noisy_state[9] = max(100.0, noisy_state[9] + float(self.rng.normal(0.0, self.config.sigma_obs_mass_kg)))

        return noisy_state
