"""
robust_spacecraft_env.py — High-fidelity stochastic spacecraft trajectory environment.

Incorporates state process noise, navigation observation noise, actuation execution errors,
and stochastic thruster outages following Zavoli & Federici (2021).
"""
import os
import warnings
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from astropy.time import Time
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import joblib

from src.env.uncertainty import UncertaintyConfig, DisturbanceModel

_ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "artifacts")


class RobustSpacecraftEnv(gym.Env):
    """
    Stochastic gymnasium environment for autonomous interplanetary trajectory optimization
    under partial observability and dynamical disturbances.
    """
    metadata = {"render_modes": []}

    def __init__(
        self,
        uncertainty_config: UncertaintyConfig | None = None,
        enable_anomalies: bool = False,
        seed: int | None = None,
    ):
        super().__init__()

        # Disturbance configuration (defaults to Zavoli & Federici 2021 if None)
        self.uncertainty_config = (
            uncertainty_config if uncertainty_config is not None
            else UncertaintyConfig.zavoli_federici_2021()
        )
        self.np_random = np.random.default_rng(seed)
        self.disturbance_model = DisturbanceModel(self.uncertainty_config, self.np_random)

        # Loading Isolation Forest model
        iso_path = os.path.join(_ARTIFACTS_DIR, "isolation_forest.pkl")
        if os.path.exists(iso_path):
            self.anomaly_detector = joblib.load(iso_path)
        else:
            self.anomaly_detector = None

        # Thruster parameters - SPT-140, Test Point 4 (Hargus & Fife, AFRL/NASA Glenn)
        self.Tmax = 0.289        # N
        self.Isp_nominal = 1782  # seconds
        self.Isp_failed = 1514.7 # seconds
        self.efficiency = 0.551  # decimal
        self.t_max = 11040       # hours (460 days)
        self.mu_sun = 1.32712440018e11  # km^3 / s^2
        self.g0 = 9.80665        # m/s^2
        self.dt = 3600.0         # seconds per step (1 hour)
        self.capture_radius_km = 577000.0 # Mars capture sphere of influence

        # Precompute Mars ephemeris table
        self.mars_pos_table, self.mars_vel_table = self._precompute_ephemeris()
        self.current_step = 0
        self.enable_anomalies = enable_anomalies

        # Minimum and maximum values of the physical state space for min-max scaling
        self.obs_min = np.array([
            -249000000.0, -249000000.0, -249000000.0,
            -401000000.0, -401000000.0, -401000000.0,
            -45.0, -45.0, -45.0,
            1648.0,
            0.0,
            1514.7
        ], dtype=np.float64)

        self.obs_max = np.array([
            249000000.0, 249000000.0, 249000000.0,
            401000000.0, 401000000.0, 401000000.0,
            45.0, 45.0, 45.0,
            2747.0,
            11040.0,
            1782.0
        ], dtype=np.float64)

        # Baseline nominal launch state (Earth departure on 2027-02-19)
        self.nominal_init_pos = np.array([-127761765.999, 67742862.352, 29378238.56], dtype=np.float64)
        self.nominal_init_vel = np.array([-15.4827348, -23.69526119, -10.27023734], dtype=np.float64)
        self.nominal_init_mass = 2747.0

        # State storage
        self.state = np.zeros(12, dtype=np.float64) # True physical state
        self.vel = np.zeros(3, dtype=np.float64)   # True heliocentric velocity
        self.last_observed_state = np.zeros(12, dtype=np.float64)
        self.prev_error = 0.0
        self.prev_phase_angle = 0.0

        # Observation space [0, 1]^12
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(12,),
            dtype=np.float32
        )

        # Action space: [thrust (N), azimuthal angle theta (rad), polar angle phi (rad)]
        self.action_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([self.Tmax, 2 * np.pi, np.pi], dtype=np.float32),
            dtype=np.float32
        )

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)

        self.disturbance_model.reset(self.np_random)
        self.current_step = 0

        # 1. Sample launch injection dispersion
        d_pos, d_vel, d_mass = self.disturbance_model.sample_initial_state_perturbation()

        # 2. Initialize true physical state
        self.vel = self.nominal_init_vel + d_vel
        init_pos = self.nominal_init_pos + d_pos
        init_mass = max(1648.0, self.nominal_init_mass + d_mass)

        # Relative state to Mars at launch
        init_rel_pos = self.mars_pos_table[0] - init_pos
        init_rel_vel = self.vel - self.mars_vel_table[0]

        self.state = np.array([
            init_pos[0], init_pos[1], init_pos[2],
            init_rel_pos[0], init_rel_pos[1], init_rel_pos[2],
            init_rel_vel[0], init_rel_vel[1], init_rel_vel[2],
            init_mass,
            float(self.t_max),
            float(self.Isp_nominal),
        ], dtype=np.float64)

        # 3. Specific orbital energy and phase angle initialization
        v_mag = float(np.linalg.norm(self.vel))
        r_mag = float(np.linalg.norm(self.state[0:3]))
        current_energy = (v_mag**2 / 2.0) - (self.mu_sun / r_mag)

        target_v_mag = float(np.linalg.norm(self.mars_vel_table[0]))
        target_r_mag = float(np.linalg.norm(self.mars_pos_table[0]))
        target_energy = (target_v_mag**2 / 2.0) - (self.mu_sun / target_r_mag)
        self.prev_error = abs(target_energy - current_energy)

        r_sc = self.state[0:3]
        r_mars = self.mars_pos_table[0]
        cos_theta = np.dot(r_sc, r_mars) / (r_mag * target_r_mag)
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        self.prev_phase_angle = float(np.arccos(cos_theta))

        # 4. Generate noisy initial observation
        raw_obs = self.disturbance_model.apply_observation_noise(self.state)
        self.last_observed_state = raw_obs
        obs = self._normalize(raw_obs, self.obs_min, self.obs_max).astype(np.float32)

        info = {
            "true_state": self.state.copy(),
            "noisy_state": raw_obs.copy(),
            "mars_dist_km": float(np.linalg.norm(self.state[3:6])),
            "mass_kg": float(self.state[9]),
        }

        return obs, info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        # 1. Unpack commanded action
        cmd_T = float(action[0])
        cmd_theta = float(action[1])
        cmd_phi = float(action[2])

        # 2. Apply control / actuation uncertainty & missed thrust events
        act_T, act_theta, act_phi, is_missed = self.disturbance_model.apply_control_uncertainty(
            cmd_T, cmd_theta, cmd_phi, self.Tmax
        )

        # 3. Convert spherical to Cartesian thrust vector
        Fx = act_T * np.sin(act_phi) * np.cos(act_theta)
        Fy = act_T * np.sin(act_phi) * np.sin(act_theta)
        Fz = act_T * np.cos(act_phi)

        # 4. Dynamics Update with Process Noise
        mass = self.state[9]
        isp = self.state[11]

        # Thruster acceleration (km/s^2)
        ax = (Fx / mass) / 1000.0
        ay = (Fy / mass) / 1000.0
        az = (Fz / mass) / 1000.0

        # Sun's gravitational acceleration (km/s^2)
        r_mag = float(np.linalg.norm(self.state[0:3]))
        gx = -self.mu_sun * self.state[0] / (r_mag**3)
        gy = -self.mu_sun * self.state[1] / (r_mag**3)
        gz = -self.mu_sun * self.state[2] / (r_mag**3)

        # Sample process noise on dynamics
        w_pos, w_vel = self.disturbance_model.sample_process_noise()

        # Update absolute velocity (including process noise)
        self.vel[0] += (ax + gx) * self.dt + w_vel[0]
        self.vel[1] += (ay + gy) * self.dt + w_vel[1]
        self.vel[2] += (az + gz) * self.dt + w_vel[2]

        # Update position (including process noise)
        self.state[0] += self.vel[0] * self.dt + w_pos[0]
        self.state[1] += self.vel[1] * self.dt + w_pos[1]
        self.state[2] += self.vel[2] * self.dt + w_pos[2]

        # Update propellant mass
        if act_T > 1e-6:
            m_dot = act_T / (isp * self.g0)
            self.state[9] -= m_dot * self.dt
            self.state[9] = max(self.state[9], self.obs_min[9])

        # Step counter & mission time
        self.current_step += 1
        self.state[10] -= 1.0

        # Update relative position & velocity to Mars
        current_mars_pos = self.mars_pos_table[self.current_step]
        current_mars_vel = self.mars_vel_table[self.current_step]

        self.state[3] = current_mars_pos[0] - self.state[0]
        self.state[4] = current_mars_pos[1] - self.state[1]
        self.state[5] = current_mars_pos[2] - self.state[2]

        self.state[6] = self.vel[0] - current_mars_vel[0]
        self.state[7] = self.vel[1] - current_mars_vel[1]
        self.state[8] = self.vel[2] - current_mars_vel[2]

        current_distance = float(np.linalg.norm(self.state[3:6]))

        # Specific Orbital Energy Calculation
        v_mag = float(np.linalg.norm(self.vel))
        r_mag = float(np.linalg.norm(self.state[0:3]))
        current_energy = (v_mag**2 / 2.0) - (self.mu_sun / r_mag)

        target_v_mag = float(np.linalg.norm(current_mars_vel))
        target_r_mag = float(np.linalg.norm(current_mars_pos))
        target_energy = (target_v_mag**2 / 2.0) - (self.mu_sun / target_r_mag)

        current_energy_error = abs(target_energy - current_energy)
        energy_error_delta = self.prev_error - current_energy_error

        # Heliocentric Phase Angle Calculation
        r_sc = self.state[0:3]
        r_mars = current_mars_pos
        cos_theta = np.dot(r_sc, r_mars) / (r_mag * target_r_mag)
        cos_theta = float(np.clip(cos_theta, -1.0, 1.0))
        current_phase_angle = float(np.arccos(cos_theta))
        phase_delta = self.prev_phase_angle - current_phase_angle

        # Multi-Objective Reward Blending
        thrust_penalty = 0.1 * (act_T / self.Tmax)
        reward = (energy_error_delta * 500.0) + (phase_delta * 1000.0) - thrust_penalty

        if current_distance < self.capture_radius_km:
            reward += 100000.0

        # Update trackers
        self.prev_error = current_energy_error
        self.prev_phase_angle = current_phase_angle

        terminated = bool(
            self.state[9] <= self.obs_min[9]
            or self.state[10] <= self.obs_min[10]
            or current_distance < self.capture_radius_km
        )

        # Anomaly Injection Check
        if self.enable_anomalies and self.anomaly_detector is not None:
            solar_temp = self.np_random.normal(45.0, 0.5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                anomaly_flag = self.anomaly_detector.predict([[self.state[11], solar_temp]])
            if anomaly_flag == -1:
                self.state[11] = self.Isp_failed
        elif not self.enable_anomalies:
            self.state[11] = self.Isp_nominal

        # 5. Generate Noisy Observation
        raw_obs = self.disturbance_model.apply_observation_noise(self.state)
        self.last_observed_state = raw_obs
        obs = self._normalize(raw_obs, self.obs_min, self.obs_max).astype(np.float32)

        info = {
            "true_state": self.state.copy(),
            "noisy_state": raw_obs.copy(),
            "cmd_action": np.array([cmd_T, cmd_theta, cmd_phi]),
            "actual_action": np.array([act_T, act_theta, act_phi]),
            "is_missed_thrust": is_missed,
            "mars_dist_km": current_distance,
            "mass_kg": float(self.state[9]),
            "energy_error": current_energy_error,
            "phase_angle_rad": current_phase_angle,
            "total_missed_steps": self.disturbance_model.total_missed_steps,
        }

        return obs, reward, terminated, False, info

    def _normalize(self, value: np.ndarray, min_val: np.ndarray, max_val: np.ndarray) -> np.ndarray:
        """Min-max scaling to [0, 1]."""
        return np.clip((value - min_val) / (max_val - min_val), 0.0, 1.0)

    def _precompute_ephemeris(self) -> tuple[np.ndarray, np.ndarray]:
        launch = Time('2027-02-19')
        times = launch + np.arange(self.t_max + 1) * u.hour
        mars_pos_bary, mars_vel_bary = get_body_barycentric_posvel('mars', times)
        mars_pos = mars_pos_bary.xyz.value.T * 149597870.7  # AU to km
        mars_vel = mars_vel_bary.xyz.value.T * 1731.4568    # AU/day to km/s
        return mars_pos, mars_vel
