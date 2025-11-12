import gymnasium as gym
import numpy as np
from gymnasium import spaces
from astropy.time import Time, TimeDelta
from astropy.coordinates import get_body_barycentric_posvel
import astropy.units as u
import joblib
import pandas as pd
import warnings

class SpacecraftEnv(gym.Env):
    def __init__(self):
        super().__init__()
        
        # Loading Isolation Forest model
        self.anomaly_detector = joblib.load('isolation_forest.pkl')

        # Thruster parameters - SPT-140, Test Point 4
        # Source: Hargus & Fife, AFRL/NASA Glenn, DTIC 2000
        self.Tmax = 0.289      # N
        self.Isp = 1782       # seconds
        self.efficiency = 0.551 # decimal
        self.t_max = 11040
        self.mars_pos_table, self.mars_vel_table = self._precompute_ephemeris()
        self.current_step = 0
        self.mu_sun = 1.32712440018e11 # km^3 / s^2

        # Phase 1 Curriculum Toggle
        self.enable_anomalies = False

        # Minimum and maximum values of the observation space
        self.obs_min = np.array([
            -249000000,-249000000,-249000000,
            -401000000,-401000000,-401000000,
            -45,-45,-45,
            1648,
            0,
            1514.7
        ])
        self.obs_max = np.array([
            249000000,249000000,249000000,
            401000000,401000000,401000000,
            45,45,45,
            2747,
            11040,
            1782
        ])

        # Absolute spacecraft velocity relative to Sun (km/s)
        self.vel = np.array([
            # Earth's velocity at launch - from your astropy calculation
            19.16158263, -20.64057575, -8.94723395
        ])

        # Observation space - Box because the values are continuous streams of numbers and require the use of n closed intervals
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(12,),
            dtype=np.float32
        )

        # Action space - [thrust, azimuthal angle, polar angle]
        self.action_space = spaces.Box(
            low=np.array([0.0,0.0,0.0]),
            high=np.array([self.Tmax, 2 * np.pi, np.pi]),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        self.current_step = 0

        self.vel = np.array([
            19.16158263, -20.64057575, -8.94723395
        ])

        # Initialize raw state values
        self.state = np.array([
            # x, y, z position relative to Sun (km)          
            -127761765.999,67742862.352,29378238.56,
            # rx, ry, rz position relative to Mars (km)
            -88141567.361,42340646.216,26964853.96,
            # vx, vy, vz velocity relative to Mars (km/s)
            -4.3220554776,-6.377379144,-2.627783628,
            # m(t) instantaneous mass (kg)
            2747,
            # t time remaining (hours)
            11040,
            # Isp (s)
            1782
        ])
        
        # Normalize using min-max scaling
        obs = self._normalize(self.state,self.obs_min,self.obs_max)

       # Specific Orbital Energy Initialization
        v_mag = np.linalg.norm(self.vel)
        r_mag = np.linalg.norm(self.state[0:3])
        current_energy = (v_mag**2 / 2.0) - (self.mu_sun / r_mag)

        # Target Energy (Mars at step 0)
        target_v_mag = np.linalg.norm(self.mars_vel_table[0])
        target_r_mag = np.linalg.norm(self.mars_pos_table[0])
        target_energy = (target_v_mag**2 / 2.0) - (self.mu_sun / target_r_mag)

        # Initialize the error tracker
        self.prev_error = abs(target_energy - current_energy)

        # Heliocentric Phase Angle Initialization
        r_sc = self.state[0:3]
        r_mars = self.mars_pos_table[0]
        
        # Calculate initial angle using dot product
        cos_theta = np.dot(r_sc, r_mars) / (np.linalg.norm(r_sc) * np.linalg.norm(r_mars))
        cos_theta = np.clip(cos_theta, -1.0, 1.0) # Numerical stability
        self.prev_phase_angle = np.arccos(cos_theta)
        
        return obs, {}

    def step(self, action):
        # 1. Unpack action
        T = action[0]
        theta = action[1]
        phi = action[2]
        
        # 2. Convert spherical to Cartesian thrust vector
        Fx = T * np.sin(phi) * np.cos(theta)
        Fy = T * np.sin(phi) * np.sin(theta)
        Fz = T * np.cos(phi)
        
        # 3. Update velocity (a = F/m + g)
        dt = 3600  # seconds
        g0 = 9.80665  # m/s²
        mass = self.state[9]
        
        # Thruster acceleration (km/s^2)
        ax = (Fx / mass) / 1000
        ay = (Fy / mass) / 1000
        az = (Fz / mass) / 1000

        # Sun's gravitational acceleration (km/s^2)
        # r vector is self.state[0:3] (distance from sun)
        r_mag = np.linalg.norm(self.state[0:3])
        
        # a_g = -mu * r / |r|^3
        gx = -self.mu_sun * self.state[0] / (r_mag**3)
        gy = -self.mu_sun * self.state[1] / (r_mag**3)
        gz = -self.mu_sun * self.state[2] / (r_mag**3)

        # Update absolute velocity
        self.vel[0] += (ax + gx) * dt
        self.vel[1] += (ay + gy) * dt
        self.vel[2] += (az + gz) * dt

        # 4. Update position using ABSOLUTE velocity
        self.state[0] += self.vel[0] * dt
        self.state[1] += self.vel[1] * dt
        self.state[2] += self.vel[2] * dt

        # 5. Update mass
        m_dot = T / (self.state[11] * g0)
        self.state[9] -= m_dot * dt

        # 6. Update time remaining
        self.current_step += 1
        self.state[10] -= 1
        
        # 7. Update relative position to Mars
        current_mars_pos = self.mars_pos_table[self.current_step]
        self.state[3] = current_mars_pos[0] - self.state[0]
        self.state[4] = current_mars_pos[1] - self.state[1]
        self.state[5] = current_mars_pos[2] - self.state[2]

        current_mars_vel = self.mars_vel_table[self.current_step]
        self.state[6] = self.vel[0] - current_mars_vel[0]
        self.state[7] = self.vel[1] - current_mars_vel[1]
        self.state[8] = self.vel[2] - current_mars_vel[2]

        current_distance = np.linalg.norm(self.state[3:6])

        # Specific Orbital Energy Calculation (Agent)
        v_mag = np.linalg.norm(self.vel)
        r_mag = np.linalg.norm(self.state[0:3])
        current_energy = (v_mag**2 / 2.0) - (self.mu_sun / r_mag)

        # Specific Orbital Energy Calculation (Target Mars)
        target_v_mag = np.linalg.norm(self.mars_vel_table[self.current_step])
        target_r_mag = np.linalg.norm(self.mars_pos_table[self.current_step])
        target_energy = (target_v_mag**2 / 2.0) - (self.mu_sun / target_r_mag)

        # Calculate Energy Error Delta
        current_energy_error = abs(target_energy - current_energy)
        energy_error_delta = self.prev_error - current_energy_error
        
        # Heliocentric Phase Angle Calculation
        r_sc = self.state[0:3]
        r_mars = self.mars_pos_table[self.current_step]
        cos_theta = np.dot(r_sc, r_mars) / (r_mag * target_r_mag)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        current_phase_angle = np.arccos(cos_theta)
        
        # Calculate Phase Angle Delta
        phase_delta = self.prev_phase_angle - current_phase_angle

        # Multi-Objective Reward Blending
        # Scaling parameters (500.0, 1000.0) balance the gradient magnitude of energy vs. angle
        thrust_penalty = 0.1 * (T / self.Tmax)
        reward = (energy_error_delta * 500.0) + (phase_delta * 1000.0) - thrust_penalty
        
        # Update trackers for subsequent step
        self.prev_error = current_energy_error
        self.prev_phase_angle = current_phase_angle

        terminated = bool(self.state[9] <= self.obs_min[9] or self.state[10] <= self.obs_min[10] or current_distance < 577000)

        # Curriculum-controlled Anomaly Injection
        if self.enable_anomalies:
            solar_temp = np.random.normal(45, 0.5)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # Suppresses scikit-learn missing feature name warning
                anomaly_flag = self.anomaly_detector.predict([[self.state[11], solar_temp]])
            if anomaly_flag == -1:
                self.state[11] = 1514.7
        else:
            self.state[11] = 1782

        obs = self._normalize(self.state, self.obs_min, self.obs_max)
        return obs, reward, terminated, False, {}

    def _normalize(self, value, min_val, max_val):
        # Min-max scaling formula
        ans = (value-min_val)/(max_val-min_val)
        return ans

    def _precompute_ephemeris(self):
        launch = Time('2027-02-19')
        # Create a vectorized array of all timesteps at once
        times = launch + np.arange(11041) * u.hour
        
        # Get all positions and velocities in one call
        mars_states = get_body_barycentric_posvel('mars', times)
        
        # Extract values (in AU and AU/day) and convert to km and km/s
        # .T is used to get shape (11041, 3)
        mars_pos = mars_states[0].xyz.value.T * 149597870.7
        mars_vel = mars_states[1].xyz.value.T * 1731.4568

        return mars_pos, mars_vel