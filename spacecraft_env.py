import gymnasium as gym
import numpy as np
from gymnasium import spaces

class SpacecraftEnv(gym.Env):
    def __init__(self):
        super().__init__()

        # Thruster parameters - SPT-140, Test Point 4
        # Source: Hargus & Fife, AFRL/NASA Glenn, DTIC 2000
        self.Tmax = 0.289      # N
        self.Isp = 1782       # seconds
        self.efficiency = 0.551 # decimal
        self.t_max = 11040

        # Minimum and maximum values of the observation space
        self.obs_min = np.array([
            -249000000,-249000000,-249000000,
            0,0,0,
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

    def reset(self):
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
        
        return obs, {}

    def step(self, action):
        # Apply action to physics
        # Calculate reward
        # Check termination
        # Return all five values
        pass

    def _normalize(self, value, min_val, max_val):
        # Min-max scaling formula
        ans = (value-min_val)/(max_val-min_val)
        return ans