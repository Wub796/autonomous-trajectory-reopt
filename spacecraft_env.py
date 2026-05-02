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

        # Observation space - Box because...
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
        # Initialize spacecraft state
        # Return initial observation
        pass

    def step(self, action):
        # Apply action to physics
        # Calculate reward
        # Check termination
        # Return all five values
        pass