import Gymnasium as gym
import numpy as np

class SpacecraftEnv(gym.Env):
    def __init__(self):
        # Observation space uses Box because...
        self.observation_space = spaces.Box(
            low=...,
            high=...,
            shape=...,
            dtype=...
        )

        # Action space
        self.action_space = spaces.Box(
            low=np.array([...]),
            high=np.array([...]),
            dtype=...
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