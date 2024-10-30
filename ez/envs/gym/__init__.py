from ..base import BaseWrapper
import gym
from gymnasium.spaces import Box, Discrete 

class GymWrapper(BaseWrapper):
    """
    Make your own wrapper: Atari Wrapper
    """
    def __init__(self, env, obs_to_string=False):
        super().__init__(env, obs_to_string, False)
        self.observation_space = self.convert_space(self.env.observation_space)
        self.action_space = self.convert_space(self.env.action_space)

    def convert_space(self, space):
        if isinstance(space, Box):
            return gym.spaces.Box(low=space.low, high=space.high, shape=space.shape, dtype=space.dtype)
        elif isinstance(space, Discrete):
            return gym.spaces.Discrete(space.n)

    def step(self, action):
        obs, reward, _, done, info = self.env.step(action)
        info['raw_reward'] = reward
        return obs, reward, done, info

    def reset(self,):
        obs, info = self.env.reset()

        return obs