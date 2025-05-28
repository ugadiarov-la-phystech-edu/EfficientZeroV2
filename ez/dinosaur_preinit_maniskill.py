import matplotlib.pyplot as plt
import numpy as np
import torch
import gym
from gym.wrappers import TimeLimit
import cv2

from envs.maniskill3 import ManiSkill
from ocr.tools import obs_to_tensor, Dinosaur


class FailOnTimelimitWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        observation, reward, done, info = super().step(action)
        if done and 'is_success' not in info:
            info['is_success'] = False

        return observation, reward, done, info


def load_slot_extractor_dinosaur(n_slots, slot_dim, checkpoint_path):
    model_name = 'vit_small_patch8_224_dino'
    input_feature_dim = 384
    num_patches = 784
    features = (1024,1024,1024)

    dinosaur = Dinosaur(dino_model_name=model_name, n_slots=n_slots, slot_dim=slot_dim,
                        intput_feature_dim=input_feature_dim, num_patches=num_patches, features=features)

    state_dict = torch.load(checkpoint_path)['state_dict']
    state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}

    dinosaur.load_state_dict(state_dict)
    dinosaur = dinosaur.eval()
    dinosaur.requires_grad_(False)

    return dinosaur


if __name__ == '__main__':
    time_limit = 50
    env = ManiSkill(reward_mode='normalized_dense', image_size=224)
    env = TimeLimit(env, max_episode_steps=time_limit)
    env = FailOnTimelimitWrapper(env)

    device = 'cuda'
    checkpoint_path = 'ocr/maniskill.ckpt'
    dinosaur = load_slot_extractor_dinosaur(n_slots=4, slot_dim=128, checkpoint_path=checkpoint_path)
    dinosaur.to(device)

    obs = obs_to_tensor(env.reset()[np.newaxis], device=device)
    slots = [dinosaur(obs, prev_slots=None)]
    samples = [dinosaur.get_samples(obs, prev_slots=None)]
    prev_slots = slots[-1]
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        obs = obs_to_tensor(obs[np.newaxis], device=device)
        slots.append(dinosaur(obs, prev_slots=prev_slots))
        samples.append(dinosaur.get_samples(obs, prev_slots=prev_slots))
        prev_slots = slots[-1]

    for i, sample in enumerate(samples):
        if i == len(samples)-1:
            cv2.imwrite(f'maniskill.png', cv2.cvtColor(sample[0], cv2.COLOR_RGB2BGR))