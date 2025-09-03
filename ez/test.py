from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from omegaconf import OmegaConf
import cv2
from envs.wrapper import *
from ez.envs.shapes2d import shapes2d

from ocr.slate.slate import SLATE
from ocr.tools import obs_to_tensor

if __name__ == '__main__':
    seed = 0
    env = gym.make('Navigation5x5-v0')
    env = WarpFrame(env, width=64, height=64, grayscale=False)
    env.seed(seed)

    ocr_config_path = 'ocr/slate/config/navigation5x5.yaml'
    obs_size = 64
    config_ocr = OmegaConf.load(ocr_config_path)
    config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(obs_size, 3)
    slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
    device = 'cuda'
    slate.to(device)

    checkpoint_path = 'ocr/slate_weights/navigation5х5.pth'
    state_dict = torch.load(checkpoint_path)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    obs = obs_to_tensor(env.reset()[np.newaxis], device=device)
    slots = [slate._module._get_slots(obs, prev_slots=None)]
    samples = [slate._module.get_samples(obs, prev_slots=None)]
    prev_slots = slots[-1]
    done = False
    for i in range(2):
        obs, rew, done, info = env.step(1)
        obs = obs_to_tensor(obs[np.newaxis], device=device)
        slots.append(slate._module._get_slots(obs, prev_slots=prev_slots))
        samples.append(slate._module.get_samples(obs, prev_slots=prev_slots))
        prev_slots = slots[-1]

        cv2.imwrite(f'pic{i}.png', cv2.cvtColor(samples[-1]['samples'][0], cv2.COLOR_RGB2BGR))

    for i in range(6):
        print(sum(abs(slots[0][0][i] - slots[2][0][i])))