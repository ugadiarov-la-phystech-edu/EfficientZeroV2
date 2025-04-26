import matplotlib.pyplot as plt
import numpy as np
import torch
from gym.wrappers import TimeLimit
from omegaconf import OmegaConf

from envs.causal_world.cw_envs import CwTargetEnv
from ocr.slate.slate import SLATE
from ocr.tools import obs_to_tensor

if __name__ == '__main__':
    env_config_path = 'envs/causal_world/cw_envs/config/reaching-hard_orig.yaml'
    env_config = OmegaConf.load(env_config_path)
    seed = 0
    env = CwTargetEnv(env_config, seed)
    env.action_space.seed(seed)
    env = TimeLimit(env, env.unwrapped._max_episode_length)

    ocr_config_path = 'ocr/slate/config/slate_3d.yaml'
    config_ocr = OmegaConf.load(ocr_config_path)
    slate = SLATE(config_ocr, env_config, observation_space=None, preserve_slot_order=True)
    device = 'cuda'
    slate.to(device)

    checkpoint_path = 'ocr/slate_weights/slate_3d.pth'
    state_dict = torch.load(checkpoint_path)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    obs = obs_to_tensor(env.reset()[np.newaxis], device=device)
    slots = [slate._module._get_slots(obs, prev_slots=None)]
    samples = [slate._module.get_samples(obs, prev_slots=None)]
    prev_slots = slots[-1]
    done = False
    while not done:
        obs, rew, done, info = env.step(env.action_space.sample())
        obs = obs_to_tensor(obs[np.newaxis], device=device)
        slots.append(slate._module._get_slots(obs, prev_slots=prev_slots))
        samples.append(slate._module.get_samples(obs, prev_slots=prev_slots))
        prev_slots = slots[-1]

    for i, sample in enumerate(samples):
        plt.imshow(sample['samples'][0])
        plt.savefig(f"sample_{i}.png")
