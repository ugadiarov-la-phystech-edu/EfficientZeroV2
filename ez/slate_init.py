from collections import namedtuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from PIL.Image import Resampling
from omegaconf import OmegaConf

from ocr.slate.slate import SLATE
from ocr.tools import obs_to_tensor

if __name__ == '__main__':
    ocr_config_path = 'ocr/slate/config/navigation5x5.yaml'
    obs_size = 64
    config_ocr = OmegaConf.load(ocr_config_path)
    config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(obs_size, 3)
    slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
    device = 'cuda'
    slate.to(device)

    checkpoint_path = 'navigation5x5/model_best_new.pth'
    state_dict = torch.load(checkpoint_path)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    image_path = './nav5x5_image.png'
    image_array = np.array(Image.open(image_path).resize((obs_size, obs_size), Resampling.BICUBIC))
    image_array = image_array[np.newaxis] # batch_size, height, width, channels
    image_tensor = obs_to_tensor(image_array, device=device)
    slots = slate._module._get_slots(image_tensor)
    print(f'slots {slots.size()}:', slots)
    viz = slate._module.get_samples(image_tensor)
    plt.imshow(viz['samples'][0])
    plt.show()
