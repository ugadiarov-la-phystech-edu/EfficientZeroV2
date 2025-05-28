import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import gym

from envs.robosuite import RobosuiteEnv
from ocr.tools import obs_to_tensor, Dinosaur


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env, image_size):
        super().__init__(env)
        self._image_size = image_size
        assert len(env.observation_space.shape) == 3
        assert env.observation_space.shape[2] == 3
        assert env.observation_space.dtype == np.uint8
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(self._image_size, self._image_size, 3), dtype=np.uint8
        )

    def observation(self, observation):
        return cv2.resize(observation, dsize=(self._image_size, self._image_size), interpolation=cv2.INTER_CUBIC)


def load_slot_extractor_dinosaur(n_slots, slot_dim, checkpoint_path):
    model_name = 'vit_base_patch16_224_dino'
    input_feature_dim = 768
    num_patches = 196
    features = (2048, 2048, 2048)

    dinosaur = Dinosaur(dino_model_name=model_name, n_slots=n_slots, slot_dim=slot_dim,
                        intput_feature_dim=input_feature_dim, num_patches=num_patches, features=features)

    state_dict = torch.load(checkpoint_path)['state_dict']
    state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}

    dinosaur.load_state_dict(state_dict)
    dinosaur = dinosaur.eval()
    for param in dinosaur.parameters():
        param.requires_grad = False

    return dinosaur


if __name__ == '__main__':
    env = RobosuiteEnv(task='Lift', horizon=125, seed=0,)
    env = ResizeWrapper(env, image_size=224)

    device = 'cuda'
    checkpoint_path = 'ocr/robosuite.ckpt'
    dinosaur = load_slot_extractor_dinosaur(n_slots=5, slot_dim=64, checkpoint_path=checkpoint_path)
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
            cv2.imwrite(f'robosuite.png', cv2.cvtColor(sample[0], cv2.COLOR_RGB2BGR))
