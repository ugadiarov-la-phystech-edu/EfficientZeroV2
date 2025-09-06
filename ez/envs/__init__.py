import os
import dmc2gym
#from gym.wrappers import Monitor
from .gym import GymWrapper
from .atari import AtariWrapper
from .dmc import DMCWrapper
from .wrapper import *
import random
from dm_env import specs
from ez.utils.format import arr_to_str
from ez.envs.shapes2d import shapes2d
from omegaconf import OmegaConf
from ez.envs.causal_world.cw_envs import CwTargetEnv
from ez.envs.maniskill3 import ManiSkill
from ez.envs.robosuite import RobosuiteEnv
from ez.agents.models.base_model import OCRepresentationNetworkDINOSAUR, OCRepresentationNetworkSLATE
from ez.ocr.tools import SlotExtractor, Dinosaur
from collections import namedtuple
import torch
from ez.ocr.slate.slate import SLATE


def make_envs(game_setting, game_name, num_envs, seed, save_path=None, **kwargs):
    assert game_setting in ['Atari', 'DMC', 'Gym', 'Shapes2d', 'causal_world', 'robosuite', 'maniskill']
    if game_setting == 'Atari':
        _env_fn = make_atari
    elif game_setting == 'Gym':
        _env_fn = make_gym
    elif game_setting == 'DMC':
        _env_fn = make_dmc  
    elif game_setting == 'Shapes2d':
        _env_fn = make_shapes2d
    elif game_setting == 'causal_world':
        _env_fn = make_causal_world
        env_setting = 'ez/envs/causal_world/cw_envs/config/reaching-hard_orig.yaml'
    elif game_setting == 'robosuite':
        _env_fn = make_robosuite
    elif game_setting == 'maniskill':
        _env_fn = make_maniskill
    else:
        raise NotImplementedError()

    if game_setting in ['DMC']:
        seed = random.randint(1, 1000)

    if game_setting == 'causal_world':
        envs = [_env_fn(env_setting,
                        seed=i + seed,
                        save_path=save_path, **kwargs) for i in range(num_envs)]
    elif game_setting == 'maniskill':
        envs = [_env_fn(seed=i + seed, **kwargs) for i in range(num_envs)]
    else:
        envs = [_env_fn(game_name,
                        seed=i + seed,
                        # seed=seed,
                        save_path=save_path, **kwargs) for i in range(num_envs)]

    return envs


def make_env(game_setting, game_name, num_envs, seed, save_path=None, **kwargs):
    assert game_setting in ['Atari', 'DMC', 'Gym', 'Shapes2d', 'causal_world', 'robosuite', 'maniskill']
    if game_setting == 'Atari':
        _env_fn = make_atari
    elif game_setting == 'Gym':
        _env_fn = make_gym
    elif game_setting == 'DMC':
        _env_fn = make_dmc
    elif game_setting == 'Shapes2d':
        _env_fn = make_causal_world
    elif game_setting == 'causal_world':
        _env_fn = make_causal_world
        env_setting = 'ez/envs/causal_world/cw_envs/config/reaching-hard_orig.yaml'
    elif game_setting == 'robosuite':
        _env_fn = make_robosuite
    elif game_setting == 'maniskill':
        _env_fn = make_maniskill
    else:
        raise NotImplementedError()

    seed = random.randint(1, 1000)

    if game_setting == 'causal_world':
        env = _env_fn(env_setting, seed=seed, save_path=save_path, **kwargs)
    elif game_setting == 'maniksill':
        env = _env_fn(seed = seed, **kwargs)
    else:
        env = _env_fn(game_name, seed=seed, save_path=save_path, **kwargs)

    return env


def make_atari(game_name, seed, save_path=None, **kwargs):
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        skip: int
            frame skip
        obs_shape: (int, int)
            observation shape
        gray_scale: bool
            use gray observation or rgb observation
        seed: int
            seed of env
        max_episode_steps: int
            max moves for an episode
        save_path: str
            the path of saved videos; do not save video if None
            :param seed:
    """
    # params
    env_id = game_name + 'NoFrameskip-v4'
    gray_scale = kwargs.get('gray_scale')
    obs_to_string = kwargs.get('obs_to_string')
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 4
    obs_shape = kwargs['obs_shape'] if kwargs.get('obs_shape') else [3, 96, 96]
    max_episode_steps = kwargs['max_episode_steps'] if kwargs.get('max_episode_steps') else 108000 // skip
    episodic_life = kwargs.get('episodic_life')
    clip_reward = kwargs.get('clip_reward')

    env = gym.make(env_id)

    # set seed
    env.seed(seed)

    # random restart
    env = NoopResetEnv(env, noop_max=30)

    # frame skip
    env = MaxAndSkipEnv(env, skip=skip)

    # episodic trajectory
    if episodic_life:
        env = EpisodicLifeEnv(env)

    # reshape size and gray scale
    env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)
    
    # set max limit
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    # save video to given
    # if save_path:
    #     env = Monitor(env, directory=save_path, force=True)

    # your wrapper
    env = AtariWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)
    return env


def make_gym(game_name, seed, save_path=None, **kwargs):
    save_path = kwargs.get('save_path')
    obs_to_string = kwargs.get('obs_to_string')
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 4
    gray_scale = kwargs.get('gray_scale')
    obs_shape = kwargs['obs_shape']
    max_episode_steps = kwargs['max_episode_steps']

    env = gym.make(game_name)
    env = GymWrapper(env, obs_to_string=obs_to_string)

    #frame skip
    env = MaxAndSkipEnv(env, skip=skip)

    # set seed
    env.seed(seed)
    
    #save video to given
    # if save_path:
    #     env = Monitor(env, directory=save_path, force=True)
    env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)
    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = GymWrapper(env, obs_to_string=obs_to_string)
    return env


def make_dmc(game_name, seed, save_path=None, **kwargs):
    """Make Atari games
    Parameters
    ----------
    game_name: str
        name of game (Such as Breakout, Pong)
    kwargs: dict
        image_based: bool
            observation is image or state

    """
    # params
    if 'CMU' in game_name:
        domain_name, task_name = game_name.rsplit('_', 1)
    else:
        domain_name, task_name = game_name.split('_', 1)
    image_based = kwargs.get('image_based')
    obs_shape = kwargs['obs_shape'] if kwargs.get('obs_shape') else [3, 96, 96]
    skip = kwargs['n_skip'] if kwargs.get('n_skip') else 2
    max_episode_steps = kwargs['max_episode_steps'] // skip
    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    # fix the bug of env (from the paper DrQv2)
    camera_id = 2 if 'quadruped' in domain_name else 0

    # # make env
    env = dmc2gym.make(
        domain_name=domain_name,
        task_name=task_name,
        seed=seed,
        visualize_reward=False,
        from_pixels=image_based,
        height=obs_shape[1] if image_based else 96,
        width=obs_shape[1] if image_based else 96,
        frame_skip=skip,
        channels_first=False,
        camera_id=camera_id,
        # time_limit=max_episode_steps,
    )

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = DMCWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)
    return env

def make_shapes2d(game_name, seed, save_path=None, **kwargs):

    gray_scale = kwargs.get('gray_scale')
    obs_shape = kwargs['obs_shape']
    max_episode_steps = kwargs['max_episode_steps']
    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    num_slots = kwargs.get('n_slots')
    slot_dim = kwargs.get('slot_dim')
    ocr_config_path = kwargs.get('ocr_config_path')
    checkpoint_path = kwargs.get('checkpoint_path')

    env = gym.make(game_name)

    env.seed(seed)

    env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)

    #env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = AtariWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)

    config_ocr = OmegaConf.load(ocr_config_path)
    config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(obs_shape[2], 3)
    slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(checkpoint_path)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    slot_extractor = SlotExtractor(model=slate, device='cuda', name_model = 'SLATE')

    env = SlotExtractorWrapper(env, slot_extractor, num_slots, slot_dim)
    return env

def make_causal_world(env_config_path, seed, save_path=None, **kwargs):

    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    obs_shape = kwargs['obs_shape']
    gray_scale = kwargs.get('gray_scale')
    num_slots = kwargs.get('n_slots')
    slot_dim = kwargs.get('slot_dim')
    ocr_config_path = kwargs.get('ocr_config_path')
    checkpoint_path = kwargs.get('checkpoint_path')

    env_config = OmegaConf.load(env_config_path)
    env = CwTargetEnv(env_config, seed)

    env.action_space.seed(seed)

    env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)

    env = TimeLimit(env, env.unwrapped._max_episode_length)

    env = DMCWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)

    config_ocr = OmegaConf.load(ocr_config_path)
    config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(obs_shape[2], 3)
    slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
    state_dict = torch.load(checkpoint_path)["ocr_module_state_dict"]
    slate._module.load_state_dict(state_dict)
    slate.requires_grad_(False)
    slate.eval()

    slot_extractor = SlotExtractor(model=slate, device='cuda', name_model = 'SLATE')

    env = SlotExtractorWrapper(env, slot_extractor, num_slots, slot_dim)
    return env

def make_robosuite(game_name, seed, save_path=None, **kwargs):

    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    max_episode_steps = kwargs['max_episode_steps']
    obs_shape = kwargs['obs_shape']
    gray_scale = kwargs.get('gray_scale')
    num_slots = kwargs.get('n_slots')
    slot_dim = kwargs.get('slot_dim')
    model_name = kwargs.get('model_name')
    input_feature_dim = kwargs.get('input_feature_dim')
    num_patches = kwargs.get('num_patches')
    features_size = kwargs.get('features')
    features = (features_size, features_size, features_size)
    checkpoint_path = kwargs.get('checkpoint_path')

    env = RobosuiteEnv(task=game_name, horizon=max_episode_steps, seed=seed)

    env = WarpFrame(env, width=obs_shape[1], height=obs_shape[2], grayscale=gray_scale)

    env = DMCWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)

    dinosaur = Dinosaur(dino_model_name=model_name, n_slots=num_slots, slot_dim=slot_dim,
                        intput_feature_dim=input_feature_dim, num_patches=num_patches, features=features)

    state_dict = torch.load(checkpoint_path)['state_dict']
    state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}

    dinosaur.load_state_dict(state_dict)
    dinosaur = dinosaur.eval()
    dinosaur.requires_grad_(False)

    slot_extractor = SlotExtractor(model=dinosaur, device='cuda', name_model = 'DINOSAUR')

    env = SlotExtractorWrapper(env, slot_extractor, num_slots, slot_dim)
    return env

def make_maniskill(seed, **kwargs):

    clip_reward = kwargs.get('clip_reward')
    obs_to_string = kwargs.get('obs_to_string')
    max_episode_steps = kwargs['max_episode_steps']
    obs_shape = kwargs['obs_shape']
    num_slots = kwargs.get('n_slots')
    slot_dim = kwargs.get('slot_dim')
    model_name = kwargs.get('model_name')
    input_feature_dim = kwargs.get('input_feature_dim')
    num_patches = kwargs.get('num_patches')
    features_size = kwargs.get('features')
    features = (features_size, features_size, features_size)
    checkpoint_path = kwargs.get('checkpoint_path')

    env = ManiSkill(reward_mode='normalized_dense', pose_reward_coef=0.01, place_reward_coef=0.1, image_size=obs_shape[2])

    env.seed(seed)

    env = TimeLimit(env, max_episode_steps=max_episode_steps)

    env = FailOnTimelimitWrapper(env)

    env = DMCWrapper(env, obs_to_string=obs_to_string, clip_reward=clip_reward)

    dinosaur = Dinosaur(dino_model_name=model_name, n_slots=num_slots, slot_dim=slot_dim,
                        intput_feature_dim=input_feature_dim, num_patches=num_patches, features=features)

    state_dict = torch.load(checkpoint_path)['state_dict']
    state_dict = {key[len('models.'):]: value for key, value in state_dict.items()}

    dinosaur.load_state_dict(state_dict)
    dinosaur = dinosaur.eval()
    dinosaur.requires_grad_(False)

    slot_extractor = SlotExtractor(model=dinosaur, device='cuda', name_model = 'DINOSAUR')

    env = SlotExtractorWrapper(env, slot_extractor, num_slots, slot_dim)
    return env