# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import time
import copy
import math
from ez.agents.base import Agent
from omegaconf import open_dict
from ez.envs.robosuite import RobosuiteEnv

from ez.utils.format import DiscreteSupport
from ez.agents.models import EfficientZero
from ez.agents.models.base_model import *


class OCEZRobosuiteAgent(Agent):
    def __init__(self, config):
        super().__init__(config)

        self.update_config()

        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix

        self.dinosaur_weights = self.config.oc.checkpoint_path
        self.slot_dim = self.config.oc.slot_dim
        self.n_slots = self.config.oc.n_slots
        self.model_name = self.config.oc.model_name
        self.input_feature_dim = self.config.oc.input_feature_dim
        self.num_patches = self.config.oc.num_patches
        self.features = (self.config.oc.features, self.config.oc.features, self.config.oc.features)
        self.latent_dim = self.config.oc.latent_dim
        self.init_zero = config.model.init_zero
        self.fc_layers = config.model.fc_layers

    def update_config(self):
        assert not self._update

        env = RobosuiteEnv(task=self.config.env.game, horizon=self.config.env.max_episode_steps, seed=0)
        action_space_size = env.action_space.shape[0]

        obs_channel = 1 if self.config.env.gray_scale else 3

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size
        self.reward_support = reward_support

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size
        self.value_support = value_support

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.env.obs_shape[0] = obs_channel
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size

            self.config.save_path += tag

        self.obs_shape = copy.deepcopy(self.config.env.obs_shape)
        self.action_space_size = self.config.env.action_space_size

        self._update = True

    def build_model(self):
        is_continuous = (self.config.env.env == "robosuite")

        dynamics_model = OCDynamicsNetwork(self.slot_dim, self.latent_dim, self.action_space_size, self.n_slots)

        value_policy_model = OCValuePolicyNetwork(self.slot_dim, self.latent_dim, self.n_slots,
                                                  self.config.model.value_support.size,
                                                  self.action_space_size * 2,
                                                  self.fc_layers, self.init_zero, is_continuous,
                                                  v_num=self.config.train.v_num)

        reward_output_size = self.config.model.reward_support.size
        if self.value_prefix:
            reward_prediction_model = OCSupportGRUGNN(self.slot_dim, self.latent_dim, self.n_slots,
                                                      reward_output_size, self.config.model.rnn_hidden_size,
                                                      self.fc_layers, self.init_zero)
        else:
            reward_prediction_model = OCSupportNetwork(self.slot_dim, self.latent_dim, self.n_slots, reward_output_size,
                                                       self.fc_layers, self.init_zero)


        ez_model = EfficientZero(dynamics_model, reward_prediction_model, value_policy_model, self.config,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix)

        return ez_model