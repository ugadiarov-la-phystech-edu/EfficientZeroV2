# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import random
import time
import copy
import math
from ez.agents.base import Agent
from omegaconf import open_dict

from ez.envs import make_shapes2d
from ez.utils.format import DiscreteSupport
from ez.agents.models import EfficientZero
from ez.agents.models.base_model import *


class OCEZShapes2dAgent(Agent):
    def __init__(self, config, centric=None):
        super().__init__(config)

        self.update_config()

        self.state_norm = config.model.state_norm
        self.value_prefix = config.model.value_prefix

        self.slate_config = self.config.oc.ocr_config_path
        self.slate_weights = self.config.oc.checkpoint_path
        self.slot_dim = self.config.oc.slot_dim
        self.n_slots = self.config.oc.n_slots
        self.latent_dim = self.config.oc.latent_dim

    def update_config(self):
        assert not self._update

        env = make_shapes2d(self.config.env.game, seed=0, save_path=None, **self.config.env)
        action_space_size = env.action_space.n

        obs_channel = 1 if self.config.env.gray_scale else 3

        reward_support = DiscreteSupport(self.config)
        reward_size = reward_support.size

        value_support = DiscreteSupport(self.config)
        value_size = value_support.size

        localtime = time.strftime('%Y-%m-%d %H:%M:%S')
        tag = '{}-seed={}-{}/'.format(self.config.tag, self.config.env.base_seed, localtime)

        with open_dict(self.config):
            self.config.env.action_space_size = action_space_size
            self.config.mcts.num_top_actions = min(action_space_size, self.config.mcts.num_top_actions)
            self.config.env.obs_shape[0] = obs_channel
            self.config.rl.discount **= self.config.env.n_skip
            self.config.model.reward_support.size = reward_size
            self.config.model.value_support.size = value_size

            if action_space_size < 4:
                self.config.mcts.num_top_actions = 2
                self.config.mcts.num_simulations = 4
            elif action_space_size < 16:
                self.config.mcts.num_top_actions = 4
            else:
                self.config.mcts.num_top_actions = 8

            if not self.config.mcts.use_gumbel:
                self.config.mcts.num_simulations = 50
            print(f'env={self.config.env.env}, game={self.config.env.game}, |A|={action_space_size}, '
                  f'top_m={self.config.mcts.num_top_actions}, N={self.config.mcts.num_simulations}')
            self.config.save_path += tag

        self.obs_shape = copy.deepcopy(self.config.env.obs_shape)
        self.action_space_size = self.config.env.action_space_size

        self._update = True

    def build_model(self):
        representation_model = OCRepresentationNetwork(self.slate_config, self.obs_shape[2], self.slate_weights)

        dynamics_model = OCDynamicsNetwork(self.slot_dim, self.latent_dim, self.action_space_size, self.n_slots)

        value_policy_model = OCValuePolicyNetwork(self.slot_dim, self.latent_dim, self.n_slots,
                                                  self.config.model.value_support.size, self.action_space_size)

        reward_output_size = self.config.model.reward_support.size
        if self.value_prefix:
            reward_prediction_model = OCSupportLSTMNetwork(self.slot_dim, self.latent_dim, self.n_slots,
                                                           reward_output_size, self.config.model.lstm_hidden_size)
        else:
            reward_prediction_model = OCSupportNetwork(self.slot_dim, self.latent_dim, self.n_slots, reward_output_size)

        projection_layers = self.config.model.projection_layers
        head_layers = self.config.model.prjection_head_layers
        assert projection_layers[1] == head_layers[1]

        projection_model = OCProjectionNetwork(self.slot_dim, self.latent_dim, self.n_slots, projection_layers[0], projection_layers[1])
        projection_head_model = ProjectionHeadNetwork(projection_layers[1], head_layers[0], head_layers[1])

        ez_model = EfficientZero(representation_model, dynamics_model, reward_prediction_model, value_policy_model,
                                 projection_model, projection_head_model, self.config,
                                 state_norm=self.state_norm, value_prefix=self.value_prefix)

        return ez_model