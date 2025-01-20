# Copyright (c) EVAR Lab, IIIS, Tsinghua University.
#
# This source code is licensed under the GNU License, Version 3.0
# found in the LICENSE file in the root directory of this source tree.

import copy
import os
import time
import ray
import numpy as np
import pickle

@ray.remote
class GlobalStorage:
    def __init__(self, self_play_model, reanalyze_model, latest_model):
        self.models = {
            'self_play': self_play_model,
            'reanalyze': reanalyze_model,
            'latest': latest_model
        }
        self.log_scalar = {}
        self.eval_log_scalar = {}
        self.log_distribution = {}
        self.counter = 0
        self.eval_counter = 0
        self.best_score = - np.inf
        self.start = False
        # self.batch = None

    def get_weights(self, model_name):
        assert model_name in self.models.keys()
        return self.models[model_name].get_weights()

    def set_weights(self, weights, model_name):
        assert model_name in self.models.keys()
        # print('[Update] set recent model of {}'.format(model_name))
        return self.models[model_name].set_weights(weights)

    def increase_counter(self):
        self.counter += 1

    def get_counter(self):
        return self.counter

    def set_eval_counter(self, counter):
        self.eval_counter = counter

    def get_eval_counter(self):
        return self.eval_counter

    def set_start_signal(self):
        self.start = True

    def get_start_signal(self):
        return self.start

    def set_best_score(self, score):
        self.best_score = max(self.best_score, score)

    def get_best_score(self):
        return self.best_score

    def add_log_scalar(self, dic):
        for key, val in dic.items():
            if key not in self.log_scalar.keys():
                self.log_scalar[key] = []

            self.log_scalar[key].append(val)

    def add_eval_log_scalar(self, dic):
        for key, val in dic.items():
            if key not in self.eval_log_scalar.keys():
                self.eval_log_scalar[key] = []
            self.eval_log_scalar[key].append(val)

    def add_log_distribution(self, dic):
        for key, val in dic.items():
            if key not in self.log_distribution.keys():
                self.log_distribution[key] = []

            self.log_distribution[key] += val.tolist()

    def get_log(self):
        # for scalar
        scalar = {}
        for key, val in self.log_scalar.items():
            scalar[key] = np.mean(val)

        eval_scalar = {}
        for key, val in self.eval_log_scalar.items():
            eval_scalar[key] = np.mean(val)

        # for distribution
        distribution = {}
        for key, val in self.log_distribution.items():
            distribution[key] = np.array(val).flatten()

        self.log_scalar = {}
        self.eval_log_scalar = {}
        self.log_distribution = {}
        return eval_scalar, scalar, distribution


    def save_storage(self, path):
        f_models = open(os.path.join(path, 'models.b'), 'wb')
        pickle.dump(self.models, f_models)
        f_models.close()

        attributes = {'log_scalar': self.log_scalar,
                      'eval_log_scalar': self.eval_log_scalar, 'log_distribution': self.log_distribution, 'counter': self.counter,
                      'eval_counter': self.eval_counter, 'best_score': self.best_score}
        f_attributes = open(os.path.join(path, 'glob_storage_attributes.b'), 'wb')
        pickle.dump(attributes, f_attributes)
        f_attributes.close()

        return True


    def load_storage(self, path):
        f_models = open(os.path.join(path, 'models.b'), 'rb')
        self.models = pickle.load(f_models)
        f_models.close()

        f_attributes = open(os.path.join(path, 'glob_storage_attributes.b'), 'rb')
        attributes = pickle.load(f_attributes)
        f_attributes.close()

        self.log_scalar = attributes['log_scalar']
        self.eval_log_scalar = attributes['eval_log_scalar']
        self.log_distribution = attributes['log_distribution']
        self.counter = attributes['counter']
        self.eval_counter = attributes['eval_counter']
        self.best_score = attributes['best_score']
        return True

# ======================================================================================================================
# global storage server
# ======================================================================================================================