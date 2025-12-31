#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  Parent_Train.py
#
#  Copyright © 2020-2026  Leonardo Tonielli        <leonardo.tonielli@roma1.infn.it>
#  Copyright © 2020-2026  Pier Stanislao Paolucci  <pier.paolucci@roma1.infn.it>
#  Copyright © 2020-2026  Elena Pastorelli         <elena.pastorelli@roma1.infn.it>
#  Copyright © 2023-2026  Cosimo Lupo              <cosimo.lupo89@gmail.com>
#
#  Note: Please keep the list of original authors and feel free to
#  add your name if you make substantial contributions, in accordance
#  with the GPL-3 license.
#
#  SPDX-License-Identifier: GPL-3.0-only
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program. If not, see <https://www.gnu.org/licenses/>.
#

import os
import utils
from utils import FlattenList2List
import numpy as np
from time import time_ns as time

class Train:

    def __init__(self, params_sim, params_input, params_path, params_times):

        ############## SCALAR ##############
        self.name = 'training'
        self.n_class = params_input['n_class']
        self.n_ranks = params_input['n_ranks_train']
        self.t_img = params_times['t_img']
        self.t_pause = params_times['t_pause']
        self.n_fov = params_input['n_fov']
        self.stage_id = params_sim['stage_id']
        self.current_learning_cycle = params_sim['current_learning_cycle']
        self.n_learning_cycles = params_sim['n_learning_cycles']
        self.seed = params_sim[f'mnist_{self.name}_seed']
        self.dataset_selection = params_input['dataset_selection']["training"]['type']
        self.dataset_balanced = params_input['dataset_selection']["training"]['balanced']
        self.dataset_shuffle = params_input['dataset_selection']["training"]['shuffle']
        self.index = params_input['dataset_index']["training"]
        ############## PATH ##############
        self.train_dir = params_path['train_dir']
        self.input_dir = params_path['input_dir']
        self.fn_feat = os.path.join(self.input_dir, 'training_features.npy')
        self.fn_labels = os.path.join(self.input_dir, 'training_labels.npy')
        self.n_img = self.n_ranks * self.n_class
        self.n_img_stage = self.n_img * self.current_learning_cycle
        self.n_img_trial = self.n_img * self.n_learning_cycles

        ############## ARRAY ##############
        self.all_features, self.all_labels = utils.LoadFeatures(self)
        self.pattern_labels, self.pattern_features = self.TrialDataset(self.all_features, self.all_labels)

    def TrialIndices(self):

        n_labels = len(self.all_labels)
        mnist_absolute_indices = np.arange(0, n_labels, 1, dtype=int)
        mnist_class_indices = [mnist_absolute_indices[(self.all_labels == nclass)] for nclass in range(self.n_class)]

        if self.dataset_shuffle:
            index_shuffling = np.array(
                [self.rand.permutation(np.arange(ncycle * self.n_img, (ncycle + 1) * self.n_img, 1, dtype=int))
                 for ncycle in range(self.n_learning_cycles)]).flatten()
        else:
            index_shuffling = np.array([np.arange(ncycle * self.n_img, (ncycle + 1) * self.n_img, 1, dtype=int)
                 for ncycle in range(self.n_learning_cycles)]).flatten()

        if self.dataset_selection == 'random':

            if self.dataset_balanced == True:
                index = np.array([[
                    self.rand.choice(mnist_class_indices[nclass], size=self.n_ranks) for nclass in range(self.n_class)]
                    for ncycle in range(self.n_learning_cycles)]).flatten()
            else:
                index = np.array([[self.rand.choice(mnist_class_indices[nclass], size=1)
                    for nclass in np.sort(self.rand.randint(0, self.n_class, self.n_img))] for ncycle in range(self.n_learning_cycles)]).flatten()

        elif self.dataset_selection == 'deterministic':
            index = np.array(self.index).flatten()

        trial_index = {'rng_seed': self.seed, 'index mnist': index, 'index shuffling': index_shuffling}

        return index, index_shuffling, trial_index


    def TrialDataset(self, all_features, all_labels):

        print('\nTraining features and labels selection\n')

        if self.seed == '': self.seed = time() % 2**32

        if self.stage_id == '00_awake_training':
            self.rand = np.random.RandomState(self.seed)
            index, index_shuffling, trial_index = self.TrialIndices()
        else:
            trial_dict_path = os.path.join(self.train_dir, 'trial_dict.npy')
            try:
                trial_index = np.load(trial_dict_path, allow_pickle=True, encoding='latin1').item()['training']
                index = trial_index['index mnist']
                index_shuffling = trial_index['index shuffling']
                self.seed = trial_index['rng_seed']
                self.rand = np.random.RandomState(self.seed)
            except BaseException as err:
                raise err

        #Trial training set lists
        trial_features, trial_labels = [], []

        for i in index:
            trial_features.append(all_features[i])
            trial_labels.append(all_labels[i])

        trial_features = np.asarray(trial_features, dtype=object)[index_shuffling]
        trial_labels = np.asarray(trial_labels)[index_shuffling]

        self.trial_index, self.index, self.index_shuffling = trial_index, index, index_shuffling
        self.trial_index['labels'] = trial_labels

        return trial_labels, trial_features
