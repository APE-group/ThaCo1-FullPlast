#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Tue Feb 25 10:21:52 2020

@author: dav
"""

import os
import utils
from utils import FlattenList2List
import numpy as np
from time import time_ns as time

class Test:

    def __init__(self, params_sim, params_input, params_path, params_times):

        ############## SCALAR ##############
        self.name = 'test'
        self.n_class = params_input['n_class']
        self.n_ranks = params_input['n_ranks_test']
        self.t_img = params_times['t_img']
        self.t_pause = params_times['t_pause']
        self.stage_id = params_sim['stage_id']
        self.current_learning_cycle = 1
        self.seed = params_sim[f'mnist_{self.name}_seed']
        self.dataset_selection = params_input['dataset_selection']["test"]['type']
        self.dataset_balanced = params_input['dataset_selection']["test"]['balanced']
        self.dataset_shuffle = params_input['dataset_selection']["test"]['shuffle']
        self.index = params_input['dataset_index']["test"]
        ############## PATH ##############
        self.train_dir = params_path['train_dir']
        self.input_dir = params_path['input_dir']
        self.fn_feat = os.path.join(self.input_dir, 'test_features.npy')
        self.fn_labels = os.path.join(self.input_dir, 'test_labels.npy')

        if self.dataset_selection == 'random':
            self.n_img = self.n_ranks * self.n_class

        elif self.dataset_selection == 'deterministic':
            self.n_img = len(self.index)

        ############## ARRAY ##############
        self.all_features, self.all_labels = utils.LoadFeatures(self)
        self.pattern_labels, self.pattern_features = self.TrialDataset(self.all_features, self.all_labels)

    def TrialIndices(self):

        n_labels = len(self.all_labels)
        mnist_absolute_indices = np.arange(0, n_labels, 1, dtype=int)
        mnist_class_indices = [mnist_absolute_indices[(self.all_labels == nclass)] for nclass in range(self.n_class)]

        if self.dataset_shuffle:
            index_shuffling = self.rand.permutation(np.arange(0, self.n_img, 1, dtype=int))
        else:
            index_shuffling = np.arange(0, self.n_img, 1, dtype=int)


        if self.dataset_selection == 'random':

            if self.dataset_balanced == True:
                index = np.array([self.rand.choice(mnist_class_indices[nclass], size=self.n_ranks)
                                   for nclass in range(self.n_class)]).flatten()
            else:
                index = self.rand.choice(mnist_absolute_indices, size=self.n_img)

        elif self.dataset_selection == 'deterministic':
            index = np.hstack(self.index)

        trial_index = {'rng_seed': self.seed, 'index mnist': index, 'index shuffling': index_shuffling}

        return index, index_shuffling, trial_index

    def TrialDataset(self, all_features, all_labels):

        print('\nTest features and labels selection\n')

        if self.seed == '': self.seed = time() % 2 ** 32

        if self.stage_id == '00_awake_training':
            self.rand = np.random.RandomState(self.seed)
            index, index_shuffling, trial_index = self.TrialIndices()
        else:
            trial_dict_path = os.path.join(self.train_dir, 'trial_dict.npy')
            try:
                trial_index = np.load(trial_dict_path, allow_pickle=True, encoding='latin1').item()['test']
                index = trial_index['index mnist']
                index_shuffling = trial_index['index shuffling']
                self.seed = trial_index['rng_seed']
                self.rand = np.random.RandomState(self.seed)
            except BaseException as err:
                raise err

        # Trial training set lists
        trial_features, trial_labels = [], []

        for i in index:
            trial_features.append(all_features[i])
            trial_labels.append(all_labels[i])

        trial_features = np.asarray(trial_features, dtype=object)[index_shuffling]
        trial_labels = np.asarray(trial_labels)[index_shuffling]

        self.trial_index, self.index = trial_index, index
        self.trial_index['labels'] = trial_labels

        return trial_labels, trial_features