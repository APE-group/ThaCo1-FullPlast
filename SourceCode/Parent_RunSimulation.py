#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  Parent_RunSimulation.py
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

import os, yaml
from itertools import product
from SourceCode.utils import RandomString

class ThaCo:

    def __init__(self, args):

        config_complete, config_tune = self.Config(args.model_dir, args.tune_name)

        # Directories
        paths = config_complete['paths']
        if config_tune:
            if args.code_dir:
                config_tune['paths']['code_dir'] = os.path.abspath(os.path.expanduser(args.code_dir))
            if args.input_dir:
                config_tune['paths']['input_dir'] = os.path.abspath(os.path.expanduser(args.input_dir))
            if args.output_dir:
                config_tune['paths']['output_dir'] = os.path.abspath(os.path.expanduser(args.output_dir))
            if args.simulation_name:
                config_tune['parameters']['simulation']['simulation_name'] = args.simulation_name
            if args.training_set_indices:
                config_tune['parameters']['input']['dataset_selection']['training']['type'] = 'deterministic'
                config_tune['parameters']['input']['dataset_index']['training'] = [int(index) for index in args.training_set_indices.split(' ')]
            if args.test_set_indices:
                config_tune['parameters']['input']['dataset_selection']['test']['type'] = 'deterministic'
                config_tune['parameters']['input']['dataset_index']['test'] = [int(index) for index in args.test_set_indices.split(' ')]
            if isinstance(args.nest_seed, int):
                config_tune['parameters']['simulation']['nest_seed'] = args.nest_seed
            if isinstance(args.np_seed, int):
                config_tune['parameters']['simulation']['numpy_seed'] = args.np_seed

            paths.update(config_tune['paths'])
        self.code_dir = paths['code_dir']
        self.input_dir = paths['input_dir']
        self.output_dir = paths['output_dir']

        wrong_paths = []
        # TO-BE-DONE: Do we want to include output_dir in this check? Or rather let the code try to create it afterwards?
        for key,path in [('code_dir',self.code_dir), ('input_dir',self.input_dir), ('output_dir',self.output_dir)]:
            if not os.path.exists(path):
                wrong_paths.append((key,path))
        if wrong_paths:
            if len(wrong_paths)==1:
                raise FileNotFoundError('The following path does not seem to exist:' + "".join(['\n - ' + key + ': "' + path  + '"' for key,path in wrong_paths]) + '\nAre you sure you set it properly?')
            elif len(wrong_paths)>1:
                raise FileNotFoundError('The following paths do not seem to exist:' + "".join(['\n - ' + key + ': "' + path  + '"' for key,path in wrong_paths]) + '\nAre you sure you set them properly?')

        # New parameters

        #print('\n\n0', config_complete['parameters'])


        #self.default_network = config_complete['parameters']['default_network']

        #print('\n\n1', self.flatten_config_complete)

        self.config_complete = config_complete['parameters'].copy()
        if config_tune:
            self.config_tune = config_tune['parameters'].copy()
            self.config_tune_flatten = self.FlattenDict2Dict(config_tune['parameters'])
        self.config_complete['simulation']['n_threads'] = args.n_threads

        #print('\n\n3', self.flatten_config_complete)


    def Config(self, model_dir, tune_name):

        model_dir = os.path.abspath(model_dir)

        config_complete = os.path.join(model_dir, 'complete.yaml')
        try:
            with open(config_complete, 'r') as f:
                config_complete = yaml.safe_load(f)
        except FileNotFoundError as err:
            raise err
        for key in config_complete['paths'].keys():
            config_complete['paths'][key] = os.path.abspath(os.path.expanduser(config_complete['paths'][key]))

        if tune_name != '': tune_name = f'_{tune_name}'
        config_tune = os.path.join(model_dir, f'tune{tune_name}.yaml')
        try:
            with open(config_tune, 'r') as f:
                config_tune = yaml.safe_load(f)
            for key in config_tune['paths'].keys():
                config_tune['paths'][key] = os.path.abspath(os.path.expanduser(config_tune['paths'][key]))
        except FileNotFoundError as err:
            config_tune = None
            print(f'No `tune{tune_name}.yaml` file found. Model parameters are read only from `complete.yaml` file and from `thaco.py` CLI args.')

        return config_complete, config_tune

    def FlattenDict2Dict(self, dic, parent_key=''):

        items = []
        sep = '.'

        for k, v in dic.items():
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(self.FlattenDict2Dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def ConfigsDict(self, config_tune_flatten):

        ds_selection_train = config_tune_flatten['input.dataset_selection.training.type']
        ds_selection_test = config_tune_flatten['input.dataset_selection.test.type']

        if ds_selection_train == 'deterministic':
            train_index = config_tune_flatten['input.dataset_index.training']
        if ds_selection_test == 'deterministic':
            test_index = config_tune_flatten['input.dataset_index.test']

        config_tune_flatten['input.dataset_index.training'] = -1
        config_tune_flatten['input.dataset_index.test'] = -1

        tuned_parms_comb = list(
            product(*([v] if not isinstance(v, (list, tuple)) else v for v in config_tune_flatten.values())))
        configs_dict = {}

        for n, comb in enumerate(tuned_parms_comb):
            config_dict = dict(zip(config_tune_flatten.keys(), comb))
            configs_dict['config%d' % (n)] = config_dict

        stages, n_trials = config_tune_flatten['simulation.stages'], config_tune_flatten['simulation.n_trials']
        n_configs =  len(configs_dict.keys()) // len(stages)
        cycles_dict = {trial:{conf: {} for conf in range(n_configs)} for trial in range(n_trials)}

        for n_trial in range(n_trials):
            for n_config, config in enumerate(configs_dict.values()):
                id_config = n_config % n_configs
                id_stage = n_config // n_configs
                config_stage = f"{id_stage:02}_{config['simulation.stages']}"
                if ds_selection_train == 'deterministic':
                    config['input.dataset_index.training'] = train_index
                if ds_selection_test == 'deterministic':
                    config['input.dataset_index.test'] = test_index
                cycles_dict[n_trial][id_config][config_stage] = config

        return cycles_dict

    def Run(self, config_complete, config_tune):

        save_syn = config_tune['paths']['save_syn']
        save_out = config_tune['paths']['save_out']
        load_syn = config_tune['paths']['load_syn']

        if not os.path.exists(save_syn):
            try:
                os.makedirs(save_syn)
            except BaseException as err:
                raise err

        if not os.path.exists(save_out):
            try:
                os.makedirs(save_out)
            except BaseException as err:
                raise err

        try:
            with open(os.path.join(save_syn, 'config_complete.yaml'), 'w') as config_file:
                yaml.dump(config_complete, config_file)
        except BaseException as err:
            raise err

        try:
            with open(os.path.join(save_syn, 'config_tune.yaml'), 'w') as config_file:
                yaml.dump(config_tune, config_file)
        except BaseException as err:
            raise err

        try:
            with open(os.path.join(save_out, 'config_complete.yaml'), 'w') as config_file:
                yaml.dump(config_complete, config_file)
        except BaseException as err:
            raise err

        try:
            with open(os.path.join(save_out, 'config_tune.yaml'), 'w') as config_file:
                yaml.dump(config_tune, config_file)
        except BaseException as err:
            raise err

        exec_path = os.path.join(self.code_dir, 'net_sim.py')
        config_complete_path = os.path.join(save_out, 'config_complete.yaml')
        config_tune_path = os.path.join(save_out, 'config_tune.yaml')

        try:
            res = os.system('python3 {} --config_complete_path {} --config_tune_path {}'.format(exec_path, config_complete_path, config_tune_path))
            if res != 0:
                raise RuntimeError('ERROR! Call to net_sim.py through os.system() failed.')
        except BaseException as err:
            raise err

    def SavePath(self):

        n_class = self.config_complete['input']['n_class']
        n_ranks_train = self.config_complete['input']['n_ranks_train']
        sim_name = self.config_tune['simulation']['simulation_name'] if self.config_tune['simulation']['simulation_name']!='' else 'default'

        root_output_dir = os.path.join(self.output_dir, 'SimulationOutput', 'train_classes_%d' % n_class, 'train_example_%d' % n_ranks_train)

        if self.config_tune['simulation']['trial_id'] == '':
            trial_id = RandomString(lenght=10)
        else:
            trial_id = self.config_tune['simulation']['trial_id']

        trial_syn_dir = os.path.join(root_output_dir, 'Synapses', '%s' % sim_name, '%s' % trial_id)
        trial_main_output_dir = os.path.join(root_output_dir, 'MainOutput', '%s' % sim_name, '%s' % trial_id)

        return trial_syn_dir, trial_main_output_dir, trial_id

    def LoadPath(self, config, trial_id, stage_id, load_stage_id, trial_syn_dir, trial_main_output_dir, current_learning_cycle,n_learning_cycles):

        config['simulation.trial_id'] = trial_id
        config['simulation.stage_id'] = stage_id
        config['simulation.stage'] = stage_id[3:]
        config['simulation.load_stage_id'] = load_stage_id
        config['simulation.load_stage'] = load_stage_id[3:] if load_stage_id != '' else ''
        config['simulation.current_learning_cycle'] = current_learning_cycle
        config['simulation.n_learning_cycles'] = n_learning_cycles


        config['paths.input_dir'] = self.input_dir
        config['paths.save_syn'] = os.path.join(trial_syn_dir, stage_id)
        config['paths.save_out'] = os.path.join(trial_main_output_dir, stage_id)
        config['paths.train_dir'] = os.path.join(trial_main_output_dir, '00_awake_training')
        if load_stage_id != '':
            config['paths.load_syn'] = os.path.join(trial_syn_dir, load_stage_id)
        else:
            config['paths.load_syn'] = ''

        return config
