#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  Parent_Network.py
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

import numpy as np
import nest
from Parent_Connections import SynapsesParameters
from Parent_Connections import ConnectFW,ConnectBW,ConnectInter
from Parent_Noise import Param_Noise as NoiseParameters
from Parent_Layer import Layer
from Parent_Train import Train
from Parent_Test import Test
from utils import MergeDicts, RemoveValueFromDict, KeyExists
import os, sys
from time import time_ns


class Net:

    def __init__(self, config_complete, config_tune):

        self.config_complete = config_complete
        self.config_tune = config_tune

        #SET BRAIN STATE
        self.DefaultNetParams()
        #SET KERNEL STATUS
        self.SetKernelStatus()

        #PARAMS
        simulation, input, paths = self.params_default['simulation'], self.params_default['input'], self.params_default['paths']

        #INIT VARIABLES
        self.State()

        #TRAINING & TEST THALAMUS PATTERN
        self.trial_dict = {}
        self.TrainPattern()
        self.TestPattern()
        self.trial_dict['nest_seed'] = simulation['nest_seed']
        self.trial_dict['numpy_seed'] = simulation['numpy_seed']
        if self.stage_id == '00_awake_training':
            trial_dict_path = os.path.join(self.training_path, 'trial_dict.npy')
            label_arr_path = os.path.join(self.training_path, 'train_labels_arr.npy')
            try:
                np.save(label_arr_path, self.train.pattern_labels)
            except BaseException as err:
                raise err
            try:
                np.save(trial_dict_path, self.trial_dict)
            except BaseException as err:
                raise err

        #LAYERS
        self.Thalamus()
        self.Cortex()
        self.Readout()

        #INTER-AREA & INTER-LAYER CONNECTIONS
        self.Connections()

    def DefaultNetParams(self):

        # CONFIG COMPLETE PARAMETERS
        #self.config_complete['default_network']['synapses'] = RemoveValueFromDict(self.config_complete['default_network']['synapses'], None)

        params_network = self.config_complete['default_network']
        params_simulation, params_input, params_paths = (self.config_complete['simulation'], self.config_complete['input'],
                                                         {})

        # CONFIG TUNE PARAMETERS
        tune_network, tune_simulation, tune_input, tune_paths = {}, {}, {}, {}
        if 'default_network' in self.config_tune.keys(): tune_network = self.config_tune['default_network']
        if 'simulation' in self.config_tune.keys(): tune_simulation = self.config_tune['simulation']
        if 'input' in self.config_tune.keys(): tune_input = self.config_tune['input']
        if 'paths' in self.config_tune.keys(): tune_paths = self.config_tune['paths']

        # DEFAULT PARAMETERS TUNING
        params_default_network = MergeDicts(params_network, tune_network)
        params_default_simulation = MergeDicts(params_simulation, tune_simulation)
        params_default_input = MergeDicts(params_input, tune_input)
        params_default_paths = MergeDicts(params_paths, tune_paths)

        self.params_default = {
        'paths': params_default_paths,
        'simulation': params_default_simulation,
        'input': params_default_input,
        'network': params_default_network,
        }

        self.config_complete = MergeDicts(self.config_complete, self.config_tune)
        self.params_stage = self.config_complete['default_network'].copy()

        #for stage in self.config_complete['simulation']['stages']:
        #    if 'synapses' in self.config_complete[stage].keys():
        #        self.config_complete[stage]['synapses'] = RemoveValueFromDict(self.config_complete[stage]['synapses'], None)

    def State(self):

        params_sim, params_input, params_paths = self.params_default['simulation'], self.params_default['input'], self.params_default['paths']
        params_times_train, params_times_test = self.config_complete['awake_training']['times'], self.params_default['network']['times']
        params_struct = self.params_default['network']['structure']

        # STAGE
        self.trial_id = params_sim['trial_id']
        self.stage = params_sim['stage']
        self.stage_id = params_sim['stage_id']
        self.load_stage = params_sim['load_stage']
        self.load_stage_id = params_sim['load_stage_id']
        self.save_all_output = params_sim['save_all_output']

        self.start_stage = self.load_stage_id
        #self.stage = self.stage_id
        self.classification = params_sim['test']

        # PATHS
        self.save_path = params_paths['save_out']
        self.save_state = params_paths['save_syn']
        self.load_path = params_paths['load_syn']
        self.training_path = params_paths['train_dir'] #os.path.join(self.save_path[:-(len(self.stage_id) + 1)], '00_awake_training')

        #TIMES
        self.t_simulation = 0.1
        self.current_learning_cycle = params_sim['current_learning_cycle']
        self.n_learning_cycles = params_sim['n_learning_cycles']
        self.lesion = params_struct['cx']['lesion']

        if self.classification == 'classification' or self.classification == 'retrieval':
            self.stage_test = self.stage_id.split('_')[0] + '_awake_test'

        if self.classification == 'only':
            sys.stdout = open(os.path.join(self.save_path, 'run_%s_classification.log' % self.stage), 'w')
        else:
            sys.stdout = open(os.path.join(self.save_path, 'run_%s.log' % self.stage), 'w')


        print(f"Areas: {params_struct['cx']['areas']}")
        print(f"Neurons per group: {params_struct['cx']['n_exc_ca']}")
        print(f"Coding number: {params_input['coding_number']}")
        print(f"FOVs number {params_input['n_fov']}")
        print(f"Scale image input {64} x {64}")
        print(f"N. of classes: {params_input['n_class']}")
        print(f"N. of examples per class: {params_input['n_ranks_train']}")
        print(f"Stage: {self.stage}, classification: {self.classification}")
        print(f'Main Output path: {self.save_path}')
        print(f'Synapses Output path: {self.load_path}')
        print(f'Synapses loading path: {self.save_state}')
        print(f'Save all output {self.save_all_output}')
    def SetNetState(self, params_start_state, params_new_state):

        start_state_name, new_state_name = params_start_state['name'], params_new_state['name']
        print(f'\nChange State\n{start_state_name} --> {new_state_name}')

        if KeyExists(params_new_state, 'neuron'):
            params_neuron = params_new_state['neuron']
        else:
            params_neuron = {}

        if KeyExists(params_new_state, 'noise'):
            params_noise = params_new_state['noise']
        else:
            params_noise = {}

        if KeyExists(params_new_state, 'synapses'):
            params_syn = params_new_state['synapses'].copy()
            params_syn = RemoveValueFromDict(params_syn, None)
        else:
            params_syn = {}

        conn_noise_dict, pop_noise_dict = self.conn_dict_layers['noise']['all'], self.pop_dict['noise']
        conn_syn_dict = self.conn_dict_layers

        #SET NEURON PARAMETER
        if params_neuron != {}:
            for layer in params_neuron.keys():
                for neur_type in params_neuron[layer]:
                    pop, params_set = self.pop_dict[layer][neur_type], params_neuron[layer][neur_type]
                    pop.set(params_set)
                    print(f'\nSetting {layer} neuron parameters:', params_set)
        else:
            print('\nNo NEURON parameters to set')

        #SET NOISE PARAMETER
        if params_noise != {}:
            print('\nSetting noise parameters:', params_noise)
            for layer_name, pop_names in params_noise['weights'].items():
                for pop_name in pop_names:
                    print(f'\n {layer_name} - {pop_name}')
                    if params_noise['rates'][layer_name][pop_name] != None:
                        conn = conn_noise_dict[layer_name][pop_name]['conn']
                        rate, weight = params_noise['rates'][layer_name][pop_name], params_noise['weights'][layer_name][pop_name]
                        conn.set({'weight': weight})
                        print(f'rate: {rate}\nweight: {weight}')
                        if layer_name == 'th':
                            self.LayerTh.p_noise_exc.rate = rate
                        elif layer_name == 'cx' and pop_name=='inh':
                            self.Layer1.p_noise_inh.rate = rate
        else:
            print('\nNo NOISE parameters to set')

        #SET SYNAPTIC PARAMETERS
        if params_syn != {}:
            print('\nSetting SYNAPTIC parameters:', params_syn)
            if 'conn_type' in params_syn.keys():
                conn_type = params_syn['conn_type']
                if 'stdp_params' in conn_type: stdp_params = conn_type['stdp_params']
            else:
                stdp_params = {}

            if 'conn_params' in params_syn.keys(): syn_params = params_syn['conn_params']
            for layer, conn_spec1 in syn_params.items():
                for conn_spec2, conn_params in conn_spec1.items():
                    for conn_name, param_set in conn_params.items():

                        areas = self.params_default['network']['structure'][layer_name]['areas']

                        if param_set != {}:


                            if conn_spec2 == 'all':
                                conn_spec_stdp = 'intra'
                            else:
                                conn_spec_stdp = conn_spec2

                            stdp =  self.params_default['network']['synapses']['conn_params'][layer][conn_spec_stdp][conn_name]['stdp']
                            if 'modulation' in param_set:
                                modulation = param_set.pop('modulation')
                            else:
                                modulation = None

                            for area in range(areas):
                                if modulation != None:
                                    if 'ratio' in modulation:
                                        if conn_spec2 == 'all':
                                            start_syn = conn_syn_dict[layer][conn_spec2][conn_name]['conn']
                                        else:
                                            start_syn = conn_syn_dict[layer][conn_spec2][conn_name]['conn'][area]
                                        ratio = modulation['ratio']

                                        if type(ratio) != np.ndarray or type(ratio) != list: ratio = np.ones(areas) * ratio
                                        starting_weights = np.array(start_syn.get('weight'))
                                        new_weights = ratio[area] * starting_weights
                                        #print(f'area {area} ratio {ratio[area]}')
                                        #print('starting_weights', starting_weights[(starting_weights>1)])
                                        #print('new_weights', new_weights[(new_weights > 1)])
                                        param_set['weight'] = new_weights
                                        if stdp == True:  param_set['Wmax'] = ratio[area] *  np.array(start_syn.get('Wmax'))
                                if stdp_params != {} and stdp == True:
                                    if KeyExists(stdp_params, 'model'): stdp_params.pop('model')
                                    param_set.update(stdp_params)
                                if KeyExists(param_set, 'stdp'): param_set.pop('stdp')
                                print(f'\n{layer} - {conn_spec2} - {conn_name} - area {area}')
                                print(f'parameters: {param_set}')


                                if conn_spec2 == 'all':
                                    conn_syn_dict[layer][conn_spec2][conn_name]['conn'].set(param_set)
                                else:
                                    if  conn_name == 'inhf_exc':
                                        conn_syn_dict[layer][conn_spec2]['inh_exc']['conn'][0][area].set(param_set)
                                    elif conn_name == 'inhs_exc':
                                        conn_syn_dict[layer][conn_spec2]['inh_exc']['conn'][1][area].set(param_set)
                                    elif conn_name == 'inhf_inhf':
                                        conn_syn_dict[layer][conn_spec2]['inh_inh']['conn'][0][area].set(param_set)
                                    elif conn_name == 'inhs_inhs':
                                        conn_syn_dict[layer][conn_spec2]['inh_inh']['conn'][1][area].set(param_set)
                                    elif conn_name == 'exc_inhf':
                                        conn_syn_dict[layer][conn_spec2]['exc_inh']['conn'][0][area].set(param_set)
                                    elif conn_name == 'exc_inhs':
                                        conn_syn_dict[layer][conn_spec2]['exc_inh']['conn'][1][area].set(param_set)
                                    else:
                                        conn_syn_dict[layer][conn_spec2][conn_name]['conn'][area].set(param_set)

    def SetKernelStatus(self):

        par = self.params_default['simulation']
        nest_seed, numpy_seed, n_threads = par['nest_seed'], par['numpy_seed'], par['n_threads']

        if nest_seed == '':
            nest_seed = time_ns() % 2 ** 32
            par['nest_seed'] = nest_seed
        if numpy_seed == '':
            numpy_seed = time_ns() % 2 ** 32
            par['numpy_seed'] = numpy_seed

        nest.SetKernelStatus({"local_num_threads": n_threads})
        nest.SetKernelStatus({'rng_seed': nest_seed})
        np.random.seed(numpy_seed)

    def TrainPattern(self):

        print('\nLOADING TRAINING FILE')

        params_sim, params_input, params_paths = self.params_default['simulation'], self.params_default['input'], self.params_default['paths']
        params_times = self.config_complete['awake_training']['times']
        trial_id = params_sim['trial_id']

        self.train = Train(params_sim=params_sim, params_input=params_input, params_path=params_paths, params_times=params_times)
        self.t_training = (params_times['t_img'] + params_times['t_pause']) * params_input['n_ranks_train'] * params_input['n_class'] + params_times['t_pause']

        print(f'\nTRAINING TRIAL ID: {trial_id}\n')
        print(f'\nTRAINING LABELS: {self.train.pattern_labels}\n')
        print(f'\nTRAINING INDEX: {self.train.trial_index.items()}\n')

        self.trial_dict['training'] = self.train.trial_index


    def TestPattern(self):

        print('\nLOADING TEST FILE')

        params_sim, params_input, params_paths = self.params_default['simulation'], self.params_default['input'], self.params_default[ 'paths']
        params_times = self.params_default['network']['times']
        trial_id = params_sim['trial_id']

        self.test = Test(params_sim=params_sim, params_input=params_input, params_path=params_paths, params_times=params_times)

        if self.classification == 'retrieval':

            self.test.pattern_labels = self.train.pattern_labels[:self.train.n_img_stage]
            self.test.pattern_features = self.train.pattern_features[:self.train.n_img_stage]
            #self.test.trial_index = self.train.trial_index
            self.test.n_ranks = self.train.n_ranks
            self.test.n_img = self.train.n_img_stage
            self.test.seed = self.train.seed
            self.test.dataset_selection = self.train.dataset_selection
            self.test.index = self.train.index

        self.t_test = (params_times['t_img'] + params_times['t_pause']) * self.test.n_img
        self.trial_dict['test'] = self.test.trial_index

        print('\nTEST TRIAL: %s' %(trial_id))
        print(f'\nTEST LABELS: {self.test.pattern_labels}\n')
        if self.classification == 'retrieval':
            print(f'\nTEST INDEX: {self.train.trial_index.items()}\n')
        else:
            print(f'\nTEST INDEX: {self.test.trial_index.items()}\n')

    def Thalamus(self):

        params = self.params_default
        print()

        params_neuron, params_th, params_input, params_sim = params['network']['neuron']['th'], params['network']['structure']['th'], params['input'], params['simulation']
        noise, synapses = params['network']['noise'], params['network']['synapses']

        print('\n\nTHALAMUS CREATION')
        name = 'th'
        params_th['n_exc'] = params_input['n_fov'] * params_input['coding_number'] * 9
        params_th['n_exc_ca'] = 1
        params_th['n_pop_inh'] = 1

        #--------- DICTIONARIES

        #NEURON PARAMETERS
        neur_dict = params_neuron.copy()

        #CONNECTIONS PARAMETERS
        conns = synapses['conn_params']['th']['intra']

        #CONN RULES
        conn_rule_fwd = synapses['conn_type']['conn_rule']
        conn_rule_ee = synapses['conn_type']['conn_rule']
        conn_rule_ei = synapses['conn_type']['conn_rule']
        conn_rule_ie = synapses['conn_type']['conn_rule']
        conn_rule_ii = synapses['conn_type']['conn_rule']

        #STDP
        STDP = synapses['conn_type']['stdp_params']
        stdp_model = 'stdp_synapse'
        stdp_alpha =  STDP['alpha']
        stdp_lambda = STDP['lambda']
        stdp_mu_plus = STDP['mu_plus']
        stdp_mu_minus = STDP['mu_minus']
        stdp_tau_plus = STDP['tau_plus']

        #DELAYS
        delay_ee = conns['exc_exc']['delay']
        delay_ei = conns['exc_inh']['delay']
        delay_ie = conns['inh_exc']['delay']
        delay_ii = conns['inh_inh']['delay']
        delay_fwd = conns['fwd']['delay']

        #WEIGHTS
        w_ee = conns['exc_exc']['weight']
        w_ei = conns['exc_inh']['weight']
        w_ie = conns['inh_exc']['weight']
        w_ii = conns['inh_inh']['weight']
        w_fwd_0, w_fwd_max = conns['fwd']['weight'], conns['fwd']['Wmax']

        #NOISE
        rate_noise_exc = noise['rates']['th']['exc']
        rate_noise_inh = noise['rates']['th']['inh']
        w_noise_exc, delay_noise_exc = noise['weights']['th']['exc'], noise['delays']['th']['exc']
        w_noise_inh, delay_noise_inh = noise['weights']['th']['inh'], noise['delays']['th']['inh']

        #SPIKE RECORDERS
        params_device = params['network']['devices']['spike_recorder']
        spike_rec_dict = {
            'neurons': params_device['th'],
            'noise': params_device['noise']['th']
        }

        #EXCITATORY SYNAPSES PARAMETERS TH -->
        conn_exc = SynapsesParameters(
            'Null', {'Null': 0},
            conn_rule_ei, [{"weight": w_ei, "delay": delay_ei}],
            conn_rule_fwd,{"synapse_model": stdp_model,
            "alpha": stdp_alpha, "mu_plus": stdp_mu_plus, "mu_minus": stdp_mu_minus,
            "lambda": stdp_lambda,"weight": w_fwd_0, "Wmax": w_fwd_max, "delay": delay_fwd},
            "Null", {"Null": 0},
            "Null", {"Null": 0},
            "Null", {"Null": 0})

        #INHIBITORY SYNAPSES PARAMETERS TH <--> TH
        conn_inh = SynapsesParameters(
            "Null", {'Null': 0},
            conn_rule_ie,[{"weight": w_ie, "delay": delay_ie}],
            'Null', {'Null': 0},
            'Null', {'Null': 0},
            'Null', {'Null': 0},
            "Null", {"Null": 0})

        #NOISE SYNAPSES PARAMETERS NOISE --> TH
        conn_noise_exc = NoiseParameters({"weight": w_noise_exc, "delay": delay_noise_exc}, rate_noise_exc)
        conn_noise_inh = NoiseParameters({"Null": 0.0}, 0.0)

        #LAYER CREATION & INTRA LAYER CONNECTIONS
        self.LayerTh = Layer(name=name, params_layer=params_th, params_input=params_input, params_sim=params_sim)
        self.LayerTh.Create(neur_dict, conn_exc, conn_inh, conn_noise_exc, conn_noise_inh, spike_rec_dict)


    def Cortex(self):

        params = self.params_default
        params_neuron, params_cx, params_input, params_sim = params['network']['neuron']['cx'], params['network']['structure']['cx'], params['input'], params['simulation']
        noise, synapses = params['network']['noise'], params['network']['synapses']

        print('\n\nCORTEX CREATION')
        name = 'cx'
        params_cx['n_exc'] = params_cx['n_exc_ca'] * params_input['n_class'] * params_input['n_ranks_train'] * params_cx['areas'] * params_sim['current_learning_cycle']
        params_cx['n_inh'] = np.array([params_cx['n_inh_fast'], params_cx['n_inh_slow']])
        params_cx['n_pop_inh'] = sum((np.array(params_cx['n_inh']) > 0))

        # --------- DICTIONARIES

        # NEURON PARAMETERS
        neur_dict = params_neuron.copy()
        for pop, params_pop in neur_dict.items():
            for param, value in params_pop.items():
                if isinstance(value, dict):
                    distr_name = value['distribution']
                    if distr_name == 'uniform':
                        vmin, vmax = value['low'], value['high']
                        new_value = vmin + (vmax - vmin) * np.random.rand(params_cx['n_exc'])
                        params_pop[param] = new_value

        #CONNECTIONS PARAMETERS
        conns_intra, conns_inter = synapses['conn_params']['cx']['intra'], synapses['conn_params']['cx']['inter']

        #CONN RULES
        conn_rule_ee_intra = {'rule': synapses['conn_type']['conn_rule'], 'allow_autapses': conns_intra['exc_exc']['autapses']}
        conn_rule_ee_inter = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_inter['exc_exc']['autapses']}
        conn_rule_eif = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['exc_inh']['autapses']}
        conn_rule_ifif = {'rule': synapses['conn_type']['conn_rule'], 'allow_autapses': conns_intra['inhf_inhf']['autapses']}
        conn_rule_ife = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['inh_exc']['autapses']}
        conn_rule_fwd = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['fwd']['autapses']}
        conn_rule_bwd = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['bwd']['autapses']}

        # STDP
        STDP = synapses['conn_type']['stdp_params']
        stdp_model = 'stdp_synapse'
        stdp_alpha = STDP['alpha']
        stdp_lambda = STDP['lambda']
        stdp_mu_plus = STDP['mu_plus']
        stdp_mu_minus = STDP['mu_minus']
        stdp_tau_plus = STDP['tau_plus']

        #DELAY
        delay_ee_intra = conns_intra['exc_exc']['delay']
        delay_ee_inter = conns_inter['exc_exc']['delay']
        delay_eif,  delay_eis= conns_intra['exc_inhf']['delay'], conns_intra['exc_inhs']['delay']
        delay_ifif, delay_isis = conns_intra['inhf_inhf']['delay'], conns_intra['inhs_inhs']['delay']
        delay_ife, delay_ise = conns_intra['inhf_exc']['delay'], conns_intra['inhs_exc']['delay']
        delay_fwd = conns_intra['fwd']['delay']
        delay_bwd = conns_intra['bwd']['delay']

        #WEIGHTS
        w_ee_intra_0, w_ee_intra_max  = conns_intra['exc_exc']['weight'], conns_intra['exc_exc']['Wmax']
        w_ee_inter_0, w_ee_inter_max = conns_inter['exc_exc']['weight'], conns_inter['exc_exc']['Wmax']
        w_eif, w_eis = conns_intra['exc_inhf']['weight'], conns_intra['exc_inhs']['weight']
        w_ife, w_ise = conns_intra['inhf_exc']['weight'], conns_intra['inhs_exc']['weight']
        w_ifif, w_isis = conns_intra['inhf_inhf']['weight'], conns_intra['inhs_inhs']['weight']
        w_fwd_0, w_fwd_max = conns_intra['fwd']['weight'], conns_intra['fwd']['Wmax']
        w_bwd_0, w_bwd_max = conns_intra['bwd']['weight'], conns_intra['bwd']['Wmax']

        # NOISE
        n_groups, n_exc = params_input['n_ranks_train'] * params_input['n_class'] * params_cx['areas'], params_cx['n_exc']

        rate_cx_train = self.config_complete['awake_training']['noise']['rates']['cx']['exc_1']
        rate_groups = [rate_cx_train for n in range(n_groups)]
        rate_neurons = np.ones(n_exc) * rate_cx_train
        dt_noise_exc = self.config_complete['awake_training']['noise']['ou_noise']['dt_noise']

        rate_noise_exc_1, rate_noise_exc_2 = {'groups': rate_groups, 'neurons': rate_neurons}, noise['rates']['cx']['exc_2']
        rate_noise_inh = noise['rates']['cx']['inh']
        w_noise_exc_1, delay_noise_exc_1 = noise['weights']['cx']['exc_1'], noise['delays']['cx']['exc_1']
        w_noise_exc_2, delay_noise_exc_2 = noise['weights']['cx']['exc_2'], noise['delays']['cx']['exc_2']
        w_noise_inh, delay_noise_inh = noise['weights']['cx']['inh'], noise['delays']['cx']['inh']

        # SPIKE RECORDERS
        params_device = params['network']['devices']['spike_recorder']
        spike_rec_dict = {
            'neurons': params_device['cx'],
            'noise': params_device['noise']['cx']
        }

        #EXCITATORY SYNAPSES PARAMETERS CX -->
        conn_exc = SynapsesParameters(
            conn_rule_ee_intra,{"synapse_model": stdp_model,
                                  "alpha": stdp_alpha, "mu_plus": stdp_mu_plus, "mu_minus": stdp_mu_minus,
                                  "lambda": stdp_lambda,"Wmax": w_ee_intra_max,
                                  "weight": w_ee_intra_0,"delay":delay_ee_intra},
            conn_rule_eif,[{"weight":w_eif, "delay": delay_eif},{"weight": w_eis, "delay": delay_eis}],
            conn_rule_fwd,{"synapse_model": stdp_model,
                             "alpha": stdp_alpha, "mu_plus": stdp_mu_plus,"mu_minus": stdp_mu_minus,
                             "lambda": stdp_lambda, "weight": w_fwd_0, "Wmax":w_fwd_max, "delay": delay_fwd},
            conn_rule_bwd,{"synapse_model": stdp_model,
                                  "alpha": stdp_alpha, "mu_plus": stdp_mu_plus, "mu_minus": stdp_mu_minus,
                                  "lambda": stdp_lambda,"weight": w_bwd_0, "Wmax": w_bwd_max, "delay": delay_bwd},
            "Null", {"Null": 0.0},
            conn_rule_ee_inter,{"synapse_model": stdp_model,
                                  "alpha": stdp_alpha, "mu_plus": stdp_mu_plus, "mu_minus": stdp_mu_minus,
                                  "lambda": stdp_lambda,"weight": w_ee_inter_0,
                                  "Wmax": w_ee_inter_max,"delay": delay_ee_inter})

        #EXCITATORY SYNAPSES PARAMETERS CX <--> CX
        conn_inh = SynapsesParameters(
            conn_rule_ifif,[{"weight": w_ifif, "delay": delay_ifif}, {"weight": w_isis, "delay": delay_isis}],
            conn_rule_ife,[{"weight": w_ife, "delay": delay_ife}, {"weight": w_ise, "delay": delay_ise}],
            "Null", {"Null": 0.0},
            "Null", {"Null": 0.0},
            "Null", {"Null": 0.0},
            "Null",{"Null": 0})

        #NOISE SYNAPSES PARAMETERS NOISE --> CX
        conn_noise_exc_1 = NoiseParameters({"weight": w_noise_exc_1, "delay": delay_noise_exc_1}, rate_noise_exc_1, dt_noise_exc)
        conn_noise_exc_2 = NoiseParameters({"weight": w_noise_exc_2, "delay": delay_noise_exc_2}, rate_noise_exc_2)
        conn_noise_inh = NoiseParameters({"weight": w_noise_inh, "delay": delay_noise_inh}, rate_noise_inh)

        #LAYER CREATION & INTRA LAYER CONNECTIONS
        self.Layer1 = Layer(name=name, params_layer=params_cx, params_input=params_input, params_sim=params_sim)
        self.Layer1.Create(neur_dict, conn_exc, conn_inh, conn_noise_exc_1, conn_noise_inh, spike_rec_dict)


    def Readout(self):

        params = self.params_default
        params_neuron, params_ro, params_input, params_sim = params['network']['neuron']['ro'], params['network']['structure']['ro'], params[
            'input'], params['simulation']
        noise, synapses = params['network']['noise'], params['network']['synapses']

        print('\n\nREADOUT CREATION')
        name = 'ro'
        params_ro['n_exc'] = params_ro['n_exc_ca'] * params_input['n_class']
        params_ro['n_pop_inh'] = 0


        # --------- DICTIONARIES

        # NEURON PARAMETERS
        neur_dict = params_neuron

        #CONNECTIONS PARAMETERS
        conns_intra = synapses['conn_params']['ro']['intra']

        #CONN RULES
        conn_rule_ee = {'rule': synapses['conn_type']['conn_rule'], 'allow_autapses': conns_intra['exc_exc']['autapses']}
        conn_rule_ei = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['exc_inh']['autapses']}
        conn_rule_ii = {'rule': synapses['conn_type']['conn_rule'], 'allow_autapses': conns_intra['inh_inh']['autapses']}
        conn_rule_ie = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['inh_exc']['autapses']}
        conn_rule_bwd = {'rule': synapses['conn_type']['conn_rule']}#, 'allow_autapses': weights_intra['bwd']['autapses']}

        # STDP
        STDP = synapses['conn_type']['stdp_params']
        stdp_model = 'stdp_synapse'
        stdp_alpha = STDP['alpha']
        stdp_lambda = STDP['lambda']
        stdp_mu_plus = STDP['mu_plus']
        stdp_mu_minus = STDP['mu_minus']
        stdp_tau_plus = STDP['tau_plus']

        #DELAY
        delay_ee = conns_intra['exc_exc']['delay']
        delay_ei= conns_intra['exc_inh']['delay']
        delay_ii = conns_intra['inh_inh']['delay']
        delay_ie = conns_intra['inh_exc']['delay']
        delay_bwd = conns_intra['bwd']['delay']

        #WEIGHTS
        w_ee_0, w_ee_max  = conns_intra['exc_exc']['weight'], conns_intra['exc_exc']['Wmax']
        w_ei = conns_intra['exc_inh']['weight']
        w_ie = conns_intra['inh_exc']['weight']
        w_ii = conns_intra['inh_inh']['weight']
        w_bwd_0, w_bwd_max = conns_intra['bwd']['weight'], conns_intra['bwd']['Wmax']

        # NOISE
        dt_noise_exc = self.config_complete['awake_training']['noise']['ou_noise']['dt_noise']
        rate_noise_exc = noise['rates']['ro']['exc']
        rate_noise_inh = noise['rates']['ro']['inh']
        w_noise_exc, delay_noise_exc = noise['weights']['ro']['exc'], noise['delays']['ro']['exc']
        w_noise_inh, delay_noise_inh = noise['weights']['ro']['inh'], noise['delays']['ro']['inh']

        # SPIKE RECORDERS
        params_device = params['network']['devices']['spike_recorder']
        spike_rec_dict = {
            'neurons': params_device['ro'],
            'noise': params_device['noise']['ro']
        }

        #EXCITATORY SYNAPSES PARAMETERS RO -->
        conn_l2_exc = SynapsesParameters(
            conn_rule_ee,{"synapse_model": stdp_model,
                                  "alpha": stdp_alpha, "mu_plus": stdp_mu_plus, "mu_minus": stdp_mu_minus,
                                  "lambda": stdp_lambda, "Wmax": w_ee_max,
                                  "weight": w_ee_0, "delay":delay_ee},
            "Null", {"Null": 0.0},
            "Null", {"Null": 0.0},
            conn_rule_bwd, {"synapse_model": stdp_model,
                                  "alpha": stdp_alpha, "mu_plus": stdp_mu_plus, "mu_minus": stdp_mu_minus,
                                  "lambda": stdp_lambda, "Wmax": w_bwd_max,
                                  "weight": w_bwd_0, "delay":delay_bwd},
            "Null", {"Null": 0.0},
            "Null", {"Null": 0.0})

        conn_l2_noise_exc = NoiseParameters({"weight": w_noise_exc, "delay": delay_noise_exc}, rate_noise_exc, dt_noise_exc)

        #LAYER CREATION & INTRA LAYER CONNECTIONS
        self.Layer2 = Layer(name=name, params_layer=params_ro, params_input=params_input, params_sim=params_sim)
        self.Layer2.Create(neur_dict, conn_l2_exc,None, conn_l2_noise_exc,None, spike_rec_dict)

        self.Layer2.cycle = params_sim['cycle']


    def Connections(self):

        cx_areas = self.params_default['network']['structure']['cx']['areas']

        #CONNECTION LIST
        inter_area_list = [self.Layer1]
        inter_lay_fwd_list = [[self.LayerTh, self.Layer1], [self.Layer1, self.Layer2]]
        inter_lay_bwd_list = [[self.Layer1, self.LayerTh]]
        self.inter_lay_list = inter_lay_fwd_list + inter_lay_bwd_list
        self.intra_lay_list = [[self.Layer1, self.Layer1], [self.Layer2, self.Layer2]]

        #INTER AREA CONNECTIONS
        if cx_areas>1: ConnectInter(inter_area_list)

        #INTER LAYER CONNECTIONS
        ConnectFW(inter_lay_fwd_list)
        ConnectBW(inter_lay_bwd_list)

        #CONNECTIONS DICTIONARY
        neur_noise = tuple([self.Layer1.neur_area_noise_exc1[area] for area in range(cx_areas)])
        neur_noise_rem = tuple([self.Layer1.neur_area_noise_exc2[area] for area in range(cx_areas)])
        neur_th_exc = tuple([self.LayerTh.neur_area_exc[area] for area in range(cx_areas)])
        neur_th_inh = tuple([self.LayerTh.neur_area_inh[area] for area in range(cx_areas)])
        neur_cx_exc = tuple([self.Layer1.neur_area_exc[area] for area in range(cx_areas)])
        neur_cx_inh = tuple([[self.Layer1.neur_pop_inh[pop][area] for area in range(cx_areas)] for pop in range(self.Layer1.n_pop_inh)])

        conn_th_cx = nest.GetConnections(self.LayerTh.exc, self.Layer1.exc)
        conn_cx_th = nest.GetConnections(self.Layer1.exc, self.LayerTh.exc)
        conn_cx_ee = nest.GetConnections(self.Layer1.exc, self.Layer1.exc)
        conn_cx_ei = [[nest.GetConnections(neur_cx_exc[area], neur_cx_inh[pop][area]) for area in range(cx_areas)] for pop in range(self.Layer1.n_pop_inh)]
        conn_cx_ie = [[nest.GetConnections(neur_cx_inh[pop][area], neur_cx_exc[area]) for area in range(cx_areas)] for pop in range(self.Layer1.n_pop_inh)]
        conn_cx_ii = [[nest.GetConnections(neur_cx_inh[pop][area], neur_cx_inh[pop][area]) for area in range(cx_areas)] for pop in range(self.Layer1.n_pop_inh)]
        conn_cx_ro = nest.GetConnections(self.Layer1.exc, self.Layer2.exc)
        conn_ro_ro = nest.GetConnections(self.Layer2.exc, self.Layer2.exc)
        conn_noise_cx_1 = [nest.GetConnections(neur_noise[area], neur_cx_exc[area]) for area in range(cx_areas)]
        conn_noise_cx_2 = [nest.GetConnections(neur_noise_rem[area], neur_cx_exc[area]) for area in range(cx_areas)]
        conn_noise_cx_exc_1 = nest.GetConnections(self.Layer1.noise_exc, self.Layer1.exc)
        conn_noise_cx_exc_2 = nest.GetConnections(self.Layer1.noise_rem, self.Layer1.exc)
        conn_noise_cx_inh = nest.GetConnections(self.Layer1.noise_inh, self.Layer1.inh)
        conn_noise_th_exc = nest.GetConnections(self.LayerTh.noise_exc, self.LayerTh.exc)
        conn_noise_ro_exc = nest.GetConnections(self.Layer2.noise_exc, self.Layer2.exc)

        self.pop_dict = {
            'th':{
                'exc': self.LayerTh.exc,
                'inh': self.LayerTh.inh
            },
            'cx':{
                'exc': self.Layer1.exc,
                'inh': self.Layer1.inh
            },
            'noise': {
                'th': {
                    'exc': self.LayerTh.noise_exc,
                },
                'cx': {
                    'exc_1': self.Layer1.noise_exc,
                    'exc_2': self.Layer1.noise_rem,
                    'inh': self.Layer1.noise_inh,
                },
                'ro': {
                    'exc': self.Layer2.noise_exc,
                },
            },
        }

        self.conn_dict_layers = {
            'th':{
                'all':{
                    'fwd': {'conn': conn_th_cx, 'source': conn_th_cx.get('source'),
                               'target': conn_th_cx.get('target')},
                },
            },
            'cx':{
                'all':{
                    'exc_exc': {'conn': conn_cx_ee, 'source': conn_cx_ee.get('source'),
                               'target': conn_cx_ee.get('target')},
                    'bwd': {'conn': conn_cx_th, 'source': conn_cx_th.get('source'),
                            'target': conn_cx_th.get('target')},
                    'fwd': {'conn': conn_cx_ro, 'source': conn_cx_ro.get('source'),
                               'target': conn_cx_ro.get('target')}
                },
                'intra': {
                    'exc_inh': {'conn': conn_cx_ei,
                                'source': [[conn_cx_ei[pop][area].get('source') for area in range(cx_areas)] for pop in
                                           range(self.Layer1.n_pop_inh)],
                                'target': [[conn_cx_ei[pop][area].get('target') for area in range(cx_areas)] for pop in
                                           range(self.Layer1.n_pop_inh)]},
                    'inh_exc': {'conn': conn_cx_ie,
                                'source': [[conn_cx_ie[pop][area].get('source') for area in range(cx_areas)] for pop in
                                           range(self.Layer1.n_pop_inh)],
                                'target': [[conn_cx_ie[pop][area].get('target') for area in range(cx_areas)] for pop in
                                           range(self.Layer1.n_pop_inh)]},
                    'inh_inh': {'conn': conn_cx_ii,
                                'source': [[conn_cx_ii[pop][area].get('source') for area in range(cx_areas)] for pop in
                                           range(self.Layer1.n_pop_inh)],
                                'target': [[conn_cx_ii[pop][area].get('target') for area in range(cx_areas)] for pop in
                                           range(self.Layer1.n_pop_inh)]},

                },
            },
            'ro':{
                'all':{
                    'exc_exc': {'conn': conn_ro_ro, 'source': conn_ro_ro.get('source'),
                               'target': conn_ro_ro.get('target')},
                },
            },
            'noise': {
                'all': {
                    'th': {
                        'exc': {'conn': conn_noise_th_exc, 'source': conn_noise_th_exc.get('source'),
                               'target': conn_noise_th_exc.get('target')},
                    },

                    'cx': {
                        'exc_1': {'conn': conn_noise_cx_exc_1, 'source': conn_noise_cx_exc_1.get('source'),
                                 'target': conn_noise_cx_exc_1.get('target')},
                        'exc_2': {'conn': conn_noise_cx_exc_2, 'source': conn_noise_cx_exc_2.get('source'),
                                  'target': conn_noise_cx_exc_2.get('target')},
                        'inh': {'conn': conn_noise_cx_inh, 'source': conn_noise_cx_inh.get('source'),
                               'target': conn_noise_cx_inh.get('target')},
                    },

                    'ro': {
                        'exc': {'conn': conn_noise_ro_exc, 'source': conn_noise_ro_exc.get('source'),
                                 'target': conn_noise_ro_exc.get('target')},
                    },
                },
                'intra': {

                    'cx': {
                        'exc_1': {'conn': conn_noise_cx_1,},
                        'exc_2': {'conn': conn_noise_cx_2,},
                    },
                },
            },
        }


        if self.save_all_output == True:

            conn_th_ei = [nest.GetConnections(neur_th_exc[area], neur_th_inh[area]) for area in range(cx_areas)]
            conn_th_ie = [nest.GetConnections(neur_th_inh[area], neur_th_exc[area]) for area in range(cx_areas)]

            conn_dict = {
                'th': {
                    'intra': {
                        'exc_inh': {'conn': conn_th_ei,
                                    'source': [conn_th_ei[area].get('source') for area in range(cx_areas)],
                                    'target': [conn_th_ei[area].get('target') for area in range(cx_areas)]},
                        'inh_exc': {'conn': conn_th_ie,
                                    'source': [conn_th_ie[area].get('source') for area in range(cx_areas)],
                                    'target': [conn_th_ie[area].get('target') for area in range(cx_areas)]},
                    },
                },
            }

            self.conn_dict_layers = MergeDicts(self.conn_dict_layers, conn_dict)


        if self.stage in ['nrem', 'rem']:

            conn_th_cx = [nest.GetConnections(neur_th_exc[area], neur_cx_exc[area]) for area in range(cx_areas)]
            conn_cx_th = [nest.GetConnections(neur_cx_exc[area], neur_th_exc[area]) for area in range(cx_areas)]
            conn_cx_ee_intra = [nest.GetConnections(neur_cx_exc[area], neur_cx_exc[area]) for area in range(cx_areas)]

            conn_dict = {
                'th': {
                    'intra': {
                        'fwd': {'conn': conn_th_cx,},
                    },
                },
                'cx': {
                    'intra': {
                        'exc_exc': {'conn': conn_cx_ee_intra,},
                        'bwd': {'conn': conn_cx_th,},
                    },
                },
            }

            self.conn_dict_layers = MergeDicts(self.conn_dict_layers, conn_dict)

            if cx_areas>1:
                conn_cx_ee_inter = [nest.GetConnections(neur_cx_exc[0], neur_cx_exc[1]), nest.GetConnections(neur_cx_exc[1], neur_cx_exc[0])]
                conn_dict = {
                    'cx': {
                        'inter': {
                            'exc_exc': {'conn': conn_cx_ee_inter,},
                        },
                    },
                }

                self.conn_dict_layers = MergeDicts(self.conn_dict_layers, conn_dict)

            if self.save_all_output == False:
                conn_th_ie = [nest.GetConnections(neur_th_inh[area], neur_th_exc[area]) for area in range(cx_areas)]

                conn_dict = {
                    'th': {
                        'intra': {
                            'inh_exc': {'conn': conn_th_ie,},
                        },
                    },
                }

                self.conn_dict_layers = MergeDicts(self.conn_dict_layers, conn_dict)
