#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  Parent_BrainStates.py
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

import time
import numpy as np
from Parent_Connections import SynapticDynamics, SynapticDynamicsNREM, SynapticDynamicsREM, SynapticModulation, \
    SaveWeights, LoadWeights, WeightMap, WeightIntraGroup
from Parent_SpikeDetector import SaveEvents, Rastergram, Accuracy
from utils import DownloadNetParams, MergeDicts, KeyExists
import Parent_Noise as Noise
import nest, os

def Training(ThaCo):

    LayerTh = ThaCo.LayerTh
    Layer1 = ThaCo.Layer1
    Layer2 = ThaCo.Layer2
    train = ThaCo.train
    conn_layers = ThaCo.conn_dict_layers
    t_simulation = ThaCo.t_simulation
    save_state = ThaCo.save_state
    save_path = ThaCo.save_path
    startTime = ThaCo.startTime
    stage = ThaCo.stage_id
    cx_groups_shuffling = ThaCo.trial_dict['training']['index shuffling']
    cycle = ThaCo.current_learning_cycle

    if ThaCo.save_all_output == True:
        weights_save_data_list = [LayerTh, Layer1, Layer2]
        save_path_init = save_path + '_init'
        save_syn_init = save_state + '_init'
        params_save_data_list = [LayerTh, Layer1]
        if not os.path.exists(save_syn_init):
            try:
                os.makedirs(save_syn_init)
            except BaseException as err:
                raise err
        if not os.path.exists(save_path_init):
            try:
                os.makedirs(save_path_init)
            except BaseException as err:
                raise err

        SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_syn_init,
                    stage=stage + '_init', save_all_output=ThaCo.save_all_output)
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path_init, stage=stage+'_init')

    print('\n\n')
    print('#####################            #####################')
    print('#####################  TRAINING  #####################')
    print('#####################            #####################')
    print('\n\n')

    t_current = t_simulation
    t_training = (train.t_img + train.t_pause) * train.n_img


    print('\nPre-Training simulated time: %g s' % (t_current / 1000.))
    print('Pre-Training machine time: %g s' % (time.time() - startTime))
    print('Image simulation time: %g s\n' % (train.t_img / 1000.))

    #SIMULATION
    Noise.Thalamic(train, LayerTh, t_start=t_current)
    Noise.ContextExc(train, Layer1, t_start=t_current)
    Noise.ContextInh(train, Layer1, t_start=t_current)
    Noise.ContextExc(train, Layer2, t_start=t_current)

    nest.Simulate(t_training)
    t_current += t_training

    print('\nPost-Training simulated time: %g s' % (t_current / 1000.))
    print('Post-Training machine time: %g s\n' % (time.time() - startTime))

    #SAVING
    print('\n\nSAVING POST-TRAINING')
    weights_save_data_list, spikes_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh]
    if ThaCo.save_all_output == True:
        weights_save_data_list, spikes_save_data_list, params_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh, Layer2], [Layer1, LayerTh]
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path, stage=stage)

    SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_state,
                stage=stage, save_all_output=ThaCo.save_all_output)
    SaveEvents(spikes_save_data_list, save_path, stage)

    Rastergram([LayerTh, Layer1], save_path, t_simulation, stage, shuffling=cx_groups_shuffling, fr=False, net=['exc','inh'])
    WeightMap(Layer1, conn_layers,cx_groups_shuffling, save_path, stage + '_log', cycle, log=True)

    ThaCo.t_simulation += t_training

def Test(ThaCo):

    LayerTh = ThaCo.LayerTh
    Layer1 = ThaCo.Layer1
    Layer2 = ThaCo.Layer2
    test = ThaCo.test
    conn_layers = ThaCo.conn_dict_layers
    t_simulation = ThaCo.t_simulation
    save_path = ThaCo.save_path
    save_state = ThaCo.save_state
    cx_groups_shuffling = ThaCo.trial_dict['training']['index shuffling']
    startTime = ThaCo.startTime
    stage = ThaCo.stage_id

    print('\n\n')
    print('#####################            #####################')
    print('#####################  TEST %s   #####################'%(stage))
    print('#####################            #####################')
    print('\n\n')

    t_current = t_simulation
    t_test = ThaCo.t_test
    t_relaxation = ThaCo.params_stage['times']['t_relaxation']

    print('\nPre-Test simulated time: %g s' % (t_current / 1000.))
    print('Pre-Test machine time: %g s' % (time.time() - startTime))
    print('Image simulation time: %g s\n' % (test.t_img / 1000.))

    #RELAXATION
    if t_relaxation > 0:
        rate_relaxation = ThaCo.params_stage['noise']['rates']['cx']['inh']['relaxation']
        Noise.Aspecific(Layer1.noise_inh, rate_relaxation, t_current, t_relaxation)
        nest.Simulate(t_relaxation)
        t_current += t_relaxation

    #SIMULATION
    Noise.Thalamic(test, LayerTh, t_start=t_current)
    Noise.ContextInh(test, Layer1, t_start=t_current)
    nest.Simulate(t_test)
    t_current += t_test

    print('\nPost-Test simulated time: %g s' % ((t_current) / 1000.))
    print('Post-Test machine time: %g s\n' % (time.time() - startTime))

    print('\n\n')
    print('#####################            #####################')
    print('#####################  ACCURACY  #####################')
    print('#####################            #####################')
    print('\n\n')

    time_pre_acc = time.time()
    #Accuracy([Layer1,Layer2], train, test, conn_layers, t_simulation, t_test, stage, save_path)
    print('time accuracy: %lg' %(time.time()-time_pre_acc))

    #SAVING
    print('\n\nSAVING POST-TEST')
    weights_save_data_list, spikes_save_data_list = [], [Layer1, LayerTh]
    if ThaCo.save_all_output == True:
        weights_save_data_list, spikes_save_data_list, params_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh, Layer2], [Layer1, LayerTh]
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path, stage=stage)

    SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_state,
                stage=stage, save_all_output=ThaCo.save_all_output)
    SaveEvents(spikes_save_data_list, save_path, stage)

    Rastergram([LayerTh, Layer1], save_path, t_simulation, stage + '_test', shuffling=cx_groups_shuffling, fr=False, net=['exc','inh'])

    ThaCo.t_simulation += t_test

def NREM(ThaCo):

    LayerTh = ThaCo.LayerTh
    Layer1 = ThaCo.Layer1
    Layer2 = ThaCo.Layer2
    conn_layers = ThaCo.conn_dict_layers
    t_simulation = ThaCo.t_simulation
    save_state = ThaCo.save_state
    save_path = ThaCo.save_path
    cx_groups_shuffling = ThaCo.trial_dict['training']['index shuffling']
    startTime = ThaCo.startTime
    stage = ThaCo.stage_id
    cycle = ThaCo.current_learning_cycle
    n_exc_ca = Layer1.n_exc_ca
    n_class = Layer1.n_class
    n_ranks = Layer1.n_ranks_train
    stage_name = ThaCo.stage


    params_old = ThaCo.params_stage.copy()
    params_stage = ThaCo.config_complete[stage_name]

    if ThaCo.save_all_output == True:
        weights_save_data_list = [LayerTh, Layer1, Layer2]
        save_path_init = save_path + '_init'
        save_syn_init = save_state + '_init'
        params_save_data_list = [LayerTh, Layer1]
        if not os.path.exists(save_syn_init):
            try:
                os.makedirs(save_syn_init)
            except BaseException as err:
                raise err
        if not os.path.exists(save_path_init):
            try:
                os.makedirs(save_path_init)
            except BaseException as err:
                raise err

        SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_syn_init,
                    stage=stage + '_init', save_all_output=ThaCo.save_all_output)
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path_init, stage=stage+'_init')

    print('\n\n')
    print('#####################                #####################')
    print('#####################     NREM %s    #####################' % (stage))
    print('#####################                #####################')
    print('\n\n')

    t_current = t_simulation
    t_thermalization = params_stage['times']['t_thermalization']
    t_nrem =  params_stage['times']['t_nrem']
    t_sleep = t_nrem + t_thermalization

    conn_l1_l1_intra = conn_layers['cx']['intra']['exc_exc']['conn']
    if Layer1.areas>1: conn_l1_l1_inter = conn_layers['cx']['inter']['exc_exc']['conn']
    lambda_intra = params_stage['synapses']['conn_params']['cx']['intra']['exc_exc']['lambda']
    rate_high = params_stage['noise']['rates']['cx']['exc_1']


    print('\nPre-NREM Thermalization simulated time: %g s' % (t_current / 1000.))
    print('Pre-NREM Thermalization machine time: %g s' % (time.time() - startTime))
    print('NREM Thermalization simulation time: %g s\n' % (t_thermalization / 1000.))

    ## WEIGHT MODULATION (NEUROMODULATION)

    #CX INTRA
    if KeyExists(params_stage, ('synapses', 'conn_params', 'cx', 'intra', 'exc_exc', 'modulation')):
        modulation = params_stage['synapses']['conn_params']['cx']['intra']['exc_exc'].pop('modulation')
        if KeyExists(modulation, 'weight'):
            w_new = modulation['weight']
            ratio_cx_intra_on = SynapticModulation(w_new=w_new,conn=conn_l1_l1_intra,conn_type='intra',n_exc_ca=n_exc_ca,n_group_area=n_class*n_ranks)
        elif KeyExists(modulation, 'ratio'):
            ratio_cx_intra_on = modulation['ratio']
        else:
            ratio_cx_intra_on = np.ones(Layer1.areas)
    else:
        ratio_cx_intra_on = np.ones(Layer1.areas)

    #CX INTER
    ratio_cx_inter_on = np.ones(Layer1.areas) * 1e-20

    if Layer1.areas > 1:
        modulation_dict = {'synapses': {'conn_params': {
            'cx': {
                'intra': {'exc_exc': {'modulation': {'ratio': ratio_cx_intra_on}}},
                'inter': {'exc_exc': {'modulation': {'ratio': ratio_cx_inter_on}}}},
        }}}
    else:
        modulation_dict = {'synapses': {'conn_params': {'cx': {'intra': {'exc_exc': {'modulation': {'ratio': ratio_cx_intra_on}}}}}}}

    params_new = MergeDicts(params_stage, modulation_dict)
    ThaCo.params_stage = params_new
    ThaCo.SetNetState(params_start_state=params_old, params_new_state=params_new)

    if ThaCo.save_all_output == True:
        weights_save_data_list = [LayerTh, Layer1, Layer2]
        save_path_mod = save_path + '_modulation'
        save_syn_mod = save_state + '_modulation'
        params_save_data_list = [LayerTh, Layer1]
        if not os.path.exists(save_syn_mod):
            try:
                os.makedirs(save_syn_mod)
            except BaseException as err:
                raise err
        if not os.path.exists(save_path_mod):
            try:
                os.makedirs(save_path_mod)
            except BaseException as err:
                raise err

        SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_syn_mod,
                    stage=stage + '_modulation', save_all_output=ThaCo.save_all_output)
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path_mod, stage=stage+'_modulation')

    #------------------- SIMULATION

    #Thermalization
    for area in range(Layer1.areas): conn_l1_l1_intra[area].set({'lambda': 0})
    if t_thermalization > 0:
        Noise.Aspecific(Layer1.noise_exc, rate_high, t_current, t_thermalization)
        nest.Simulate(t_thermalization)
        t_current += t_thermalization

    #NREM
    print('\nPost-NREM Thermalization simulated time: %g s' % (t_current / 1000.))
    print('Post-NREM Thermalization machine time: %g s' % (time.time() - startTime))
    print('NREM simulation time: %g s\n' % (t_nrem / 1000.))

    for area in range(Layer1.areas): conn_l1_l1_intra[area].set({'lambda': lambda_intra})

    if t_nrem > 0:
        if ThaCo.save_all_output == True:
            n_step_nrem = 10
            t_step_nrem = t_nrem / n_step_nrem
            for step in range(n_step_nrem):
                Noise.Aspecific(Layer1.noise_exc, rate_high, t_current, t_step_nrem)
                nest.Simulate(t_step_nrem)
                t_current += t_step_nrem
                stage_step = f'{stage}_{step}'
                SaveWeights(layer_list=['cx'], conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_state,
                            stage=stage_step, save_all_output=ThaCo.save_all_output)
        else:
            Noise.Aspecific(Layer1.noise_exc, rate_high, t_current, t_nrem)
            nest.Simulate(t_nrem)
            t_current += t_nrem

    #WEIGHT RESCALING
    ratio_cx_intra_off = 1. / ratio_cx_intra_on
    if Layer1.areas > 1:
        #mu_intra = WeightIntraGroup(conn=conn_l1_l1_intra, conn_type='intra', n_exc_ca=n_exc_ca)
        #mu_inter = WeightIntraGroup(conn=conn_l1_l1_inter, conn_type='inter', n_exc_ca=n_exc_ca)
        #ratio_cx_intra_off = (mu_intra / np.mean(ratio_cx_intra_on)) * (1 / mu_inter)
        ratio_cx_inter_off = 1 / ratio_cx_inter_on

    modulation_dict = {}

    #CX EXC-EXC
    if Layer1.areas > 1:
        modulation_cx_ee = {'synapses': {'conn_params':
            {'cx': {
                'intra': {'exc_exc': {'modulation': {'ratio': ratio_cx_intra_off}}},
                'inter': {'exc_exc': {'modulation': {'ratio': ratio_cx_inter_off}}} },
        }}}
    else:
        modulation_cx_ee = {'synapses': {'conn_params': {
            'cx': {'intra': {'exc_exc': {'modulation': {'ratio': ratio_cx_intra_off}}}},
        }}}

    modulation_dict = MergeDicts(modulation_dict, modulation_cx_ee)

    # CX BWD
    if KeyExists(params_stage, ('synapses', 'conn_params', 'cx', 'intra', 'bwd', 'modulation')):
        ratio_cx_intra_bwd_on = params_stage['synapses']['conn_params']['cx']['intra']['bwd']['modulation']['ratio']
        ratio_cx_intra_bwd_off = 1 / ratio_cx_intra_bwd_on
        modulation_cx_bwd = {'synapses': {'conn_params': {
            'cx': {'intra': {'bwd': {'modulation': {'ratio': ratio_cx_intra_bwd_off}}}},
        }}}
        modulation_dict = MergeDicts(modulation_dict, modulation_cx_bwd)

    # TH FWD
    if KeyExists(params_stage, ('synapses', 'conn_params', 'th', 'intra', 'fwd', 'modulation')):
        ratio_th_intra_fwd_on = params_stage['synapses']['conn_params']['th']['intra']['fwd']['modulation']['ratio']
        ratio_th_intra_fwd_off = 1 / ratio_th_intra_fwd_on
        modulation_th_fwd = {'synapses': {'conn_params': {
            'th': {'intra': {'fwd': {'modulation': {'ratio': ratio_th_intra_fwd_off}}}},
        }}}
        modulation_dict = MergeDicts(modulation_dict, modulation_th_fwd)

    params_old = ThaCo.params_stage.copy()
    params_new = ThaCo.config_complete['awake_test'].copy()
    params_new = MergeDicts(params_new, modulation_dict)
    ThaCo.params_stage = params_new
    ThaCo.SetNetState(params_start_state=params_old, params_new_state=params_new)

    print('\nPost-NREM simulated time: %g s' % ((t_current) / 1000.))
    print('Post-NREM machine time: %g s\n' % (time.time() - startTime))

    #SAVING
    print('\n\nSAVING NREM')
    weights_save_data_list, spikes_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh]
    if ThaCo.save_all_output == True:
        weights_save_data_list, spikes_save_data_list, params_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh, Layer2], [Layer1, LayerTh]
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path, stage=stage)

    SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_state,
                stage=stage, save_all_output=ThaCo.save_all_output)
    SaveEvents(spikes_save_data_list, save_path, stage)

    Rastergram([LayerTh, Layer1], save_path, t_simulation, stage, shuffling=cx_groups_shuffling, net=['exc', 'inh'], fr=False, raster=True)
    WeightMap(Layer1, conn_layers,cx_groups_shuffling, save_path, stage + '_log', cycle, log=True)

    ThaCo.t_simulation += t_sleep

def REM(ThaCo):

    LayerTh = ThaCo.LayerTh
    Layer1 = ThaCo.Layer1
    Layer2 = ThaCo.Layer2
    conn_layers = ThaCo.conn_dict_layers
    t_simulation = ThaCo.t_simulation
    save_state = ThaCo.save_state
    save_path = ThaCo.save_path
    cx_groups_shuffling = ThaCo.trial_dict['training']['index shuffling']
    startTime = ThaCo.startTime
    stage = ThaCo.stage_id
    cycle = ThaCo.current_learning_cycle
    n_exc_ca = Layer1.n_exc_ca
    n_class = Layer1.n_class
    n_ranks = Layer1.n_ranks_train
    stage_name = ThaCo.stage

    params_old = ThaCo.params_stage.copy()
    params_stage = ThaCo.config_complete[stage_name]

    if ThaCo.save_all_output == True:
        weights_save_data_list = [LayerTh, Layer1, Layer2]
        save_path_init = save_path + '_init'
        save_syn_init = save_state + '_init'
        params_save_data_list = [LayerTh, Layer1]
        if not os.path.exists(save_syn_init):
            try:
                os.makedirs(save_syn_init)
            except BaseException as err:
                raise err

        if not os.path.exists(save_path_init):
            try:
                os.makedirs(save_path_init)
            except BaseException as err:
                raise err

        SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_syn_init,
                    stage=stage + '_init', save_all_output=ThaCo.save_all_output)
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path_init, stage=stage+'_init')

    print('\n\n')
    print('#####################                  #####################')
    print('#####################      REM  %s     #####################'%(stage))
    print('#####################                  #####################')
    print('\n\n')

    t_current = t_simulation
    t_thermalization =params_stage['times']['t_thermalization']
    t_rem = params_stage['times']['t_rem']
    t_sleep = t_rem + t_thermalization

    conn_l1_l1_intra = conn_layers['cx']['intra']['exc_exc']['conn']
    conn_l1_l1_inter = conn_layers['cx']['inter']['exc_exc']['conn']
    lambda_intra = params_stage['synapses']['conn_params']['cx']['intra']['exc_exc']['lambda']
    lambda_inter = params_stage['synapses']['conn_params']['cx']['inter']['exc_exc']['lambda']

    rate_high = params_stage['noise']['rates']['cx']['exc_1']
    rate_low = params_stage['noise']['rates']['cx']['exc_2']

    print('\nPre-REM Thermalization simulated time: %g s' % (t_current / 1000.))
    print('Pre-REM Thermalization machine time: %g s' % (time.time() - startTime))
    print('REM Thermalization time: %g s\n' % (t_thermalization / 1000.))

    ## WEIGHT MODULATION (NEUROMODULATION)

    #CX INTRA
    if KeyExists(params_stage, ('synapses', 'conn_params', 'cx', 'intra', 'exc_exc', 'modulation')):
        modulation = params_stage['synapses']['conn_params']['cx']['intra']['exc_exc'].pop('modulation')
        if KeyExists(modulation, 'weight'):
            w_new_intra = modulation['weight']
            ratio_cx_intra_on = SynapticModulation(w_new=w_new_intra,conn=conn_l1_l1_inter,conn_type='intra',n_exc_ca=n_exc_ca,n_group_area=n_class*n_ranks)
        elif KeyExists(modulation, 'ratio'):
            ratio_cx_intra_on = modulation['ratio']
        else:
            ratio_cx_intra_on = np.ones(Layer1.areas)
    else:
        ratio_cx_intra_on = np.ones(Layer1.areas)

    # CX INTER
    if KeyExists(params_stage, ('synapses', 'conn_params', 'cx', 'inter', 'exc_exc', 'modulation')):
        modulation = params_stage['synapses']['conn_params']['cx']['inter']['exc_exc'].pop('modulation')
        if KeyExists(modulation, 'weight'):
            w_new_inter = modulation['weight']
            ratio_cx_inter_on = SynapticModulation(w_new=w_new_inter,conn=conn_l1_l1_inter,conn_type='inter',n_exc_ca=n_exc_ca,n_group_area=n_class*n_ranks)
        elif KeyExists(modulation, 'ratio'):
            ratio_cx_inter_on = modulation['ratio']
        else:
            ratio_cx_inter_on = np.ones(Layer1.areas)
    else:
        ratio_cx_inter_on = np.ones(Layer1.areas)

    modulation_dict = {}

    modulation_cx_ee = {'synapses': {'conn_params': {
        'cx': {
            'intra': {'exc_exc': {'modulation': {'ratio': ratio_cx_intra_on}}},
            'inter': {'exc_exc': {'modulation': {'ratio': ratio_cx_inter_on}}}},
    }}}

    modulation_dict = MergeDicts(modulation_dict, modulation_cx_ee)

    # CX BWD
    if KeyExists(params_stage, ('synapses', 'conn_params', 'cx', 'intra', 'bwd', 'modulation')):
        ratio_cx_intra_bwd_on = params_stage['synapses']['conn_params']['cx']['intra']['bwd']['modulation']['ratio']
        ratio_cx_intra_bwd_off = 1 / ratio_cx_intra_bwd_on
        modulation_cx_bwd = {'synapses': {'conn_params': {
            'cx': {'intra': {'bwd': {'modulation': {'ratio': ratio_cx_intra_bwd_off}}}},
        }}}
        modulation_dict = MergeDicts(modulation_dict, modulation_cx_bwd)

    # TH FWD
    if KeyExists(params_stage, ('synapses', 'conn_params', 'th', 'intra', 'fwd', 'modulation')):
        ratio_th_intra_fwd_on = params_stage['synapses']['conn_params']['th']['intra']['fwd']['modulation']['ratio']
        ratio_th_intra_fwd_off = 1 / ratio_th_intra_fwd_on
        modulation_th_fwd = {'synapses': {'conn_params': {
            'th': {'intra': {'fwd': {'modulation': {'ratio': ratio_th_intra_fwd_off}}}},
        }}}
        modulation_dict = MergeDicts(modulation_dict, modulation_th_fwd)

    params_new = MergeDicts(ThaCo.config_complete[stage_name], modulation_dict)
    ThaCo.params_stage = params_new
    ThaCo.SetNetState(params_start_state=params_old, params_new_state=params_new)

    if ThaCo.save_all_output == True:
        weights_save_data_list = [LayerTh, Layer1, Layer2]
        save_path_mod = save_path + '_modulation'
        save_syn_mod = save_state + '_modulation'
        params_save_data_list = [LayerTh, Layer1]
        if not os.path.exists(save_syn_mod):
            try:
                os.makedirs(save_syn_mod)
            except BaseException as err:
                raise err
        if not os.path.exists(save_path_mod):
            try:
                os.makedirs(save_path_mod)
            except BaseException as err:
                raise err

        SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_syn_mod,
                    stage=stage + '_modulation', save_all_output=ThaCo.save_all_output)
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path_mod, stage=stage+'_modulation')

    #Thermalization
    for area in range(Layer1.areas): conn_l1_l1_intra[area].set({'lambda': 0})
    for area in range(Layer1.areas): conn_l1_l1_inter[area].set({'lambda': 0})

    if t_thermalization > 0:
        Noise.Aspecific(Layer1.noise_exc, rate_high, t_current, t_thermalization)
        Noise.Aspecific(Layer1.noise_rem, rate_low, t_current, t_thermalization)
        nest.Simulate(t_thermalization)
        t_current += t_thermalization

    #REM
    print('\nPost-REM Thermalization simulated time: %g s' % (t_current / 1000.))
    print('Post-REM Thermalization machine time: %g s' % (time.time() - startTime))
    print('REM simulation time: %g s\n' % (t_rem / 1000.))

    for area in range(Layer1.areas): conn_l1_l1_intra[area].set({'lambda': lambda_intra})
    for area in range(Layer1.areas): conn_l1_l1_inter[area].set({'lambda': lambda_inter})

    if t_rem > 0:
        if ThaCo.save_all_output == True:
            n_step_rem = 10
            t_step_rem = t_rem / n_step_rem
            for step in range(n_step_rem):
                Noise.Aspecific(Layer1.noise_exc, rate_high, t_current, t_step_rem)
                Noise.Aspecific(Layer1.noise_rem, rate_low, t_current, t_step_rem)
                nest.Simulate(t_step_rem)
                t_current += t_step_rem
                stage_step = f'{stage}_{step}'
                SaveWeights(layer_list=['cx'], conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_state,
                            stage=stage_step, save_all_output=ThaCo.save_all_output)
        else:
            Noise.Aspecific(Layer1.noise_exc, rate_high, t_current, t_rem)
            Noise.Aspecific(Layer1.noise_rem, rate_low, t_current, t_rem)
            nest.Simulate(t_rem)
            t_current += t_rem

    print('\nPost-REM simulated time: %g s' % ((t_current) / 1000.))
    print('Post-REM machine time: %g s\n' % (time.time() - startTime))

    #WEIGHT RESCALING
    ratio_cx_intra_off = 1 / ratio_cx_intra_on
    ratio_cx_inter_off = 1 / ratio_cx_inter_on

    modulation_dict = {}

    modulation_cx_ee = {'synapses': {'conn_params': {
        'cx': {
            'intra': {'exc_exc': {'modulation': {'ratio': ratio_cx_intra_off}}},
            'inter': {'exc_exc': {'modulation': {'ratio': ratio_cx_inter_off}}}},
    }}}
    modulation_dict = MergeDicts(modulation_dict, modulation_cx_ee)

    # CX BWD
    if KeyExists(params_stage, ('synapses', 'conn_params', 'cx', 'intra', 'bwd', 'modulation')):
        ratio_cx_intra_bwd_on = params_stage['synapses']['conn_params']['cx']['intra']['bwd']['modulation']['ratio']
        ratio_cx_intra_bwd_off = 1 / ratio_cx_intra_bwd_on
        modulation_cx_bwd = {'synapses': {'conn_params': {
            'cx': {'intra': {'bwd': {'modulation': {'ratio': ratio_cx_intra_bwd_off}}}},
        }}}
        modulation_dict = MergeDicts(modulation_dict, modulation_cx_bwd)

    # TH FWD
    if KeyExists(params_stage, ('synapses', 'conn_params', 'th', 'intra', 'fwd', 'modulation')):
        ratio_th_intra_fwd_on = params_stage['synapses']['conn_params']['th']['intra']['fwd']['modulation']['ratio']
        ratio_th_intra_fwd_off = 1 / ratio_th_intra_fwd_on
        modulation_th_fwd = {'synapses': {'conn_params': {
            'th': {'intra': {'fwd': {'modulation': {'ratio': ratio_th_intra_fwd_off}}}},
        }}}
        modulation_dict = MergeDicts(modulation_dict, modulation_th_fwd)



    params_old = ThaCo.params_stage.copy()
    params_new = ThaCo.config_complete['awake_test'].copy()
    params_new = MergeDicts(params_new, modulation_dict)
    ThaCo.params_stage = params_new
    ThaCo.SetNetState(params_start_state=params_old, params_new_state=params_new)

    #SAVING
    print('\n\nSAVING REM')
    weights_save_data_list, spikes_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh]
    if ThaCo.save_all_output == True:
        weights_save_data_list, spikes_save_data_list, params_save_data_list = [LayerTh, Layer1, Layer2], [Layer1, LayerTh, Layer2], [Layer1, LayerTh]
        DownloadNetParams(layers=params_save_data_list, conn_dict=conn_layers, save_path=save_path, stage=stage)

    SaveWeights(layer_list=weights_save_data_list, conn_layers=conn_layers, shuffling=cx_groups_shuffling, save_path=save_state,
                stage=stage, save_all_output=ThaCo.save_all_output)
    SaveEvents(spikes_save_data_list, save_path, stage)

    Rastergram([LayerTh, Layer1], save_path, t_simulation, stage, shuffling=cx_groups_shuffling, net=['exc', 'inh'], fr=False, raster=True)
    WeightMap(Layer1, conn_layers,cx_groups_shuffling, save_path, stage + '_log', cycle, log=True)

    ThaCo.t_simulation += t_sleep

def Incremental(ThaCo):

    LayerTh = ThaCo.LayerTh
    Layer1 = ThaCo.Layer1
    Layer2 = ThaCo.Layer2
    conn_dict_layers = ThaCo.conn_dict_layers
    start_stage = ThaCo.start_stage
    stage = ThaCo.stage_id
    cycle = ThaCo.current_learning_cycle
    cx_groups_shuffling = ThaCo.trial_dict['training']['index shuffling']
    save_path = ThaCo.save_path
    load_path = ThaCo.load_path
    training_path = ThaCo.training_path
    run_test = ThaCo.classification
    lesion = ThaCo.lesion

    syn_params = ThaCo.params_default['network']['synapses']['conn_params']

    stage_name = ThaCo.stage

    print('LOAD PATH: %s' % load_path)
    print('SAVE PATH: %s' % save_path)

    if run_test != 'only':
        if load_path != '':
            weightTime = time.time()
            LoadWeights(layer_list=[LayerTh, Layer1, Layer2], conn_layers=conn_dict_layers, syn_params=syn_params,
                        load_path=load_path, stage=start_stage)
            print('\nWeights loading time: %g s' % (time.time() - weightTime))
            WeightMap(Layer1, conn_dict_layers, cx_groups_shuffling, save_path, stage + '_load', cycle, log=True)

        if stage_name == 'awake_training':

            params_old = ThaCo.params_stage.copy()
            params_new = ThaCo.config_complete[stage_name]
            ThaCo.params_stage = params_new
            ThaCo.SetNetState(params_start_state=params_old, params_new_state=params_new)
            Training(ThaCo)

            if run_test != 'none':
                params_old = ThaCo.params_stage.copy()
                params_new = ThaCo.config_complete['awake_test'].copy()
                ThaCo.params_stage = params_new
                ThaCo.SetNetState(params_start_state=params_old, params_new_state=params_new)
                Test(ThaCo)

        if stage_name == 'nrem':
            NREM(ThaCo)
            if run_test != 'none':
                Test(ThaCo)

        if stage_name == 'rem':
            REM(ThaCo)
            if run_test != 'none':
                Test(ThaCo)

    elif run_test == 'only':

        if stage_name == 'awake_training':
            weightTime = time.time()
            LoadWeights(layer_list=[LayerTh, Layer1, Layer2], conn_layers=conn_dict_layers, syn_params=syn_params,
                        load_path=load_path, stage=start_stage)
            print('\nWeights loading time: %g s' % (time.time() - weightTime))
            Test(ThaCo)

        if stage_name == 'nrem':
            weightTime = time.time()
            LoadWeights(layer_list=[LayerTh, Layer1, Layer2], conn_layers=conn_dict_layers, syn_params=syn_params,
                        load_path=load_path, stage=start_stage)
            print('\nWeights loading time: %g s' % (time.time() - weightTime))
            Test(ThaCo)

        if stage_name == 'rem':
            weightTime = time.time()
            LoadWeights(layer_list=[LayerTh, Layer1, Layer2], conn_layers=conn_dict_layers, syn_params=syn_params,
                        load_path=load_path, stage=start_stage)
            print('\nWeights loading time: %g s' % (time.time() - weightTime))
            Test(ThaCo)
