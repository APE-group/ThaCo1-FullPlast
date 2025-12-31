#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  utils.py
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

import os.path

import numpy as np
import psutil, string
import scipy.signal as signal
from time import time, time_ns

def LoadFeatures(self):

    print('\nLoading features and labels\n')
    try:
        all_features = np.load(self.fn_feat)
    except BaseException as err:
        raise err
    try:
        all_labels = np.load(self.fn_labels)
    except BaseException as err:
        raise err

    return all_features, all_labels

def WindowExp(tau):

    window = signal.exponential(M=int(12 * tau), tau=tau, )

    window_bio = []

    for t in range(len(window)):
        window_bio.append(window[t] * t / (tau ** 2))
    return window_bio


def WindowGauss(sigma):

    window_gauss = signal.gaussian(int(2 * 5 * sigma), sigma)

    window_gauss /= np.sqrt(2 * np.pi * sigma ** 2)

    return window_gauss

def Decimate(x, r):

    y = []

    for i in range(len(x)):
        if i % r == 0:
            y.append(x[i])
    return np.array(y)

def WeightMatrix(layer, conn_layers):

    if layer.name == 'cx':
        conn = conn_layers['cx']['all']['exc_exc']
        W, S, T = conn['conn'].get('weight'), conn['source'], conn['target']
        n_S = len(np.unique(S))
        n_T = len(np.unique(T))

        # Source - Target - Pesi
        S -= np.min(S)
        T -= np.min(T)

        w = np.zeros((n_S, n_T))

        for s, t, n in zip(S, T, range(len(S))):
            w[s][t] = W[n]

        return w

def Power(layer, conn_layers):

    n_neur = layer.n_exc

    w_cx = WeightMatrix(layer,conn_layers)

    spikes_neurons = layer.sd_exc.get("events")
    P = np.zeros(n_neur)
    for neuron in range(n_neur):
        n_spikes = len(spikes_neurons[neuron]['times'])
        w = np.sum(w_cx[neuron])
        P[neuron] = w * n_spikes

    return P

def OUProcess(x0, mu, si, theta, dt, n_steps, seed=time()):

    print('\nOrnstein-Uhlenbeck Process generation')

    N_sims = len(x0)
    N_t = n_steps
    index = np.arange(0, N_t, dtype=int)

    rng = np.random.RandomState(seed)

    print('\nN process: %lg , N steps:%lg'%(N_sims,N_t))

    dW = si * rng.normal(0, np.sqrt(dt), (N_t, N_sims))
    x = np.zeros_like(dW)
    x[0] = x0
    x_min = np.zeros_like(mu)
    x_max = mu + 100*si #si

    for t in index[1:N_t]:
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) * dt + dW[t]
        inf = (x[t]<x_min)
        sup = (x[t]>x_max)
        if len(x[t][inf])>0: x[t][inf] = x_min[inf]
        if len(x[t][sup])>0: x[t][sup] = x_max[sup]


    return np.transpose(x)

def ClippedGaussian(mu, si, N, Min=None, Max=None, seed=None):

    seed = seed if type(seed) != None else time()

    rng = np.random.RandomState(seed)

    numbers = rng.normal(mu,si,N)

    print(Min,Max)


    if Min!=None and Max==None:
        low = (numbers<Min)
        n_low = np.sum(low)
        while n_low>0:
            numbers_new = rng.normal(mu,si,n_low)
            numbers[low] = numbers_new
            low = (numbers < Min)
            n_low = np.sum(low)

    if Min==None and Max!=None:
        high = (numbers>Max)
        n_high = np.sum(high)
        while n_high>0:
            numbers_new = rng.normal(mu,si,n_high)
            numbers[high] = numbers_new
            high = (numbers > Max)
            n_high = np.sum(high)

    if Min!=None and Max!=None:

        print(Min,Max)

        low = (numbers<Min)
        n_low = np.sum(low)
        while n_low>0:
            numbers_new = rng.normal(mu,si,n_low)
            numbers[low] = numbers_new
            low = (numbers < Min)
            n_low = np.sum(low)

        high = (numbers>Max)
        n_high = np.sum(high)
        while n_high>0:
            numbers_new = rng.normal(mu,si,n_high)
            numbers[high] = numbers_new
            high = (numbers > Max)
            n_high = np.sum(high)

    return numbers

def MemoryUsage():

    process = psutil.Process()
    mem_info = process.memory_info()
    return mem_info.rss

def FlattenList2List(input_list):

    flatten_list = []
    for item in input_list:
        if isinstance(item, list) or isinstance(item, np.ndarray):
            flatten_list.extend(FlattenList2List(item))
        else:
            flatten_list.append(item)

    return flatten_list

def FlattenDict2List(input_dict, fl_list=True):

    flatten_list = []
    for key, value in input_dict.items():
        if isinstance(value, dict):
            flatten_list.extend(FlattenDict2List(value))
        elif isinstance(value, list) and fl_list==True:
            flatten_list.extend(FlattenList2List(value))
        else:
            flatten_list.append(value)

    return flatten_list

def FlattenDict2Dict(input_dict):
    flatten_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            flatten_dict.update(FlattenDict2Dict(value))
        else:
            flatten_dict[key] = value
    return flatten_dict

def NestDictToDict(dic, sep = '.'):

    nested_dic = {}

    for key, value in dic.items():
        keys = key.split(sep)
        current_dict = nested_dic
        for k in keys[:-1]:
            if k not in current_dict:
                current_dict[k] = {}
            current_dict = current_dict[k]
        current_dict[keys[-1]] = value
    return nested_dic

def RemoveValueFromDict(dic, value):
    if isinstance(dic, dict):
        return {
            k: RemoveValueFromDict(v, value)
            for k, v in dic.items() if v is not value
        }
    elif isinstance(dic, list):
        return [RemoveValueFromDict(v, value) for v in dic if v is not value]
    else:
        return dic

def MergeDicts(dict1, dict2):

    for key, value in dict2.items():
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(value, dict):
                MergeDicts(dict1[key], value)
            else:
                dict1[key] = value
        else:
            dict1[key] = value

    return dict1

def KeyExists(dic, keys):

    if type(keys) == str: keys = (keys,)

    current_level = dic
    for key in keys:
        if isinstance(current_level, dict) and key in current_level:
            current_level = current_level[key]
        else:
            return False
    return True

def DictsDiff(dic1, dic2):
    diff = {}
    for key in dic2:
        if key not in dic1:
            diff[key] = dic2[key]
        else:
            if isinstance(dic2[key], dict) and isinstance(dic1[key], dict):
                nested_diff = DictsDiff(dic1[key], dic2[key])
                if nested_diff:
                    diff[key] = nested_diff
            elif dic2[key] != dic1[key]:
                diff[key] = dic2[key]
    return diff

def GetDictValue(dic, keys):
    for key in keys:
        if isinstance(dic, dict) and key in dic:
            dic = dic[key]
        else:
            return None
    return dic

def RandomString(lenght=10):

    seed = time_ns() % 2 ** 32
    rng = np.random.RandomState(seed)
    alphabet = np.array(list(string.ascii_letters + string.digits))
    random_string = ''.join(rng.choice(alphabet) for n in range(lenght))

    return random_string

def DownloadNetParams(layers, conn_dict, stage, save_path):

    neur_params_name = ['a', 'b', 'Delta_T', 'E_ex', 'E_in', 'E_L', 'g_L',
        'model', 't_ref', 'tau_minus', 'tau_syn_ex', 'C_m',
                   'tau_syn_in', 'tau_w', 'V_peak', 'V_reset', 'V_th']

    syn_params_name = ['Wmax', 'alpha', 'lambda', 'mu_minus', 'mu_plus', 'synapse_model', 'tau_plus', 'weight', 'delay']

    conn_types = {'th_cx': conn_dict['th']['all']['fwd']['conn'], 'cx_th':conn_dict['cx']['all']['bwd']['conn'],
                  'cx_ee': conn_dict['cx']['all']['exc_exc']['conn'], 'cx_ei': conn_dict['cx']['intra']['exc_inh']['conn'],
                  'cx_ie': conn_dict['cx']['intra']['inh_exc']['conn'], 'cx_ii': conn_dict['cx']['intra']['inh_inh']['conn'],
                  'th_ei': conn_dict['th']['intra']['exc_inh']['conn'], 'th_ie': conn_dict['th']['intra']['inh_exc']['conn'],
                  'noise_th_exc': conn_dict['noise']['all']['th']['exc']['conn'], 'noise_cx_exc_1': conn_dict['noise']['all']['cx']['exc_1']['conn'],
                  'noise_cx_exc_2': conn_dict['noise']['all']['cx']['exc_2']['conn'], 'noise_cx_inh': conn_dict['noise']['all']['cx']['inh']['conn']}

    params_dict = {'neurons':{}, 'synapses':{}, 'noise':{}}

    for layer in layers:

        layer_name = layer.name
        params_dict['neurons'][layer_name] = {}

        exc, inh = layer.exc, layer.inh

        exc_params, inh_params = exc.get(), inh.get()

        params_dict['neurons'][layer_name]['exc'] = {name: np.unique(exc_params[name]) for name in neur_params_name}
        params_dict['neurons'][layer_name]['inh'] = {name: np.unique(inh_params[name]) for name in neur_params_name}
        params_dict['neurons'][layer_name]['exc'].update({'n_exc': layer.n_exc})
        params_dict['neurons'][layer_name]['exc'].update({'n_inh': layer.n_inh})


    for syn_type, conn in conn_types.items():

        syn_params = {}
        params_dict['synapses'][syn_type] = {}

        if type(conn) != list:
            syn_params = conn.get()
            params_dict['synapses'][syn_type] = {name: (np.mean(syn_params[name]), len(syn_params[name])) if name in ['weight', 'delay'] else np.unique(syn_params[name]) for name in syn_params_name if name in syn_params.keys()}
        elif type(conn[0]) != list:
            n_areas = len(conn)
            syn_params = [conn[area].get() for area in range(n_areas)]
            syn_params_name_valid = [name for name in syn_params_name if name in syn_params[0].keys()]
            params_dict['synapses'][syn_type] = {name:[(np.mean(syn_params[area][name]), len(syn_params[area][name])) if name in ['weight', 'delay'] else np.unique(syn_params[area][name]) for area in range(n_areas)] for name in syn_params_name_valid}
        else:
            n_areas, n_pop_inh = len(conn[0]), len(conn)
            syn_params = [[conn[pop][area].get() for area in range(n_areas)] for pop in range(n_pop_inh)]
            syn_params_name_valid = [name for name in syn_params_name if name in syn_params[0][0].keys()]
            params_dict['synapses'][syn_type] = {name:[[(np.mean(syn_params[pop][area][name]), len(syn_params[pop][area][name])) if name in ['weight', 'delay'] else np.unique(syn_params[pop][area][name]) for pop in range(n_pop_inh)] for area in range(n_areas)] for name in syn_params_name_valid}

    np.save(os.path.join(save_path, f'net_params_{stage}.npy'), params_dict)
