#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Jul 31 00:05:31 2020

@author: rodja
"""

import json
import nest
import numpy as np
import os
import scipy.signal as signal
from itertools import chain
from utils import WindowExp, WindowGauss, Decimate, Power
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
from plot_utils import set_ticks
from utils import KeyExists


def Create(layer, spike_rec_dict):

        print('\nCrea spike detector sul %s: %s\n' %(layer.name, json.dumps({"withgid": True, "withtime": True})))

        key_exc, key_inh = spike_rec_dict['neurons']['exc'], spike_rec_dict['neurons']['inh']
        if KeyExists(spike_rec_dict, ('noise', 'exc')):
            key_noise_exc = spike_rec_dict['noise']['exc']
        else:
            key_noise_exc = spike_rec_dict['noise']['exc_1']
            key_noise_exc_2 = spike_rec_dict['noise']['exc_2']
        key_noise_inh = spike_rec_dict['noise']['inh']

        if layer.save_all_output:
            if layer.name == 'th':
                key_exc, key_inh, key_noise_exc, key_noise_inh = True, True, True, False
            if layer.name == 'cx':
                key_exc, key_inh, key_noise_exc, key_noise_exc_2, key_noise_inh = True, True, True, True, True
            if layer.name == 'ro':
                key_exc, key_inh, key_noise_exc, key_noise_inh = True, False, True, False

        if  key_exc:
            spike_detect_exc = nest.Create("spike_recorder", layer.n_exc)
            layer.sd_exc = spike_detect_exc
            nest.Connect(layer.exc, layer.sd_exc, "one_to_one")
        else:
            layer.sd_exc = None

        if key_inh:
            spike_detect_inh = nest.Create("spike_recorder", np.sum(layer.n_inh))
            layer.sd_inh = spike_detect_inh
            nest.Connect(layer.inh, layer.sd_inh, "one_to_one")
        else:
            layer.sd_inh = None

        if key_noise_exc:
            spike_detect_noise_exc = nest.Create("spike_recorder", layer.n_exc)
            layer.sd_noise_exc = spike_detect_noise_exc
            nest.Connect(layer.noise_exc, layer.sd_noise_exc, "one_to_one")
            if layer.name == 'cx':
                if key_noise_exc_2:
                    spike_detect_noise_exc_2 = nest.Create("spike_recorder", layer.n_exc)
                    layer.sd_noise_exc_2 = spike_detect_noise_exc_2
                    nest.Connect(layer.noise_rem, layer.sd_noise_exc_2, "one_to_one")
                else:
                    layer.sd_noise_exc_2 = None
        else:
            layer.sd_noise_exc = None

        if key_noise_inh:
            spike_detect_noise_inh = nest.Create("spike_recorder", np.sum(layer.n_inh))
            layer.sd_noise_inh = spike_detect_noise_inh
            nest.Connect(layer.noise_inh, layer.sd_noise_inh, "one_to_one")
        else:
            layer.sd_noise_inh = None

def SpikesGroup(layer, test, t0, stage):

    t_img = test.t_img
    t_pause = test.t_pause
    n_groups = layer.n_groups
    n_exc_ca = layer.n_exc_ca

    for example in range(test.n_img):

        t_min = t0 + example * (t_img + t_pause)
        t_max = t_min + t_img

        print("\n\n%s Example %d  Label %d   t_min %lf   t_max %lf\n\n" % (
        layer.name, example, test.pattern_labels[example], t_min, t_max))

        evt_sd = layer.sd_exc.get("events")

        for group in range(n_groups):
            evt_group = evt_sd[n_exc_ca * group:n_exc_ca * (group + 1)]
            for neuron in range(n_exc_ca):
                t_sd = evt_group[neuron]["times"]
                t_sd = [(t_sd >= t_min) & (t_sd <= t_max)]
                layer.spikes[stage]['group']['spikes'][example][group] += np.sum(t_sd)

    layer.spikes[stage]['count'] = np.sum(np.hstack(layer.spikes[stage]['group']['spikes']))


def GroupPred(layer, train, ex, stage):

    name = layer.name
    n_class = layer.n_class
    areas = layer.areas
    n_ranks = layer.n_ranks_train
    n_groups = layer.n_groups

    spikes_groups = layer.spikes[stage]['group']['spikes'][ex]

    if name == 'cx':
        group_area = np.array([group // (n_ranks * n_class) % areas for group in range(n_groups)])
        group_output = np.sum([spikes_groups[(group_area==area)] for area in range(layer.areas)],axis=0) #spikes_groups[(group_area==0)] + spikes_groups[(group_area==1)]
    else:
        group_output = spikes_groups[:]

    group = np.argmax(group_output)
    print('\nGroup prediction', group)
    group_label = train.pattern_all_labels[group] if layer.name == 'cx' else group

    layer.spikes[stage]['group']['output'][ex] = group_label


def ClassPred(layer, train, ex, stage):

    name = layer.name
    labels = train.pattern_labels[:]
    n_class = train.n_class
    cycle = layer.cycle
    areas = layer.areas
    n_ranks = layer.n_ranks_train
    n_groups = layer.n_groups

    groups = layer.spikes[stage]['group']['spikes'][ex]

    if name == 'cx':
        if cycle>0:
            for classe in range(n_class):
                mask = (labels == classe)
                spikes_group = 0
                for cy in range(cycle):
                    if type(spikes_group)==int:

                        if areas==1:
                            spikes_group = groups[cy * n_class * n_ranks * areas : cy * n_class * n_ranks * areas + n_class * n_ranks]
                        else:
                            spikes_group = groups[cy * n_class * n_ranks * areas: cy * n_class * n_ranks * areas + n_class * n_ranks] + \
                                           groups[cy * n_class * n_ranks * areas + n_class * n_ranks: cy * n_class * n_ranks * areas + 2 * n_class * n_ranks]
                    else:
                        if areas == 1:
                            spikes_group += groups[cy * n_class * n_ranks * areas : cy * n_class * n_ranks * areas + n_class * n_ranks]
                        else:
                            spikes_group += groups[cy * n_class * n_ranks * areas: cy * n_class * n_ranks * areas + n_class * n_ranks] + \
                                            groups[cy * n_class * n_ranks * areas + n_class * n_ranks: cy * n_class * n_ranks * areas + 2 * n_class * n_ranks]
                spikes_class = np.sum(spikes_group[mask])
                layer.spikes[stage]['class']['spikes'][ex][classe] = spikes_class
        else:
            for classe in range(n_class):
                mask = (labels == classe)
                spikes_group = groups[:n_groups // 2] + groups[n_groups // 2:]
                spikes_class = np.sum(spikes_group[mask])
                layer.spikes[stage]['class']['spikes'][ex][classe] = spikes_class

    elif name == 'ro':
        for classe in range(n_class):
            layer.spikes[stage]['class']['spikes'][ex][classe] = groups[classe]

    class_label = np.argmax(layer.spikes[stage]['class']['spikes'][ex])
    layer.spikes[stage]['class']['output'][ex] = class_label


def Prediction(layer, train, test, stage):

    name = layer.name
    n_img = test.n_img

    acc_group = 0.
    acc_class = 0.

    for label, example in zip(test.pattern_labels, range(n_img)):

        GroupPred(layer,train,example,stage)
        ClassPred(layer,train,example,stage)

        true_label = label
        group_label = layer.spikes[stage]['group']['output'][example]
        class_label = layer.spikes[stage]['class']['output'][example]

        print('\n%s Groups spikes: %s\nClass spikes: %s\n' %(name,layer.spikes[stage]['group']['spikes'][example],
                                                             layer.spikes[stage]['class']['spikes'][example]))
        print('\nExample %d, Test label: %d, Group Prediction: %d, Class Prediction: %d\n' % (example,true_label,group_label,class_label))

        acc_group = acc_group + 1. if group_label==true_label else acc_group
        acc_class = acc_class + 1. if class_label == true_label else acc_class

    layer.accuracy[stage]['group'] = acc_group / n_img
    layer.accuracy[stage]['class'] = acc_class / n_img

    print('\n%s Group Accuracy: %lg\nClass Accuracy: %lg\n' % (layer.name,acc_group/n_img,acc_class/n_img))


def WPA(layer, conn_layers, stage, t0, t_simulate, save_path):

    file_name = os.path.join(save_path, 'Performance_%s.txt' % stage)
    # TO-BE-FIXED
    if not os.path.exists(file_name):
        f = open(file_name, 'w')
    else:
        f = open(file_name, 'r')
        n_lines = len(f.readlines())
        f.close()
        if n_lines<4:
            f = open(file_name, 'a')
        else:
            f = open(file_name, 'w')

    if layer.name == 'cx':

        P = Power(layer,conn_layers) * 1000. / (0.4*t_simulate)

        acc_group = layer.accuracy[stage]['group']
        acc_class = layer.accuracy[stage]['class']
        P_mean = np.mean(P)

        print('%lg' % P_mean, file=f)
        print('%lg' % acc_group, file=f)
        print('%lg' % acc_class, file=f)

    elif layer.name == 'ro':

        acc_group = layer.accuracy[stage]['group']

        print('%lg' % acc_group, file=f)

    f.close()


def Accuracy(layers, train, test, conn_layers, t0, t_simulate, stage, save_path):

    n_img = test.n_img

    for layer in layers:
        n_groups = layer.n_groups

        layer.spikes[stage] = {'group': {'spikes': np.zeros((n_img, n_groups)), 'output': np.zeros(n_img)},
                        'class': {'spikes': np.zeros((n_img, layer.n_class)), 'output': np.zeros(n_img)}, 'count': 0.}
        layer.accuracy[stage] = {'group': 0.,'class': 0.}

        print('\n%s SPIKES COUNT %s\n' % (train.name, layer.name))
        SpikesGroup(layer, test, t0, stage)
        Prediction(layer, train, test, stage)
        WPA(layer, conn_layers, stage, t0, t_simulate, save_path)


def SaveEvents(layer_list, save_path, stage):

    path = os.path.join(save_path, 'Events')
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except BaseException as err:
            raise err

    for layer in layer_list:

        print('\nSave Events %s\n' % layer.name)
        evts_exc = layer.sd_exc.get('events')
        spike_times_exc = [evts_exc[n]['times'] for n in range(layer.n_exc)]
        if layer.save_all_output == True:
            evts_noise_exc = layer.sd_noise_exc.get('events')
            spike_times_noise_exc = [evts_noise_exc[n]['times'] for n in range(layer.n_exc)]
            if layer.name == 'cx':
                evts_noise_exc_2 = layer.sd_noise_exc_2.get('events')
                spike_times_noise_exc_2 = [evts_noise_exc_2[n]['times'] for n in range(layer.n_exc)]
        else:
            spike_times_noise_exc = []
            if layer.name == 'cx': spike_times_noise_exc_2 = []

        if np.sum(layer.n_inh)>0:
            evts_inh = layer.sd_inh.get('events')
            spike_times_inh = [evts_inh[n]['times'] for n in range(np.sum(layer.n_inh))]
            if layer.save_all_output == True and layer.name=='cx':
                evts_noise_inh = layer.sd_noise_inh.get('events')
                spike_times_noise_inh = [evts_noise_inh[n]['times'] for n in range(np.sum(layer.n_inh))]
            else: spike_times_noise_inh = []
        else:
            spike_times_inh = []

        evts = {'evt_exc': spike_times_exc, 'evt_inh': spike_times_inh,
                'evt_noise_exc': spike_times_noise_exc, 'evt_noise_inh': spike_times_noise_inh}
        if layer.name == 'cx': evts['evt_noise_exc_2'] = spike_times_noise_exc_2

        try:
           np.save(os.path.join(path, '%s_%s' % (layer.name, stage)), evts)
        except BaseException as err:
           raise err


def Rastergram(lay_list, save_path, t0, stage, net=['exc'], shuffling=None, raster=True, fr=False):

    save_graph = os.path.join(save_path, 'Plot')
    if not os.path.exists(save_graph):
        try:
            os.makedirs(save_graph)
        except BaseException as err:
            raise err

    for layer in lay_list:

        if 'exc' in net:
            evt_sd_exc = nest.GetStatus(layer.sd_exc, keys="events")
            spikes_exc = [[] for ngroup in range(layer.n_groups)] if layer.name == 'cx' else []
            for n in range(layer.n_exc):
                events = evt_sd_exc[n]['times']
                mask = (events >= t0)
                times = evt_sd_exc[n]['times'][mask]
                if layer.name == 'cx':
                    group_id = n // layer.n_exc_ca
                    group_id_sh = shuffling[group_id]
                    spikes_exc[group_id_sh].append(times)
                else:
                    spikes_exc.append(times)
            if layer.name == 'cx': spikes_exc = list(chain(*spikes_exc))

            if raster == True:
                plt.figure(figsize=(50, 30))
                set_ticks(30, 2, 10, 2)
                fontsize = 60
                plt.eventplot(np.array(spikes_exc, dtype=object) / 1000., orientation='horizontal', lineoffsets=1,
                              linelengths=1, linewidths=None, colors='black')
                plt.title('Raster plot EXC Neurons: ' + layer.name, fontsize=fontsize)
                plt.ylabel('neuron ID', fontsize=fontsize)
                plt.xlabel('t [s]', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                try:
                    plt.savefig(os.path.join(save_graph, 'rp_%s_%s_exc.png' % (layer.name, stage)))
                except BaseException as err:
                    raise err

            if fr==True and len(np.hstack(spikes_exc)) > 0:

                n_exc = layer.n_exc

                x = np.sort(np.hstack(spikes_exc))

                dt = 0.1
                nu_t = 5
                si_t = 1000. / (2 * np.pi * nu_t)

                times_sim = np.arange(x[0], x[-1], dt)
                fr_istant = []

                for t in times_sim:
                    mask = (x >= t - 0.5 * dt) & (x <= t + 0.5 * dt)
                    fr_istant.append(int(np.sum(mask)))

                fr_istant = np.array(fr_istant)
                fr_alpha = signal.convolve(fr_istant, WindowExp(3), 'same')
                fr_lp = signal.convolve(fr_alpha, WindowGauss(si_t), 'same')
                lfp = Decimate(fr_lp, 10)
                lfp = signal.convolve(lfp, WindowGauss(si_t), 'same')
                times_lfp = Decimate(times_sim, 10)

                plt.figure(figsize=(50, 30))
                set_ticks(30, 2, 10, 2)
                fontsize = 60
                plt.title('Firing Rate EXC Neurons: ' + layer.name, fontsize=fontsize)
                plt.plot(times_lfp, lfp * (1000. / n_exc), lw=1, color='red')
                plt.ylabel('FR [Hz]', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                try:
                    plt.savefig(os.path.join(save_graph, 'fr_%s_%s_exc.png' % (layer.name, stage)))
                except BaseException as err:
                    raise err

        if 'inh' in net:
            evt_sd_inh = nest.GetStatus(layer.sd_inh, keys="events")
            neurons_inh = []
            spikes_inh = []
            n_inh = np.sum(layer.n_inh)
            for n in range(n_inh):
                events = evt_sd_inh[n]['times']
                mask = (events >= t0)
                times = evt_sd_inh[n]['times'][mask]
                senders = evt_sd_inh[n]['senders'][mask]
                spikes_inh.append(times)
                neurons_inh.append(senders)

            if raster == True and len(spikes_inh) > 0:
                plt.figure(figsize=(50, 30))
                set_ticks(30, 2, 10, 2)
                fontsize = 60
                plt.eventplot(np.array(spikes_inh, dtype=object) / 1000., orientation='horizontal', lineoffsets=1,
                              linelengths=1, linewidths=None, colors='black')
                plt.title('Raster plot INH Neurons: ' + layer.name, fontsize=fontsize)
                plt.ylabel('neuron ID', fontsize=fontsize)
                plt.xlabel('t [s]', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                try:
                    plt.savefig(os.path.join(save_graph, 'rp_%s_%s_inh.png' % (layer.name, stage)))
                except BaseException as err:
                    raise err

            if fr==True and len(np.hstack(spikes_inh)) > 0:

                x = np.sort(np.hstack(spikes_inh))

                dt = 0.1
                nu_t = 5
                si_t = 1000. / (2 * np.pi * nu_t)

                times_sim = np.arange(x[0], x[-1], dt)
                fr_istant = []

                for t in times_sim:
                    mask = (x >= t - 0.5 * dt) & (x <= t + 0.5 * dt)
                    fr_istant.append(int(np.sum(mask)))

                fr_istant = np.array(fr_istant)
                fr_alpha = signal.convolve(fr_istant, WindowExp(3), 'same')
                fr_lp = signal.convolve(fr_alpha, WindowGauss(si_t), 'same')
                lfp = Decimate(fr_lp, 10)
                lfp = signal.convolve(lfp, WindowGauss(si_t), 'same')
                times_lfp = Decimate(times_sim, 10)

                plt.figure(figsize=(50, 30))
                fontsize = 60
                set_ticks(30, 2, 10, 2)
                plt.title('Firing Rate INH Neurons: ' + layer.name, fontsize=fontsize)
                plt.plot(times_lfp, lfp * (1000. / n_inh), lw=1, color='red')
                plt.ylabel('FR [Hz]', fontsize=fontsize)
                plt.xticks(fontsize=fontsize)
                plt.yticks(fontsize=fontsize)

                try:
                    plt.savefig(os.path.join(save_graph, 'fr_%s_%s_inh.png' % (layer.name, stage)))
                except BaseException as err:
                    raise err
