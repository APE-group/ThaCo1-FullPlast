#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  Parent_Noise.py
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

import nest
import numpy as np
import json
from utils import OUProcess

class Param_Noise:

    def __init__(self, dictionary, rate, dt_noise=None):

        self.generator= 'inhomogeneous_poisson_generator'
        self.conn_rule='one_to_one'
        self.dict= dictionary
        self.rate = rate
        self.dt_noise = dt_noise


def Create(layer, noise_exc, noise_inh):

        ######################################
        #       SYNAPSES DICTIONARIES        #
        ######################################

        layer.p_noise_exc= noise_exc
        layer.p_noise_inh= noise_inh

        rate = layer.p_noise_exc.rate['groups'] if layer.name == 'cx' else layer.p_noise_exc.rate

        print('\nCrea generatore di rumore poissoniano sul %s\n' % layer.name)
        print('\nexc1: %s. Rate Mean: %.2f  Connection rule: %s\n' %(json.dumps(layer.p_noise_exc.dict), np.mean(rate), layer.p_noise_exc.conn_rule))
        layer.noise_exc = nest.Create(layer.p_noise_exc.generator, layer.n_exc)

        if layer.name == 'cx':
            print('\ninh: %s. Rate: %s. Connection rule: %s\n' %(json.dumps(layer.p_noise_inh.dict), layer.p_noise_inh.rate, layer.p_noise_inh.conn_rule))
            layer.noise_inh = nest.Create(layer.p_noise_inh.generator, np.sum(layer.n_inh))
            print('\nexc2: %s. Rate Mean: %.2f  Connection rule: %s\n' % (
            json.dumps(layer.p_noise_exc.dict), np.mean(layer.p_noise_exc.rate['groups']), layer.p_noise_exc.conn_rule))
            layer.noise_rem = nest.Create(layer.p_noise_exc.generator, layer.n_exc)


'''
####################################
        PERCEPTUAL STIMULUS
####################################
'''


def Thalamic(train, Thalamus, t_start):

    print('\nPerceptual stimulus activation\n')

    t_img = train.t_img
    t_pause = train.t_pause
    t_start = t_start + t_pause

    n_features = train.n_img
    cycle = train.current_learning_cycle
    features = train.pattern_features[(cycle-1)*n_features:cycle*n_features]
    labels = train.pattern_labels[(cycle-1)*n_features:cycle*n_features]

    n_th = Thalamus.n_exc
    rate = Thalamus.p_noise_exc.rate
    rate_times_start = np.zeros(n_th*n_features)
    rate_times_stop = np.zeros(n_th*n_features)
    rate_values_start = np.zeros(n_th*n_features)
    rate_values_stop = np.zeros(n_th*n_features)

    for i, feature in enumerate(features):

        print('example:%d class:%d\n'%(i,labels[i]))

        t_stop = t_start + t_img
        neur_noise = Thalamus.noise_exc[:]

        neur_up = (feature==1)
        times_start = rate_times_start[n_th*i:n_th*(i+1)]
        times_stop = rate_times_stop[n_th*i:n_th*(i+1)]
        values_start = rate_values_start[n_th*i:n_th*(i+1)]
        values_stop = rate_values_stop[n_th*i:n_th*(i+1)]

        times_start[neur_up] = t_start
        times_stop[neur_up] = t_stop
        values_start[neur_up] = rate
        values_stop[neur_up] = 0.

        neur_down = (feature==0)
        times_start = rate_times_start[n_th*i:n_th*(i+1)]
        times_stop = rate_times_stop[n_th*i:n_th*(i+1)]
        values_start = rate_values_start[n_th*i:n_th*(i+1)]
        values_stop = rate_values_stop[n_th*i:n_th*(i+1)]

        times_start[neur_down] = t_start
        times_stop[neur_down] = t_stop
        values_start[neur_down] = 0.
        values_stop[neur_down] = 0.

        N_up = np.sum(neur_up)
        n_id = np.arange(n_th)[neur_up]

        print("\nBITs UP %d\n" %N_up)
        print('\nThalamus EXC neurons activated\n%s\n' % (n_id))
        print('\nt_start %g t_stop %g rate %s\n' %(t_start/1000., t_stop/1000., rate))
        t_start += t_img + t_pause

    rate_times = np.array(list(zip(rate_times_start,rate_times_stop)))
    rate_values = np.array(list(zip(rate_values_start, rate_values_stop)))

    for n in range(n_th):
        neur_noise[n].set({'rate_times': np.hstack(rate_times[n::n_th]), 'rate_values': np.hstack(rate_values[n::n_th])})


def ContextExc(train, layer, t_start):

    print('\nContextual stimulus activation on %s\n' %(layer.name))

    t_img = train.t_img
    t_pause = train.t_pause
    t_start = t_start + t_pause - 0.1
    dt = t_img#layer.p_noise_exc.dt_noise
    n_steps = 1 #int(t_img / dt) if layer.name == 'cx' else 1
    areas = layer.areas
    cycle = train.current_learning_cycle
    n_img = train.n_img

    neur_noise = layer.noise_exc[:]
    n_exc = layer.n_exc
    n_exc_ca = layer.n_exc_ca
    labels = train.pattern_labels[(cycle-1)*n_img:cycle*n_img]
    n_groups = layer.n_groups

    neur_groups = np.hstack([np.ones(n_exc_ca*n_steps,dtype=object)*n for n in range(n_groups //areas)]*areas) if layer.name == 'cx' else np.hstack([np.ones(n_exc_ca*n_steps)*n for n in range(n_groups)])
    rate_neurons = layer.p_noise_exc.rate['neurons'] if layer.name=='cx' else layer.p_noise_exc.rate*np.ones(layer.n_exc)
    #if layer.name=='cx':
    #    group_activated = np.hstack([np.ones(n_exc_ca) * n for n in range(n_groups // areas)] * areas)
    #    rate_groups = np.hstack([np.ones(n_exc_ca) * layer.p_noise_exc.rate['groups'][n] for n in range(n_groups)])
    #    rate_save = []
    rate_times_start = np.zeros(n_exc*n_steps*n_img, dtype=object)
    rate_times_stop = np.zeros(n_exc*n_steps*n_img, dtype=object)
    rate_values_start = np.zeros(n_exc*n_steps*n_img, dtype=object)
    rate_values_stop = np.zeros(n_exc*n_steps*n_img, dtype=object)

    for i, label in enumerate(labels):

        print('group:%d class:%d\n'%(i, label))
        t_stop = t_start + t_img
        group = (cycle - 1) * n_img + i if layer.name == 'cx' else label

        neur_down = (neur_groups != group) | (neur_groups == group)
        times_start = rate_times_start[n_exc*n_steps*i:n_exc*n_steps*(i+1)]
        times_stop = rate_times_stop[n_exc*n_steps*i:n_exc*n_steps*(i+1)]
        values_start = rate_values_start[n_exc*n_steps*i:n_exc*n_steps*(i+1)]
        values_stop = rate_values_stop[n_exc*n_steps*i:n_exc*n_steps*(i+1)]

        t_start_down = np.hstack([np.arange(t_start,t_stop,dt) for i in range(n_exc)])
        t_stop_down = np.hstack([np.arange(t_start+dt,t_stop+dt,dt) for i in range(n_exc)])

        times_start[neur_down] = t_start_down
        times_stop[neur_down] = t_stop_down
        values_start[neur_down] = 0.
        values_stop[neur_down] = 0.

        neur_up = (neur_groups == group)
        times_start = rate_times_start[n_exc*n_steps*i:n_exc*n_steps*(i+1)]
        times_stop = rate_times_stop[n_exc*n_steps*i:n_exc*n_steps*(i+1)]
        values_start = rate_values_start[n_exc*n_steps*i:n_exc*n_steps*(i+1)]
        values_stop = rate_values_stop[n_exc*n_steps*i:n_exc*n_steps*(i+1)]

        t_start_up = np.hstack([np.arange(t_start,t_stop,dt) for i in range(n_exc_ca*areas)])
        t_stop_up = np.hstack([np.arange(t_start+dt,t_stop+dt,dt) for i in range(n_exc_ca*areas)])

        #if layer.name=='cx':
        #    x0 = rate_neurons[(group_activated == group)]
        #    mu = rate_groups[(group_activated == group)]
        #    si = sigma[(group_activated == group)]

        #    dt_ou = 0.1
        #    n_steps_ou = int(t_img / dt_ou)

        #    seed = 121121
        #    rate_t = OUProcess(x0, mu, si, theta, dt=dt_ou, n_steps=n_steps_ou, seed=seed)
        #    rate_save.append(rate_t)

        #    rate_t = rate_t[:,::int(dt/dt_ou)]
        #    rate_stop = np.copy(rate_t)
        #    rate_stop[:, -1] = 0
        #    rate_t = rate_t.flatten()
        #    rate_stop = rate_stop.flatten()

        times_start[neur_up] = t_start_up
        times_stop[neur_up] = t_stop_up
        values_start[neur_up] = rate_neurons[neur_up] #rate_t if layer.name=='cx' else rate_neurons[neur_up]
        values_stop[neur_up] = 0 #rate_stop if layer.name=='cx' else 0.#0

        N_up = np.sum(neur_up) // n_steps
        n_id = np.arange((cycle-1)*n_exc,cycle*n_exc)

        print("\nNeurons UP %d\n" %N_up)
        print('\n%s EXC neurons activated\n%s\n' % (layer.name, n_id[neur_up]))
        print('\nt_start %g t_stop %g rate %s\n' %(t_start/1000., t_stop/1000., rate_neurons[neur_up]))
        t_start += t_img + t_pause

    rate_times = np.array(list(zip(rate_times_start,rate_times_stop)))
    rate_values = np.array(list(zip(rate_values_start, rate_values_stop)))

    rate_times = np.reshape(rate_times,(int(len(rate_times)/n_steps),n_steps,2))
    rate_values = np.reshape(rate_values,(int(len(rate_values)/n_steps),n_steps,2))

    for n in range(n_exc):
        times = rate_times[n::n_exc].flatten()
        rates = rate_values[n::n_exc].flatten()
        diff_time = np.diff(times)
        mask = (diff_time==0)
        mask = np.insert(mask,0,False)
        times[mask] = np.round(times[mask] + 0.1,2)
        times = np.round(times + 0.1,2)
        neur_noise[n].set({'rate_times':times , 'rate_values':rates})

def ContextInh(train, layer, t_start):

    print('\nAspecific stimulus activation on %s\n' %(layer.name))

    n_inh = np.sum(layer.n_inh)
    n_img = train.n_img

    t_img = train.t_img
    t_pause = train.t_pause

    rates = layer.p_noise_inh.rate
    rate_img, rate_pause = rates['example'], rates['pause']

    rate_times = [[] for i in range(n_inh)]
    rate_values = [[] for i in range(n_inh)]

    if layer.areas==1:
        noise_neurons = layer.neur_area_noise_inh[0]
    else:
        noise_neurons = layer.neur_area_noise_inh[0] + layer.neur_area_noise_inh[1]

    #First Example
    t0 = t_start
    t1 = t0 + t_pause
    t2 = t1 + t_img
    t3 = t2 + t_pause - 0.1
    for neur in range(n_inh):
        rate_times[neur] += [t0, t1, t2, t3]
        rate_values[neur] += [rate_pause, rate_img, rate_pause, 0]
    t_start += t_pause + t_img + t_pause

    print(f'\nt0 {t0/1000} - t1 {t1/1000} - t2 {t2/1000} - t3 {t3/1000}\nrate pause {rate_pause} - rate img {rate_img} - rate pause {rate_pause} - rate final {0} - \n')

    for img in range(1, n_img):
        t0 = t_start
        t1 = t0 + t_img
        t2 = t1 + t_pause - 0.1

        for neur in range(n_inh):
            rate_times[neur] += [t0, t1, t2]
            rate_values[neur] += [rate_img, rate_pause, 0]

        t_start += t_img + t_pause

        print('\nt0 %g - t1 %g - t2 %g\nrate img %g - rate pause %g - rate final %g - \n' %(t0/1000, t1/1000, t2/1000, rate_img, rate_pause, 0))


    for n in range(n_inh):
        noise_neurons[n].set({'rate_times': rate_times[n], 'rate_values': rate_values[n]})

def Aspecific(noise, rate, t_start, t_stimulus):

    print('\nAspecific stimulus activation on cx\n')

    t_stop = t_start + t_stimulus

    noise.set({'rate_times': [t_start, t_stop], 'rate_values': [rate, 0.]})

    rate_times = nest.GetStatus(noise,'rate_times')
    rate_values = nest.GetStatus(noise, 'rate_values')

    print('\nNeurons UP %d\n' % (len(noise)))
    #print('\nNeurons activated  \n%s\n%s\n' %(rate_times,rate_values))
    print('\nt_start %g t_stop %g rate %g\n' % (t_start / 1000., t_stop / 1000., rate))
