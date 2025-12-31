#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 30 23:36:20 2020

@author: rodja
"""

import nest
from Parent_Connections import ConnectEE, ConnectEI, ConnectII, ConnectNoise
import Parent_Noise as Noise
import Parent_SpikeDetector as Spike_Detect
import numpy as np


class Layer(object):


    def __init__(self, name, params_layer, params_input, params_sim):

        '''
        ############## NEURONS ##############
        '''

        self.name = name
        self.n_class = params_input['n_class']
        self.areas = params_layer['areas']
        self.n_ranks_train = params_input['n_ranks_train']
        self.n_ranks_test = params_input['n_ranks_test']
        self.n_exc_ca = params_layer['n_exc_ca']
        self.n_exc = params_layer['n_exc']
        self.n_inh = params_layer['n_inh'] * self.areas
        self.n_pop_inh = params_layer['n_pop_inh']
        self.n_groups = (self.n_exc // self.n_exc_ca)
        self.n_example_training = self.n_ranks_train * self.n_class
        self.save_all_output = params_sim['save_all_output']

        #SPIKE COUNT
        self.spikes = {}
        self.accuracy = {}


    def Create(self, neur_dict, par_conn_exc, par_conn_inh, noise_exc, noise_inh, spike_rec_dict):

        self.par_conn_exc = par_conn_exc

        neur_dict_exc = neur_dict['exc']
        neur_type_exc = neur_dict_exc.pop('model')

        #LAYER CREATION
        print('\nCREATING LAYER\n')
        print(f'\nCREATING {self.n_exc} EXC NEURONS: {neur_type_exc}, {neur_dict_exc}\n')
        self.exc = nest.Create(neur_type_exc, self.n_exc)
        nest.SetStatus(self.exc, neur_dict_exc)

        if np.sum(self.n_inh)>0:
            neur_dict_inh = neur_dict['inh']
            neur_type_inh = neur_dict_inh.pop('model')
            print(f'\nCREATING {self.n_inh} INH NEURONS: {neur_type_inh}, {neur_dict_inh}\n')
            self.par_conn_inh = par_conn_inh
            self.inh = nest.Create(neur_type_inh, np.sum(self.n_inh))
            nest.SetStatus(self.inh, neur_dict_inh)

        #LAYER CONNECTIONS
        if self.name == 'th':
            print(f'\nCREATING INTRA-LAYER CONNECTIONS: {self.name}\n')
            self.IntraNeurons()
            ConnectEI(self)

        if self.name == 'cx':
            print(f'\nCREATING INTRA-LAYER CONNECTIONS: {self.name}\n')
            self.IntraNeurons()
            ConnectEE(self)
            ConnectII(self)
            ConnectEI(self)

        if self.name == 'ro':
            print('\nCREATING INTRA-LAYER CONNECTIONS: %s\n' %(self.name))
            self.IntraNeurons()
            ConnectEE(self)

        #NOISE
        Noise.Create(self, noise_exc, noise_inh)
        self.NoiseNeurons()
        ConnectNoise(self)

        #DEVICES
        Spike_Detect.Create(self, spike_rec_dict)


    def IntraNeurons(self):

        areas = self.areas
        n_exc = self.n_exc
        n_inh = np.sum(self.n_inh)
        n_pop_inh = self.n_pop_inh
        n_class = self.n_class
        n_exc_ca = self.n_exc_ca
        n_ranks = self.n_ranks_train

        if self.name == 'cx':

            area_exc = np.array([n // n_exc_ca // (n_ranks * n_class) % areas for n in range(n_exc)])
            self.neur_area_exc = [self.exc[(area_exc==area)] for area in range(areas)]
            area_inh = np.array([areas * n // n_inh for n in range(n_inh)])
            self.neur_area_inh = [self.inh[(area_inh == area)] for area in range(areas)]
            self.neur_pop_inh = [[self.neur_area_inh[area][pop * self.n_inh[pop-1]//areas:(pop+1) * self.n_inh[pop]//areas] for area in range(areas)] for pop in range(n_pop_inh)]

        else:

            area_exc = np.array([areas * n // n_exc for n in range(n_exc)])
            self.neur_area_exc = [self.exc[(area_exc == area)] for area in range(areas)]

            if self.n_inh>0:
                area_inh = np.array([areas * n // n_inh for n in range(n_inh)])
                self.neur_area_inh = [self.inh[(area_inh == area)] for area in range(areas)]
                self.neur_pop_inh = [self.neur_area_inh]


    def NoiseNeurons(self):

        areas = self.areas
        n_exc = self.n_exc
        n_inh = np.sum(self.n_inh)
        n_pop_inh = self.n_pop_inh
        n_class = self.n_class
        n_exc_ca = self.n_exc_ca
        n_ranks = self.n_ranks_train


        if self.name == 'cx':

            area_noise_exc1 = np.array([n // n_exc_ca // (n_ranks * n_class) % areas for n in range(n_exc)])
            self.neur_area_noise_exc1 = [self.noise_exc[(area_noise_exc1 == area)] for area in range(areas)]

            area_noise_exc2 = np.array([n // n_exc_ca // (n_ranks * n_class) % areas for n in range(n_exc)])
            self.neur_area_noise_exc2 = [self.noise_rem[(area_noise_exc2 == area)] for area in range(areas)]

            area_noise_inh = np.array([areas * n // n_inh for n in range(n_inh)])
            self.neur_area_noise_inh = [self.noise_inh[(area_noise_inh == area)] for area in range(areas)]
            self.neur_pop_noise_inh = [[self.neur_area_noise_inh[area][pop * self.n_inh[pop-1]//areas:(pop+1) * self.n_inh[pop]//areas] for area in range(areas)] for pop in range(n_pop_inh)]

        else:
            area_noise_exc1 = np.array([areas * n // n_exc for n in range(n_exc)])
            self.neur_area_noise_exc1 = [self.noise_exc[(area_noise_exc1 == area)] for area in range(areas)]
