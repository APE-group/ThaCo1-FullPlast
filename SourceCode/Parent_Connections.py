#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 30 23:49:44 2020

@author: rodja
"""

import numpy as np
import nest
import json
import sys
import re
import copy
import os
import matplotlib as mpl
import matplotlib.colors as color
import matplotlib.pyplot as plt
mpl.use('Agg')
from plot_utils import set_ticks
from utils import GetDictValue, FlattenList2List
from operator import itemgetter


class SynapsesParameters:
    def __init__(self, conn_rule_2same, syn_dict_neur_2same,
                 conn_rule_2other, syn_dict_neur_2other,
                 conn_rule_fw_lay, syn_dict_fw_lay,
                 conn_rule_bw_lay, syn_dict_bw_lay,
                 conn_rule_Ro_lay, syn_dict_Ro_lay,
                 conn_rule_inter_area, syn_dict_inter_area):
        self.conn_rule_2same= conn_rule_2same
        self.dict_neur_2same= syn_dict_neur_2same
        self.conn_rule_2other=conn_rule_2other
        self.dict_neur_2other = syn_dict_neur_2other

        self.conn_rule_fw_lay=conn_rule_fw_lay
        self.dict_fw_lay=syn_dict_fw_lay

        self.conn_rule_bw_lay=conn_rule_bw_lay
        self.dict_bw_lay=syn_dict_bw_lay

        self.conn_rule_Ro=conn_rule_Ro_lay
        self.dict_Ro=syn_dict_Ro_lay

        self.conn_rule_inter_area = conn_rule_inter_area
        self.dict_inter_area = syn_dict_inter_area


def Distribution_Dict_Reader(feature_dict):
    features = ['weight','delay']
    new_dict = feature_dict.copy()
    for feature in features:
        dict = feature_dict[feature]
        if type(dict) == type({}):
            name = dict['distribution']
            print('feature %s dist %s'%(feature,name))
            if name == 'exponential_clipped':
                new_dict[feature] = nest.math.min((dict['low'] + nest.random.exponential(beta=dict['beta'])), dict['high'])
            if name == 'uniform':
                new_dict[feature] =  nest.random.uniform(min=dict['low'], max=dict['high'])

    return new_dict


#EXC - EXC INTRA AREA CONNECTIONS
def ConnectEE(Layer):

    print('\nCREATE EXC - EXC INTRA-AREA CONNECTIONS: %s - %s\n' %(Layer.name, Layer.name))

    areas = Layer.areas

    conn_rule_exc_exc = Layer.par_conn_exc.conn_rule_2same
    conn_dict_exc_exc = Layer.par_conn_exc.dict_neur_2same

    for area in range(areas):
        print('\nArea %d\n' % (area))
        neur_area = Layer.neur_area_exc[area]

        print('\nNEST EXC Neurons ID: %s'%(neur_area))
        print('\nConnections Rule %s\nConnections Paramters: %s\n' %(conn_rule_exc_exc,conn_dict_exc_exc))
        new_dict = Distribution_Dict_Reader(conn_dict_exc_exc)

        print(neur_area)

        nest.Connect(neur_area,neur_area,conn_rule_exc_exc, new_dict)


#INH - INH INTRA AREA CONNECTIONS
def ConnectII(Layer):

    areas = Layer.areas
    n_pop_inh = Layer.n_pop_inh

    conn_rule_inh_inh = Layer.par_conn_inh.conn_rule_2same
    conn_dict_inh_inh = Layer.par_conn_inh.dict_neur_2same

    print('\nCREATING INH - INH INTRA-AREA CONNECTIONS: %s - %s\n' %(Layer.name,Layer.name))

    for area in range(areas):
        for pop in range(n_pop_inh):
            print('\nArea %d , Population %d\n' % (area,pop))
            print(Layer.neur_pop_inh)
            neur_area = Layer.neur_pop_inh[pop][area]

            print('\nNEST INH Neurons ID: %s'%(neur_area))
            print('\nINH%d - INH%d\nConnections Rule %s\nConnections Paramters: %s\n' %(pop,pop,conn_rule_inh_inh,conn_dict_inh_inh[pop]))
            nest.Connect(neur_area,neur_area,conn_rule_inh_inh, conn_dict_inh_inh[pop])


#EXC - INH INTRA AREA CONNECTIONS
def ConnectEI(Layer):

    print('\nCREATING EXC - INH INTRA-AREA CONNECTIONS: %s - %s\n' % (Layer.name, Layer.name))

    areas = Layer.areas
    n_pop_inh = Layer.n_pop_inh

    conn_rule_exc_inh = Layer.par_conn_exc.conn_rule_2other
    conn_dict_exc_inh = Layer.par_conn_exc.dict_neur_2other

    conn_rule_inh_exc = Layer.par_conn_inh.conn_rule_2other
    conn_dict_inh_exc = Layer.par_conn_inh.dict_neur_2other

    for area in range(areas):
        for pop in range(n_pop_inh):

            print('\nArea %d , Population %d\n' % (area,pop))
            neur_area_exc = Layer.neur_area_exc[area]
            neur_area_inh = Layer.neur_pop_inh[pop][area]

            print('\nNEST EXC Neurons ID: %s'%(neur_area_exc))
            print('\nNEST INH Neurons ID: %s'%(neur_area_inh))

            print('\nEXC - INH%d\nConnections Rule %s\nConnections Paramters: %s\n' %(pop,conn_rule_exc_inh,conn_dict_exc_inh[pop]))
            nest.Connect(neur_area_exc,neur_area_inh,conn_rule_exc_inh, conn_dict_exc_inh[pop])

            print('\nINH%d - EXC\nConnections Rule %s\nConnections Paramters: %s\n' %(pop,conn_rule_inh_exc,conn_dict_inh_exc[pop]))
            nest.Connect(neur_area_inh,neur_area_exc,conn_rule_inh_exc, conn_dict_inh_exc[pop])


#NOISE --> LAYER CONNECTIONS
def ConnectNoise(Layer):

    print('\nCREATING NOISE - CORTEX INTRA-AREA CONNECTIONS: %s - %s\n' % (Layer.name, Layer.name))

    areas = Layer.areas
    n_pop_inh = Layer.n_pop_inh

    conn_rule_noise_exc = Layer.p_noise_exc.conn_rule
    conn_dict_noise_exc = Layer.p_noise_exc.dict

    for area in range(areas):
        print('\nArea %d\n' % (area))
        neur_area = Layer.neur_area_exc[area]
        noise_area = Layer.neur_area_noise_exc1[area]

        print('\nNEST EXC Neurons ID: %s' % (neur_area))
        print('\nNEST NOISE 1 Neurons ID: %s' % (noise_area))

        print('\nNOISE - EXC\nConnections Rule %s\nConnections Paramters: %s\n' % (conn_rule_noise_exc,conn_dict_noise_exc))

        nest.Connect(noise_area, neur_area, conn_rule_noise_exc, conn_dict_noise_exc)

    if Layer.name == 'cx':

        conn_rule_noise_inh = Layer.p_noise_inh.conn_rule
        conn_dict_noise_inh = Layer.p_noise_inh.dict

        for area in range(areas):
            print('\nArea %d\n' % (area))

            neur_area_exc = Layer.neur_area_exc[area]
            noise_area_exc2 = Layer.neur_area_noise_exc2[area]

            print('\nNEST NOISE EXC Neurons ID: %s' % (noise_area_exc2))
            print('\nNEST EXC Neurons ID: %s' % (neur_area_exc))
            print('\nNOISE2 - EXC\nConnections Rule %s\nConnections Paramters: %s\n' % (conn_rule_noise_exc, conn_dict_noise_exc))

            nest.Connect(noise_area_exc2, neur_area_exc, conn_rule_noise_exc, conn_dict_noise_exc)

            for pop in range(n_pop_inh):
                print('\nPopulation %d\n' % (pop))
                neur_pop_inh = Layer.neur_pop_inh[pop][area]
                noise_pop_inh = Layer.neur_pop_noise_inh[pop][area]

                print('\nNEST NOISE INH Neurons ID: %s' % (noise_pop_inh))
                print('\nNEST INH Neurons ID: %s' % (neur_pop_inh))
                print('\nNOISE - INH%d\nConnections Rule %s\nConnections Paramters: %s\n' % (pop,conn_rule_noise_inh, conn_dict_noise_inh))

                nest.Connect(noise_pop_inh, neur_pop_inh, conn_rule_noise_inh, conn_dict_noise_inh)


#INTER LAYER FORWARD CONNECTIONS
def ConnectFW(set_conn_list):

    for layers in set_conn_list:

        source=layers[0]
        target=layers[1]
        source_name = source.name
        target_name = target.name
        #SYNAPSES DICTIONARIES

        #CX TO RO
        #if target_name == 'ro': #re.match('Readout', target_name):
        #    conn_rule=source.par_conn_exc.conn_rule_Ro
        #    conn_dict=copy.deepcopy(source.par_conn_exc.dict_Ro)

        ##TH TO RO
        #elif target_name == 'cx': #re.match('Layer', target_name):
        #    conn_rule=source.par_conn_exc.conn_rule_fw_lay
        #    conn_dict=copy.deepcopy(source.par_conn_exc.dict_fw_lay)
        #else:
        #    print('\nImpossibile creare la connessione inter layer')
        #    sys.exit(1)

        conn_rule = source.par_conn_exc.conn_rule_fw_lay
        conn_dict=copy.deepcopy(source.par_conn_exc.dict_fw_lay)

        print('\nFW INTER-LAYER CONNECTIONS %s - %s\n' %(source_name, target_name))

        #SYNAPSES CREATION
        for area in range(source.areas):

            neur_area_source = source.neur_area_exc[area]
            neur_area_target = target.neur_area_exc[area] if source.areas==target.areas else target.neur_area_exc[0]

            print('\nNEST SOURCE EXC Neurons ID area %d: %s' % (area, neur_area_source))
            print('\nNEST TARGET EXC Neurons ID area %d: %s' % (area, neur_area_target))
            print('\n%s ---> %s intra area %d: \n%s, %s' % (source_name,target_name,area, conn_rule, conn_dict))
            nest.Connect(neur_area_source,neur_area_target, conn_rule, conn_dict)


#INTER LAYER BACKWARD CONNECTIONS
def ConnectBW(set_conn_list):

    for layers in set_conn_list:

        source=layers[0]
        target=layers[1]
        source_name = source.name
        target_name = target.name

        #SYNAPSES DICTIONARIES
        conn_rule=source.par_conn_exc.conn_rule_bw_lay
        conn_dict=copy.deepcopy(source.par_conn_exc.dict_bw_lay)

        print('\nCREATE CONNECTIONS BACKWARD: %s - %s\n' %(source.name,target.name))

        #SYNAPSES CREATION
        for area in range(target.areas):

            neur_area_source = source.neur_area_exc[area]
            neur_area_target = target.neur_area_exc[area]

            print('\nNEST SOURCE EXC Neurons ID area %d: %s' % (area, neur_area_source))
            print('\nNEST TARGET EXC Neurons ID area %d: %s' % (area, neur_area_target))
            print('\n%s ---> %s intra area %d: \n%s, %s' % (source_name,target_name,area, conn_rule, conn_dict))
            nest.Connect(neur_area_source,neur_area_target,conn_rule, conn_dict)


#INTER AREA CONNECTIONS
def ConnectInter(set_conn_list):

    for layer in set_conn_list:

        conn_rule=layer.par_conn_exc.conn_rule_inter_area
        conn_dict=copy.deepcopy(layer.par_conn_exc.dict_inter_area)
        new_dict = Distribution_Dict_Reader(conn_dict)
        print('dict_exc_inter: %s\n'%(type(new_dict)),new_dict)

        print('\nCreate Connections inter Area: %s' %(layer.name))

        neur_area1 = layer.neur_area_exc[0]
        neur_area2 = layer.neur_area_exc[1]

        print('\nNEST EXC Neurons ID area %d: %s'%(1, neur_area1))
        print('\nNEST EXC Neurons ID area %d: %s'%(2, neur_area2))

        print('\nCreate connections exc - exc inter area 1 - 2: \n%s, %s' %(new_dict,conn_rule))
        nest.Connect(neur_area1,neur_area2,conn_rule,new_dict)

        print('\nCreate connections exc - exc inter area 2 - 1: \n%s, %s' %(new_dict,conn_rule))
        nest.Connect(neur_area2,neur_area1,conn_rule,new_dict)


def SynapticDynamics(lay_list, key, conn_layers, lesion=None):
    for layer in lay_list:

        source = layer[0]
        target = layer[1]
        source_name = source.name
        target_name = target.name

        #SYNAPSES DICTIONARIES

        #INTRA LAYER   (EXC - EXC   &   RO - RO)
        if key == 'On' and source.name == target.name:
            set_conn_dict = {'lambda': source.par_conn_exc.dict_neur_2same.get('lambda')}

        #INTER LAYER
        elif key == 'On' and source.name != target.name:

            # CX - RO
            if target.name == 'ro':
                print('Layer', target.name)
                set_conn_dict = {'lambda': source.par_conn_exc.dict_fw_lay.get('lambda')}

            # TH - CX   &   RO - CX
            elif target.name == 'cx':
                print('Layer', target.name)
                set_conn_dict = {'lambda': source.par_conn_exc.dict_fw_lay.get('lambda')}

            elif target.name == 'th':
                print('Layer', target.name)
                set_conn_dict = {'lambda': source.par_conn_exc.dict_bw_lay.get('lambda')}

            else:
                print('\nImpossibile settare la connessione inter layer\n', source.name, target.name)
                sys.exit(1)

        #STDP OFF
        elif key == 'Off':
            conn_dict = {'lambda': 0.0}
        else:
            print('\nKey sbagliata nella funzione Set_Conn_Dynamics\n')
            sys.exit(1)

        #SET PARAMETERS
        print('\nSET CONNECTION DYNAMICS: %s - %s:\n %s\n' % (source.name, target.name, json.dumps(conn_dict)))

        if source_name == 'th':
            conn = conn_layers['conn_th_l1_all']['conn']
            conn.set(conn_dict)

        elif source_name == 'cx':

            if target_name == 'th':
                conn = conn_layers['conn_l1_th_all']['conn']
                conn.set(conn_dict)

            if target_name == 'cx':
                conn = conn_layers['conn_l1_l1_all']['conn']
                conn.set(conn_dict)
                if lesion != 'none':
                    Layer1 = source
                    neur_exc = tuple([Layer1.neur_area_exc[area] for area in range(Layer1.areas)])

                    if lesion == 'intra':
                        if 'conn_l1_l1_intra' not in conn_layers.keys():
                            conn_lesion = tuple([neur_exc[area] for area in range(Layer1.areas)])
                        else:
                            conn_lesion = conn_layers['conn_l1_l1_intra']
                        for area in range(Layer1.areas): conn_lesion[area].set({'weight': 0.})
                    elif lesion == 'inter':
                        conn_lesion = (nest.GetConnections(neur_exc[0], neur_exc[1]), nest.GetConnections(neur_exc[1], neur_exc[0]))
                        for area in range(Layer1.areas): conn_lesion[area].set({'weight': 0.})
                    elif lesion == 'both':
                        conn.set({'weight': 0.})

            if target_name == 'ro':
                conn = conn_layers['conn_l1_l2_all']['conn']
                conn.set(conn_dict)

        elif source_name == 'ro':
            conn = conn_layers['conn_l2_l2_all']['conn']
            conn.set(conn_dict)


def SynapticDynamicsNREM(layer1, layerTh, conn_dict, conn_layers):
    if layer1.name != 'cx':
        print('ERRORE! Layer sbagliato')
        sys.exit(0)

    print('\nSetting NREM Connection Dynamics\n')
    print(conn_dict)

    for ar_sou in range(layer1.areas):

        conn_noise_high = conn_layers['conn_noise_l1'][ar_sou]
        conn_noise_low = conn_layers['conn_noise_l1_rem'][ar_sou]
        conn_intra = conn_layers['conn_l1_l1_intra'][ar_sou]
        if layer1.areas>1: conn_inter = conn_layers['conn_l1_l1_inter'][ar_sou]
        conn_inh_exc = conn_layers['conn_l1_l1_inh_exc']['conn']
        conn_th_l1 = conn_layers['conn_th_l1'][ar_sou]
        conn_l1_th = conn_layers['conn_l1_th'][ar_sou]
        conn_th_inh_exc = conn_layers['conn_th_inh_exc']['conn'][ar_sou]

        neur_exc_area = layer1.neur_area_exc[ar_sou]
        neur_inh_area = layer1.neur_area_inh[ar_sou]
        neur_th_area = layerTh.neur_area_exc[ar_sou]

        #RELAX
        print('\nSet Status Layer1 intra-area %d' %(ar_sou))
        print('\ntau_w: %lg , b: %lg, a:%lg' %(conn_dict.get('tau_w_nrem'), conn_dict.get('b_nrem'),conn_dict.get('a_nrem')))
        nest.SetStatus(neur_exc_area,{'tau_w':conn_dict.get('tau_w_nrem'),'b':conn_dict.get('b_nrem'),'a':conn_dict.get('a_nrem')})
        nest.SetStatus(neur_exc_area, {'V_m': nest.random.uniform(min=-71.2,max=-70)})
        nest.SetStatus(neur_inh_area, {'V_m': nest.random.uniform(min=-71.2,max=-70)})
        nest.SetStatus(neur_th_area,{'V_m': nest.random.uniform(min=-71.2,max=-70)})

        print('\nSet Status: intra-area %d noise - l1 connections' % (ar_sou))
        print('\nW NOISE HIGH - EXC: %lg' % (conn_dict.get('W_noise_high_nrem')))
        conn_noise_high.set({'weight': conn_dict.get('W_noise_high_nrem')})

        print('\nW NOISE LOW - EXC: %lg' % (conn_dict.get('W_noise_low_nrem')))
        conn_noise_low.set({'weight': conn_dict.get('W_noise_low_nrem')})

        # CX (EXC INTRA)
        print('\nSet Connection Dynamics: Layer1 intra-area %d exc - exc' % (ar_sou))
        print('\nlambda intra: %lg , alpha intra: %lg , W new intra: %lg , ratio exc intra: %lg'
              % (conn_dict.get('lambda_intra_nrem'), conn_dict.get('alpha_nrem'), conn_dict.get('W_new_nrem'),
                 conn_dict.get('ratio_intra_nrem')[ar_sou]))
        Wmax_intra = np.array(conn_intra.get('Wmax')) * conn_dict.get('ratio_intra_nrem')[ar_sou]
        w_exc_exc_intra = np.array(conn_intra.get('weight')) * conn_dict.get('ratio_intra_nrem')[ar_sou]
        conn_intra.set({'lambda': conn_dict.get('lambda_intra_nrem'), 'alpha': conn_dict.get('alpha_nrem'),
                        'Wmax':Wmax_intra, 'weight': w_exc_exc_intra})

        if layer1.areas>1:
            # CX (EXC INTER)
            n1 = 1 if ar_sou == 0 else 2
            n2 = 2 if ar_sou == 0 else 1
            print('\nSet Connection Dynamics: %s Layer1 inter-area %d ---> %d\n' % (layer1.name, n1, n2))
            print('\nlambda inter: %lg , ratio inter: %lg' % (
            conn_dict.get('lambda_inter_nrem'), conn_dict.get('ratio_inter_nrem')[ar_sou]))
            w_exc_exc_inter = np.array(conn_inter.get('weight')) * conn_dict.get('ratio_inter_nrem')[ar_sou]
            conn_inter.set({'lambda': conn_dict.get('lambda_inter_nrem'), 'weight': w_exc_exc_inter})

        # CX (INH - EXC)
        print('\nSet Status: Layer1  intra-area %d' % (ar_sou))
        print('\nW INH1-EXC: %lg\nW INH2-EXC: %lg\n' % (conn_dict.get('W_inh_exc_1_nrem'),conn_dict.get('W_inh_exc_2_nrem')))
        for pop in range(layer1.n_pop_inh):
            label = f'W_inh_exc_{pop+1}_nrem'
            conn = conn_inh_exc[pop][ar_sou]
            weight = conn_dict.get(label)
            conn.set({'weight': weight})

        # TH - CX
        print('\nSet Status: intra-area %d th - l1 ' % (ar_sou))
        print('\nratio TH - CX: %lg' % (conn_dict.get('ratio_th_l1_nrem')))
        w_th_l1_intra = np.array(conn_th_l1.get('weight')) * conn_dict.get('ratio_th_l1_nrem')
        Wmax = np.array(conn_th_l1.get('Wmax')) * conn_dict.get('ratio_th_l1_nrem')
        conn_th_l1.set({'Wmax':Wmax,'weight': w_th_l1_intra,'lambda': 0})

        # CX - TH
        print('\nSet Status: intra-area %d l1 - th ' % (ar_sou))
        print('\nratio CX - TH: %lg , lambda: 0' % (conn_dict.get('ratio_l1_th_nrem')))
        w_l1_th_intra = np.array(conn_l1_th.get('weight')) * conn_dict.get('ratio_l1_th_nrem')
        Wmax = np.array(conn_l1_th.get('Wmax')) * conn_dict.get('ratio_l1_th_nrem')
        conn_l1_th.set({'Wmax':Wmax,'weight': w_l1_th_intra,'lambda':0.})

        #TH (INH - EXC)
        print('\nSet Status: intra-area %d th - th ' %(ar_sou))
        print('\nWeight INH - EXC: %lg' %(conn_dict.get('W_th_inh_exc_nrem')))
        conn_th_inh_exc.set({'weight': conn_dict.get('W_th_inh_exc_nrem')})


def SynapticDynamicsREM(layer1, layerTh, conn_dict,conn_layers):

    if layer1.name != 'cx':
        print('ERRORE! Layer sbagliato')
        sys.exit(0)

    print('\nSetting REM Connection Dynamics\n')
    print(conn_dict)

    for ar_sou in range(layer1.areas):

        conn_noise_high = conn_layers['conn_noise_l1'][ar_sou]
        conn_noise_low = conn_layers['conn_noise_l1_rem'][ar_sou]
        conn_intra = conn_layers['conn_l1_l1_intra'][ar_sou]
        conn_inter = conn_layers['conn_l1_l1_inter'][ar_sou]
        conn_inh_exc = [conn_layers['conn_l1_l1_inh_exc']['conn'][pop][ar_sou] for pop in range(layer1.n_pop_inh)]
        conn_th_inh_exc = conn_layers['conn_th_inh_exc']['conn'][ar_sou]
        conn_th_l1 = conn_layers['conn_th_l1'][ar_sou]
        conn_l1_th = conn_layers['conn_l1_th'][ar_sou]

        neur_exc_area = layer1.neur_area_exc[ar_sou]
        neur_th_area = layerTh.neur_area_exc[ar_sou]
        neur_inh_area = layer1.neur_area_inh[ar_sou]

        #RELAX
        print('\nSet Status Layer1 intra-area %d' %(ar_sou))
        print('\ntau_w: %lg , b: %lg' %(conn_dict.get('tau_w_rem'), conn_dict.get('b_rem')))
        nest.SetStatus(neur_exc_area,{'tau_w':conn_dict.get('tau_w_rem'),'b':conn_dict.get('b_rem')})
        nest.SetStatus(neur_exc_area, {'V_m': -71.2})
        nest.SetStatus(neur_inh_area, {'V_m': -71.2})
        nest.SetStatus(neur_th_area,{'V_m':-71.2,'tau_w':conn_dict.get('tau_w_rem')})

        #NOISE (CX)
        print('\nSet Status: intra-area %d HIGH FREQUENCY noise - l1' %(ar_sou))
        print('\nW HIGH-NOISE - EXC: %lg' %(conn_dict.get('W_noise_high_rem')))
        conn_noise_high.set({'weight':conn_dict.get('W_noise_high_rem')})

        print('\nSet Status: intra-area %d LOW FREQUENCY noise - l1' %(ar_sou))
        print('\nW LOW-NOISE - EXC: %lg' %(conn_dict.get('W_noise_low_rem')))
        conn_noise_low.set({'weight':conn_dict.get('W_noise_low_rem')})

        #CX (EXC INTRA)
        print('\nSet Connection Dynamics: Layer1 intra-area %d exc - exc' %(ar_sou))
        print('\nalpha intra: %lg , lambda intra: %lg, ratio intra: %s, tau_plus: %lg'
              %(conn_dict.get('alpha_rem'),conn_dict.get('lambda_intra_rem'),conn_dict.get('ratio_intra_rem')[ar_sou],
                conn_dict.get('tau_plus_rem')))
        Wmax_intra = np.array(conn_intra.get('Wmax')) * conn_dict.get('ratio_intra_rem')[ar_sou]
        w_exc_exc_intra = np.array(conn_intra.get('weight')) * conn_dict.get('ratio_intra_rem')[ar_sou]
        conn_intra.set({'weight': w_exc_exc_intra,'alpha':conn_dict.get('alpha_rem'),'lambda':conn_dict.get('lambda_intra_rem'),
                        'Wmax': Wmax_intra,'tau_plus': conn_dict.get('tau_plus_rem')})

        #CX (INH - EXC)
        print('\nSet Status: Layer1  intra-area %d' % (ar_sou))
        print('\nW INH1-EXC: %lg\nW INH2-EXC: %lg\n' % (conn_dict.get('W_inh_exc_1_rem'),conn_dict.get('W_inh_exc_2_rem')))
        for pop in range(layer1.n_pop_inh):
            label = f'W_inh_exc_{pop+1}_rem'
            conn = conn_inh_exc[pop][ar_sou]
            weight = conn_dict.get(label)
            conn.set({'weight': weight})

        #TH (INH - EXC)
        print('\nSet Status: intra-area %d th - th ' %(ar_sou))
        print('\nWeight INH - EXC: %lg' %(conn_dict.get('W_th_inh_exc_rem')))
        conn_th_inh_exc.set({'weight': conn_dict.get('W_th_inh_exc_rem')})

        #TH - CX
        print('\nSet Status: intra-area %d th - l1 ' %(ar_sou))
        print('\nratio TH - CX: %lg , lambda: 0' %(conn_dict.get('ratio_th_l1_rem')))
        w_th_l1_intra = np.array(conn_th_l1.get('weight')) * conn_dict.get('ratio_th_l1_rem')
        Wmax = np.array(conn_th_l1.get('Wmax')) * conn_dict.get('ratio_th_l1_rem')
        conn_th_l1.set({'Wmax':Wmax,'weight': w_th_l1_intra,'lambda':0.})

        #CX - TH
        print('\nSet Status: intra-area %d l1 - th ' %(ar_sou))
        print('\nratio CX - TH: %lg , lambda: 0' %(conn_dict.get('ratio_l1_th_rem')))
        w_l1_th_intra = np.array(conn_l1_th.get('weight'))*conn_dict.get('ratio_l1_th_rem')
        Wmax = np.array(conn_l1_th.get('Wmax')) * conn_dict.get('ratio_l1_th_rem')
        conn_l1_th.set({'Wmax':Wmax,'weight': w_l1_th_intra,'lambda':0.})

        if layer1.areas>1:
            #CX (EXC INTER)
            n1 = 1 if ar_sou == 0 else 2
            n2 = 2 if ar_sou == 0 else 1
            print('\nSet Connection Dynamics: %s Layer1 inter-area;   %d ---> %d\n' % (layer1.name, n1, n2))
            print('\nalpha inter: %lg , lambda inter: %lg , ratio inter: %lg , tau_plus: %lg\n'
                  % (conn_dict.get('alpha_rem'),conn_dict.get('lambda_inter_rem'), conn_dict.get('ratio_inter_rem')[ar_sou],
                     conn_dict.get('tau_plus_rem')))
            Wmax_inter = np.array(conn_inter.get('Wmax')) * conn_dict.get('ratio_inter_rem')[ar_sou]
            w_exc_exc_inter = np.array(conn_inter.get('weight')) * conn_dict.get('ratio_inter_rem')[ar_sou]
            conn_inter.set({'weight': w_exc_exc_inter,'alpha':conn_dict.get('alpha_rem'),'lambda': conn_dict.get('lambda_inter_rem'),
                                    'Wmax': Wmax_inter,'tau_plus': conn_dict.get('tau_plus_rem')})


def SaveWeights(layer_list, conn_layers, shuffling, save_path, stage, save_all_output=False):

    path = save_path
    if not os.path.exists(path):
        try:
            os.makedirs(path)
        except BaseException as err:
            raise err

    for layer in layer_list:
        save_dict = {}
        layer_name = layer.name

        if layer_name == 'th':

            if save_all_output == False:
                conn_keys = [('th' , 'all', 'fwd')]
            elif save_all_output == True:
                conn_keys = [('th' , 'all', 'fwd'), ('th' , 'intra', 'exc_inh'), ('th' , 'intra', 'inh_exc'), ('noise' , 'all', 'th', 'exc')]

            keys_dict = {'fwd': 'fwd', 'exc_inh': 'exc_inh',
                         'inh_exc': 'inh_exc', 'exc': 'noise_exc'}

        if layer_name == 'cx':

            if save_all_output == False:
                conn_keys = [('cx' , 'all', 'exc_exc'), ('cx' , 'all', 'bwd'), ('cx' , 'all', 'fwd')]
            elif save_all_output == True:
                conn_keys = [('cx' , 'all', 'exc_exc'), ('cx' , 'all', 'bwd'), ('cx' , 'all', 'fwd'),
                             ('cx', 'intra', 'exc_inh'), ('cx', 'intra', 'inh_exc'), ('cx', 'intra', 'inh_inh'),
                             ('noise', 'all', 'cx', 'exc_1'), ('noise', 'all', 'cx', 'inh')]

            keys_dict = {'exc_exc': 'exc_exc', 'bwd': 'bwd', 'fwd': 'fwd',
                         'exc_inh': 'exc_inh', 'inh_exc': 'inh_exc', 'inh_inh': 'inh_inh',
                         'exc_1': 'noise_exc', 'inh': 'noise_inh'}

        if layer_name == 'ro':
            conn_keys = [('ro' , 'all', 'exc_exc')]
            keys_dict = {'exc_exc': 'exc_exc'}

        for keys in conn_keys:
            key_conn = keys_dict[keys[-1]]
            print(f'\nSave {layer_name} {key_conn} connections weights')
            conn_dict = GetDictValue(conn_layers, keys)
            conn = conn_dict['conn']
            if type(conn) != list:
                W, S, T = conn.get('weight'), conn_dict['source'], conn_dict['target']
            elif type(conn[0]) != list:
                n_areas = len(conn)
                W = FlattenList2List([conn[area].get('weight') for area in range(n_areas)])
                S = FlattenList2List([conn[area].get('source') for area in range(n_areas)])
                T = FlattenList2List([conn[area].get('target') for area in range(n_areas)])
            else:
                n_areas, n_pop_inh = len(conn[0]), len(conn)
                W = FlattenList2List([[conn[pop][area].get('weight') for pop in range(n_pop_inh)] for area in range(n_areas)])
                S = FlattenList2List([[conn[pop][area].get('source') for pop in range(n_pop_inh)] for area in range(n_areas)])
                T = FlattenList2List([[conn[pop][area].get('target') for pop in range(n_pop_inh)] for area in range(n_areas)])

            stw_ordered = sorted(tuple(zip(S, T, W)), key=lambda tup: (tup[0],tup[1]))
            source_ordered, target_ordered, weights_ordered = [stw[0] for stw in stw_ordered], [stw[1] for stw in stw_ordered], [stw[2] for stw in stw_ordered]

            if layer_name == 'cx' and 'inh' in key_conn and 'noise' not in key_conn:
                W = [[conn[pop][area].get('weight') for area in range(n_areas)] for pop in range(n_pop_inh)]
                S = [[conn[pop][area].get('source') for area in range(n_areas)] for pop in range(n_pop_inh)]
                T = [[conn[pop][area].get('target') for area in range(n_areas)] for pop in range(n_pop_inh)]
                pop_spec = ['inhf', 'inhs']
                for pop in range(n_pop_inh):
                    save_dict[key_conn.replace('inh', pop_spec[pop])] = {'weight': W[pop], 'source': S[pop], 'target': T[pop]}
            else:
                save_dict[key_conn] = {'weight': weights_ordered, 'source': source_ordered, 'target': target_ordered}

            if save_all_output == True:
                if type(conn) != list:
                    D = conn.get('delay')
                elif type(conn[0]) != list:
                    n_areas = len(conn)
                    D = FlattenList2List([conn[area].get('delay') for area in range(n_areas)])
                else:
                    n_areas, n_pop_inh = len(conn[0]), len(conn)
                    D = FlattenList2List([[conn[pop][area].get('delay') for pop in range(n_pop_inh)] for area in range(n_areas)])

                std_ordered = sorted(tuple(zip(S, T, D)), key=lambda tup: (tup[0], tup[1]))
                delays_ordered = [std[2] for std in std_ordered]

                if layer_name == 'cx' and 'inh' in key_conn and 'noise' not in key_conn:
                    D = [[conn[pop][area].get('delay') for area in range(n_areas)] for pop in range(n_pop_inh)]
                    pop_spec = ['inhf', 'inhs']
                    for pop in range(n_pop_inh): save_dict[key_conn.replace('inh', pop_spec[pop])].update({'delay': D[pop]})

                else:
                    save_dict[key_conn].update({'delay': delays_ordered})

        try:
            np.save(os.path.join(path, f'conn_{layer_name}_{stage}'), save_dict)
        except BaseException as err:
            raise err


def IDsDecoder(S_pre, T_pre, W_pre, S_new, T_new, W_new):

    # Source - Target - Pesi
    S_pre -= np.min(S_pre)
    T_pre -= np.min(T_pre)
    n_S_pre = len(np.unique(S_pre))
    n_T_pre = len(np.unique(T_pre))
    w_pre = np.zeros((n_S_pre, n_T_pre))

    S_new -= np.min(S_new)
    T_new -= np.min(T_new)
    n_S_new = len(np.unique(S_new))
    n_T_new = len(np.unique(T_new))
    w_new = np.zeros((n_S_new,n_T_new))

    print(n_S_pre,n_S_new,n_T_pre,n_T_new)

    print(np.shape(w_pre),np.shape(W_pre),np.shape(w_new),np.shape(W_new))

    print(np.shape(S_pre), np.shape(S_new))

    S_pre[(S_pre>=n_S_pre)] = S_pre[(S_pre>=n_S_pre)] - n_S_pre//2
    T_pre[(T_pre >= n_T_pre)] = T_pre[(T_pre >= n_T_pre)] - n_T_pre // 2
    S_new[(S_new>=n_S_new)] = S_new[(S_new>=n_S_new)] - n_S_new//2
    T_new[(T_new >= n_T_new)] = T_new[(T_new >= n_T_new)] - n_T_new // 2

    for s, t, n in zip(S_new, T_new, range(len(S_new))):
        w_new[s][t] = W_new[n]

    for s, t, n in zip(S_pre, T_pre, range(len(S_pre))):
        w_pre[s][t] = W_pre[n]

    for s, t in zip(S_pre, T_pre):
        w_new[s][t] = w_pre[s][t]

    for s, t, n in zip(S_new, T_new, range(len(S_new))):
        W_new[n] = w_new[s][t]

    return W_new


def LoadWeights(layer_list, conn_layers, syn_params, load_path, stage):

    print('\nLoading synaptic weights STAGE %s\n' %(stage))
    for layer in layer_list:

        path = os.path.join(load_path, 'conn_%s_%s.npy' % (layer.name, stage))
        print('\nLoad Path: %s' % path)
        try:
            loaded_weights = np.load(path, allow_pickle=True)
        except BaseException as err:
            raise err

        if layer.name == 'th':
            # TH --> CX
            print('\nTH --> CX weights loaded \n')
            w_th_l1 = loaded_weights.item()['fwd']
            conn = conn_layers['th']['all']['fwd']['conn']
            S_new, T_new, W_new = conn.get('source'), conn.get('target'), conn.get('weight')
            S, T, w = w_th_l1['source'], w_th_l1['target'], w_th_l1['weight']
            w_new = IDsDecoder(S,T,w,S_new,T_new,W_new)
            conn_layers['th']['all']['fwd']['source'] = S_new
            conn_layers['th']['all']['fwd']['target'] = T_new
            conn.set({'weight': w_new})

        if layer.name == 'cx':
            # CX --> TH
            print('CX --> TH weights loaded \n')
            w_l1_th = loaded_weights.item()['bwd']
            conn = conn_layers['cx']['all']['bwd']['conn']
            S_new, T_new, W_new = conn.get('source'), conn.get('target'), conn.get('weight')
            S, T, w = w_l1_th['source'], w_l1_th['target'], w_l1_th['weight']
            w_new = IDsDecoder(S,T,w,S_new,T_new,W_new)
            conn_layers['cx']['all']['bwd']['source'] = S_new
            conn_layers['cx']['all']['bwd']['target'] = T_new
            conn.set({'weight': w_new})

            # EXC <--> EXC
            print('\nCX --> CX weights loaded ')
            w_rec = loaded_weights.item()['exc_exc']
            conn = conn_layers['cx']['all']['exc_exc']['conn']
            S_new, T_new, W_new = conn.get('source'), conn.get('target'), conn.get('weight')
            S, T, w = w_rec['source'], w_rec['target'], w_rec['weight']
            w_new = IDsDecoder(S,T,w,S_new,T_new,W_new)
            conn_layers['cx']['all']['exc_exc']['source'] = S_new
            conn_layers['cx']['all']['exc_exc']['target'] = T_new
            conn.set({'weight': w_new})

            # CX --> RO
            print('CX --> RO weights loaded')
            w_l1_l2 = loaded_weights.item()['fwd']
            conn = conn_layers['cx']['all']['fwd']['conn']
            S_new, T_new, W_new = conn.get('source'), conn.get('target'), conn.get('weight')
            S, T, w = w_l1_l2['source'], w_l1_l2['target'], w_l1_l2['weight']
            w_new = IDsDecoder(S,T,w,S_new,T_new,W_new)
            conn_layers['cx']['all']['fwd']['source'] = S_new
            conn_layers['cx']['all']['fwd']['target'] = T_new
            conn.set({'weight': w_new})

        if layer.name == 'ro':
            # RO - RO
            print('\nRO --> RO weights loaded \n')
            w_rec = loaded_weights.item()['exc_exc']
            conn = conn_layers['ro']['all']['exc_exc']['conn']
            S_new, T_new, W_new = conn.get('source'), conn.get('target'), conn.get('weight')
            S, T, w = w_rec['source'], w_rec['target'], w_rec['weight']
            w_new = IDsDecoder(S,T,w,S_new,T_new,W_new)
            conn_layers['ro']['all']['exc_exc']['source'] = S_new
            conn_layers['ro']['all']['exc_exc']['target'] = T_new
            conn.set({'weight': w_new})


def log_hist(data, bins, cumulative=False, label='', color='red', fc=(1,0,0), alpha=0.7):
    data = data[(data > 0.0001)]

    trasp = alpha
    fc = [fc[i] / 255. for i in range(len(fc))]
    fc.append(1 - trasp)

    logbins = np.logspace(np.log10(min(data)), np.log10(max(data)), bins)

    plt.xscale('log')
    plt.hist(data, bins=logbins, density=True, cumulative=cumulative, fc=fc, label=label)
    if label != '': plt.legend(prop={'size': 100})
    plt.hist(data, bins=logbins, density=True, cumulative=cumulative, color=color,histtype='step')
    plt.yscale('log')


def OrderByClass(w, n_exc_ca, shuffling):

    n_S, n_T = np.shape(w)

    w_new = np.zeros((n_S,n_T))

    for s in range(n_S):
        group1 = s // n_exc_ca
        group1_sh = shuffling[group1]
        neur1_id = s % n_exc_ca
        s_new = int(group1_sh * n_exc_ca + neur1_id)
        for t in range(n_T):
            group2 = t // n_exc_ca
            group2_sh = shuffling[group2]
            neur2_id = t % n_exc_ca
            t_new = int(group2_sh * n_exc_ca + neur2_id)
            w_new[s_new, t_new] = w[s, t]

    return w_new


def HistCXCX(w, Wmax_dict, ranks, n_class, stage, plot_path, cycle, n_exc_ca):

    if cycle==0: cycle = 1
    areas = 2
    n_groups = cycle * ranks * n_class * areas
    N = len(w)


    # inter/intra class/group histogram
    w_intra_group = [[] for i in range(n_groups)]
    w_intra_class = [[] for i in range(n_class)]
    w_inter_group = [[] for i in range(n_groups)]
    w_inter_class = [[] for i in range(n_class)]
    w_diff = []

    print(np.shape(w))

    for i in range(N):
        for j in range(N):
            group1 = i // n_exc_ca
            group2 = j // n_exc_ca
            class1 = group1 // (ranks*cycle) % n_class
            class2 = group2 // (ranks*cycle) % n_class
            area1 = group1 // (n_class * ranks * cycle)
            area2 = group2 // (n_class * ranks * cycle)
            if area1 == area2:
                if group1 == group2 and i != j:
                    w_intra_group[group1].append(w[i][j])
                if class1 == class2 and group1 != group2:
                    w_intra_class[class1].append(w[i][j])
                if class1 != class2:
                    w_diff.append(w[i][j])
            elif area1 != area2:
                delta = np.abs(group1-group2)
                if  delta == n_class*ranks*cycle:
                    w_inter_group[np.min([group1,group2])].append(w[i][j])
                if class1 == class2 and delta != n_class*ranks*cycle:
                    w_inter_class[class1].append(w[i][j])
                if class1 != class2:
                    w_diff.append(w[i][j])

    mu_intra_group = np.median(np.hstack(w_intra_group))
    mu_intra_class = np.median(np.hstack(w_intra_class))
    mu_inter_group = np.median(np.hstack(w_inter_group))
    mu_inter_class = np.median(np.hstack(w_inter_class))
    mu_diff = np.mean(w_diff)

    print('\nintra group: %lg'
          '\nintra class: %lg\ninter group: %lg\ninter class: %lg\ndiff: %lg\n'
          %(mu_intra_group,mu_intra_class,mu_inter_group,mu_inter_class,mu_diff))

    Wmax = Wmax_dict['cx_cx']

    plt.figure(figsize=(50, 30))
    fontsize = 55
    set_ticks(30, 2, 10, 2)
    plt.xlabel('Weight post-%s' % (stage.split('_')[1]), fontsize=fontsize)
    #plt.yscale('log')
    #plt.xscale('log')
    plt.grid(axis='x')
    plt.xlim(0.01,Wmax)
    plt.xticks(fontsize=fontsize)

    boxes = []
    box_lab = []
    box_clr = []

    data = np.hstack(w_intra_group)
    box = plt.boxplot(data[(data>0.01)],showfliers=False,notch=False,positions=np.arange(0,1),patch_artist=True,vert=False)
    boxes.append(box)
    box_lab.append('same example\n(intra)')
    box_clr.append(['darkorange','orange'])

    if len(np.hstack(w_inter_group))>0 :
        data = np.hstack(w_inter_group)
        box = plt.boxplot(data[(data>0.01)],showfliers=False,notch=False,positions=np.arange(1,2),patch_artist=True,vert=False)
        boxes.append(box)
        box_lab.append('same example\n(inter)')
        box_clr.append(['darkblue','deepskyblue'])

    if len(np.hstack(w_intra_class))>0 :
        data = np.hstack(w_intra_class)
        box = plt.boxplot(data[(data>0.01)],showfliers=False,notch=False,positions=np.arange(2,3),patch_artist=True,vert=False)
        boxes.append(box)
        box_lab.append('same class\n(intra)')
        box_clr.append(['gold','yellow'])

    if len(np.hstack(w_inter_class))>0 :
        data = np.hstack(w_inter_class)
        box = plt.boxplot(data[(data>0.01)],showfliers=False,notch=False,positions=np.arange(3,4),patch_artist=True,vert=False)
        boxes.append(box)
        box_lab.append('same class\n(inter)')
        box_clr.append(['darkgreen','springgreen'])

    data = np.hstack(w_diff)
    box = plt.boxplot(data[(data>0.01)],showfliers=False,notch=False,positions=np.arange(4,5),patch_artist=True,vert=False)
    boxes.append(box)
    box_lab.append('else')
    box_clr.append(['darkred','red'])

    plt.yticks(np.arange(len(boxes)),box_lab,fontsize=fontsize)
    for box,n in zip(boxes,range(len(boxes))):
        for element in ['boxes', 'whiskers', 'caps']:
            plt.setp(box[element], color=box_clr[n][0],linewidth=5)
        plt.setp(box['medians'], color='black',linewidth=5)
        for patch in box['boxes']:
            patch.set_facecolor(box_clr[n][1])

    plt.legend([bp['boxes'][0] for bp in boxes], ['%.3lf'%(bp['medians'][0].get_xdata()[0]) for bp in boxes],prop={'size': 40})
    #plt.legend(prop={'size': 100})

    try:
        plt.savefig(os.path.join(plot_path, 'bp_weights_%s.png' % stage))
    except BaseException as err:
        raise err

    plt.figure(figsize=(70, 20*cycle*ranks))
    for group in range(n_groups):
        plt.subplot(2*cycle*ranks, n_class, group + 1)
        data = np.array(w_intra_group[group])
        plt.hist(data[(data>0.01)], bins=50, density=True, color='darkorange',edgecolor='black',
                 label='median = %lg' % (np.mean(w_intra_group[group])))
        plt.xlabel('weight', fontsize=30)
        plt.xlim(0,75)
        plt.title('Intragroup %d' % (group), fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 40})
    plt.tight_layout()

    try:
        plt.savefig(os.path.join(plot_path, 'hist_intragroup_weights_cx_%s.png' % stage))
    except BaseException as err:
        raise err

    plt.figure(figsize=(70, 20 * cycle * ranks))
    for group in range(n_groups//areas):
        plt.subplot(cycle*ranks, n_class, group + 1)
        data = np.array(w_inter_group[group])
        plt.hist(data[(data>0.01)], bins=50, density=True, color='deepskyblue',edgecolor='black',
                 label='median = %lg' % (np.mean(w_intra_group[group])))
        plt.xlabel('weight', fontsize=30)
        plt.xlim(0,75)
        plt.title('Intragroup %d' % (group), fontsize=30)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.legend(prop={'size': 40})
    plt.tight_layout()

    try:
        plt.savefig(os.path.join(plot_path, 'hist_intergroup_weights_cx_%s.png' % stage))
    except BaseException as err:
        raise err


def HistTHCX(conn_layers, save_path, stage):

    conn_th = conn_layers['th']['all']['fwd']
    W_th = conn_th['conn'].get('weight')
    S_th = conn_th['source']
    T_th = conn_th['target']
    Wmax_th_cx = np.unique(conn_th['conn'].get('Wmax'))

    n_S = len(np.unique(S_th))
    n_T = len(np.unique(T_th))

    # Source - Target - Pesi
    S_th -= np.min(S_th)
    T_th -= np.min(T_th)

    w_th = np.zeros((n_S, n_T))

    for s, t, n in zip(S_th, T_th, range(len(S_th))):
        w_th[s][t] = W_th[n]

    w_th = np.hstack(w_th)

    conn_cx = conn_layers['cx']['all']['bwd']
    W_cx = conn_cx['conn'].get('weight')
    S_cx = conn_cx['source']
    T_cx = conn_cx['target']
    Wmax_cx_th = np.unique(conn_cx['conn'].get('Wmax'))


    n_S = len(np.unique(S_cx))
    n_T = len(np.unique(T_cx))

    # Source - Target - Pesi
    S_cx -= np.min(S_cx)
    T_cx -= np.min(T_cx)

    w_cx = np.zeros((n_S, n_T))

    for s, t, n in zip(S_cx, T_cx, range(len(S_cx))):
        w_cx[s][t] = W_cx[n]

    w_cx = np.hstack(w_cx)

    w_cx = w_cx[(w_cx>0.1)]
    w_th = w_th[(w_th>0.3)]



    plt.figure(figsize=(70, 40))
    plt.subplot(2, 1, 1)
    plt.hist(np.log(w_th), bins=100, density=True, color='blue',label='mu = %lg' % (np.mean(w_th)))
    plt.xlabel('weight', fontsize=30)
    plt.xlim(0,Wmax_th_cx)
    plt.title('TH --> CX synapses', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(prop={'size': 24})
    plt.subplot(2, 1, 2)
    plt.hist(np.log(w_cx), bins=100, density=True, color='red',label='mu = %lg' % (np.mean(w_cx)))
    plt.xlabel('weight', fontsize=30)
    plt.xlim(0,Wmax_cx_th)
    plt.title('CX --> TH synapses', fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.legend(prop={'size': 24})
    plt.tight_layout()

    try:
        plt.savefig(os.path.join(save_path, 'ThCx_syn_%s.png' % stage))
    except BaseException as err:
        raise err


def WeightIntraGroup(conn, conn_type, n_exc_ca):

    areas = len(conn)
    mean = np.zeros(areas)

    for area in range(areas):
        S = conn[area].get('source')
        T = conn[area].get('target')
        W = conn[area].get('weight')

        # Source - Target - Pesi
        S -= np.min(S)
        T -= np.min(T)


        w_group = []

        for s, t, w in zip(S, T, W):
            group_source = s // n_exc_ca
            group_target = t // n_exc_ca
            delta = int(np.abs(group_source - group_target))
            if conn_type == 'intra' and delta == 0 and s != t:
                w_group.append(w)
            if conn_type == 'inter' and delta == 0:
                w_group.append(w)

        mean[area] = np.mean(w_group)

    return mean


def SynapticModulation(w_new, conn, conn_type, n_exc_ca, n_group_area):

    areas = len(conn)

    ratio = np.zeros(areas)

    for area in range(areas):

        S = conn[area].get('source')
        T = conn[area].get('target')
        W = conn[area].get('weight')

        # Source - Target - Pesi
        S -= np.min(S)
        T -= np.min(T)

        w_group = []

        for s, t, w in zip(S, T, W):
            group_source = s // n_exc_ca
            group_target = t // n_exc_ca
            delta = int(np.abs(group_source - group_target))
            if conn_type == 'intra' and delta == 0 and s != t:
                w_group.append(w)
            if conn_type == 'inter' and delta % n_group_area == 0:
                w_group.append(w)

        ratio[area] = w_new / np.mean(w_group)

    return ratio


def WeightMap(layer, conn_layers, shuffling, save_path, stage, cycle, log=False, conn_key='cx_cx'):

    print('\nWeight Map %s\n' % (layer.name))

    save_graph = os.path.join(save_path, 'Plot')

    if not os.path.exists(save_graph):
        try:
            os.makedirs(save_graph)
        except BaseException as err:
            raise err

    conn_dict = {'th_cx': conn_layers['th']['all']['fwd'], 'cx_th': conn_layers['cx']['all']['bwd'],
                 'cx_cx':conn_layers['cx']['all']['exc_exc'], 'cx_ro': conn_layers['cx']['all']['fwd'],
                 'ro_ro': conn_layers['ro']['all']['exc_exc']}
    Wmax_dict = {'th_cx': conn_layers['th']['all']['fwd']['conn'].get('Wmax')[0],'cx_th': conn_layers['cx']['all']['bwd']['conn'].get('Wmax')[0],
                 'cx_cx': conn_layers['cx']['all']['exc_exc']['conn'].get('Wmax')[0],'cx_ro': conn_layers['cx']['all']['fwd']['conn'].get('Wmax')[0],
                 'ro_ro': conn_layers['ro']['all']['exc_exc']['conn'].get('Wmax')[0]}

    Wmax = Wmax_dict[conn_key]
    W = conn_dict[conn_key]['conn'].get('weight')
    S = conn_dict[conn_key]['source']
    T = conn_dict[conn_key]['target']


    n_S = len(np.unique(S))
    n_T = len(np.unique(T))

    # Source - Target - Pesi
    S -= np.min(S)
    T -= np.min(T)

    w = np.zeros((n_S, n_T))

    for s, t, n in zip(S, T, range(len(S))):
        w[s][t] = W[n]

    if layer.name == 'cx': w = OrderByClass(w, layer.n_exc_ca, shuffling)

    #norm = color.LogNorm(0.001, 150) if log == True else color.Normalize(0.001, 150)

    plt.figure(figsize=(64, 64))
    cmap = 'viridis'#parula()
    if log == True:
        plt.imshow(np.log(np.transpose(w)), cmap=cmap)
        plt.clim(-4, 3)
    elif log == False:
        norm = color.Normalize(0.001, Wmax)
        plt.imshow(np.transpose(w), cmap=cmap, norm=norm)

    #plt.imshow(w, cmap=cmap, norm=norm)
    plt.ylim(-0.5, n_S-0.5)
    plt.xticks(fontsize=50)
    plt.yticks(fontsize=50)
    plt.xlabel('neuron ID', fontsize=50)
    plt.ylabel('neuron ID', fontsize=50)
    plt.title('Cortico-cortical weights heatmap', fontsize=50)
    plt.colorbar().ax.tick_params(labelsize=50)

    n_exc = layer.n_exc
    n_exc_ca = layer.n_exc_ca
    n_ranks = layer.n_ranks_train * cycle if cycle>0 else layer.n_ranks_train
    n_exc_classes = n_exc_ca * n_ranks
    n_class = np.arange(n_exc_classes-0.5, n_exc-0.5, n_exc_classes)
    plt.vlines(n_class, 0, layer.n_exc, color='darkred')
    plt.hlines(n_class, 0, layer.n_exc, color='darkred')

    try:
        plt.savefig(os.path.join(save_graph, 'wm_%s_%s.png' % (conn_key, stage)))
    except BaseException as err:
        raise err

    if conn_key == 'cx_cx':
        HistCXCX(w, Wmax_dict, ranks=layer.n_ranks_train, n_class=layer.n_class, stage=stage, plot_path=save_graph, cycle=cycle, n_exc_ca=layer.n_exc_ca)
    if layer.name == 'cx':
        HistTHCX(conn_layers, save_graph, stage)
