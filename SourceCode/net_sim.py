#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Jul 30 23:04:15 2020
@author: rodja
"""

import yaml, time, argparse
import numpy as np
import nest
import Parent_BrainStates as State
from Parent_Network import Net as Network
from utils import MemoryUsage
from scipy.stats import truncexpon

CLI = argparse.ArgumentParser()
CLI.add_argument("--config_complete_path", nargs='?', type=str, required=True,
                    help="path to config_complete.yaml file to load")
CLI.add_argument("--config_tune_path", nargs='?', type=str, required=True,
                    help="path to config_tune.yaml file to load")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    print('\n\nMAIN')

    startTime = time.time()
    start_ram = MemoryUsage()

    try:
        with open(args.config_complete_path, 'r') as f:
            config_complete = yaml.safe_load(f)
    except BaseException as err:
        raise err

    try:
        with open(args.config_tune_path, 'r') as f:
            config_tune = yaml.safe_load(f)
    except BaseException as err:
        raise err

    ThaCo = Network(config_complete, config_tune)
    print('\nNetwork Initialization time: %g' %(time.time() - startTime))

    ThaCo.startTime = startTime

    State.Incremental(ThaCo)

    print('Total simulated time: %g s' % (ThaCo.t_simulation / 1000.), end='\n')
    print('Total machine time: %g s' % (time.time() - startTime), end='\n\n')
    print('Time ratio: %g s' % (1000*(time.time() - startTime) / ThaCo.t_simulation), end='\n\n')
