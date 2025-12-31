#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  thaco.py
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

import argparse
from SourceCode.Parent_RunSimulation import ThaCo
from SourceCode.utils import NestDictToDict, MergeDicts


CLI = argparse.ArgumentParser()
CLI.add_argument("--model_dir", nargs='?', type=str, required=True,
                    help="path to model directory, where .yaml config files are located")
CLI.add_argument("--code_dir", nargs='?', type=str, required=False, default=None,
                    help="path to source code directory, where ThaCo executables are located")
CLI.add_argument("--input_dir", nargs='?', type=str, required=False, default=None,
                    help="path to input directory, where to load input data from")
CLI.add_argument("--output_dir", nargs='?', type=str, required=False, default=None,
                    help="path to output directory, where to save output data")
CLI.add_argument("--simulation_name", nargs='?', type=str, required=False, default='',
                    help="name to be given to the folder containing simulation output")
CLI.add_argument("--n_threads", nargs='?', type=int, required=True,
                    help="number of threads to be used by NEST simulator")
CLI.add_argument("--nest_seed", nargs='?', type=int, required=False,
                    help="NEST RNG seed")
CLI.add_argument("--np_seed", nargs='?', type=int, required=False,
                    help="numpy seed")
CLI.add_argument("--training_set_indices", type=str, required=False,
                    help="select training set indices")
CLI.add_argument("--test_set_indices", type=str, required=False,
                    help="select test set indices")
CLI.add_argument("--tune_name", type=str, required=False, default='',
                    help="config tune name")

if __name__ == '__main__':
    args, unknown = CLI.parse_known_args()

    thaco = ThaCo(args)

    config_complete, config_tune, config_tune_flatten = thaco.config_complete, thaco.config_tune, thaco.config_tune_flatten
    default_network = config_complete['default_network']

    cycles_configs_dict = thaco.ConfigsDict(config_tune_flatten)
    n_trials, stages  = config_tune['simulation']['n_trials'], list(cycles_configs_dict[0][0].keys())
    load_stages = ['' if stage == '00_awake_training' else stages[n_stage-1] for n_stage, stage in enumerate(stages)]
    n_learning_cycles = len([1 for stage in stages if stage[3:] == 'awake_training'])

    for n_trial in range(n_trials):
        for n_config, config_tune_trial in cycles_configs_dict[n_trial].items():
            n_configs = len(cycles_configs_dict[n_trial].keys())
            trial_syn_dir, trial_main_output_dir, trial_id = thaco.SavePath()
            current_learning_cycle = 0
            for n_stage, (stage_id, config_tune_stage_flatten) in enumerate(config_tune_trial.items()):
                if stage_id[3:] == 'awake_training': current_learning_cycle += 1
                config_tune_stage_flatten = thaco.LoadPath(config_tune_stage_flatten, trial_id, stage_id, load_stages[n_stage], trial_syn_dir, trial_main_output_dir, current_learning_cycle, n_learning_cycles)
                config_tune_stage_nested = NestDictToDict(config_tune_stage_flatten)
                config_stage_complete = config_complete.copy()

                print('\nTrial: %s (%d/%d)' % (trial_id, n_trial+1, n_trials))
                print('Configuration: %d/%d' % (n_config+1, n_configs))
                print('Stage: %s (%d/%d)' % (config_tune_stage_nested['simulation']['stage'], n_stage+1, len(stages)))
                print('Main Output path: %s' % config_tune_stage_nested['paths']['save_out'])
                print('Synapses Output path: %s' % config_tune_stage_nested['paths']['save_syn'])
                print('Synapses loading path: %s' % config_tune_stage_nested['paths']['load_syn'])

                try:
                    thaco.Run(config_stage_complete, config_tune_stage_nested)
                    pass
                except BaseException as err:
                    raise err
