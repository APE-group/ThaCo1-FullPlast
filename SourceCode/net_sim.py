#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  net_sim.py
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
