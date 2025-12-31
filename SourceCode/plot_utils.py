#!/usr/bin/env python3
#
#  -*- coding: utf-8 -*-
#
#  plot_utils.py
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

import matplotlib as mpl

def set_ticks(size, scale_size, width, scale_width):

    mpl.rcParams['xtick.major.size'] = size
    mpl.rcParams['xtick.major.width'] = width
    mpl.rcParams['xtick.minor.size'] = size/scale_size
    mpl.rcParams['xtick.minor.width'] = width/scale_width
    mpl.rcParams['ytick.major.size'] = size
    mpl.rcParams['ytick.major.width'] = width
    mpl.rcParams['ytick.minor.size'] = size/scale_size
    mpl.rcParams['ytick.minor.width'] = width/scale_width
