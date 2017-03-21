# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
#
# This file is part of GWpy.
#
# GWpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GWpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""This module provides plotting utilities for visualising GW data

The standard data types (`TimeSeries`, `Table`, `DataQualityFlag`, ...) can
all be easily visualised using the relevant plotting objects, with
many configurable parameters both interactive, and in saving to disk.
"""

from matplotlib import (rcParams, rc_params)

from .tex import (USE_TEX, MACROS as TEX_MACROS)

# record matplotlib's original rcParams
DEFAULT_RCPARAMS = rc_params()

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# set default params
GWPY_PLOT_PARAMS = {
    'axes.grid': True,
    'axes.axisbelow': False,
    'axes.formatter.limits': (-3, 4),
    'axes.labelsize': 22,
    'axes.titlesize': 26,
    'grid.linestyle': ':',
    'grid.linewidth': .5,
    'image.aspect': 'auto',
    'image.interpolation': 'nearest',
    'image.origin': 'lower',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
}

# construct new default color cycle
GWPY_COLOR_CYCLE = [
    (0.0, 0.4, 1.0),  # blue
    'r',              # red
    (0.2, 0.8, 0.2),  # green
    (1.0, 0.7, 0.0),  # yellow(ish)
    (0.5, 0., 0.75),  # magenta
    'gray',
    (0.3, 0.7, 1.0),  # light blue
    'pink',
    (0.13671875, 0.171875, 0.0859375),  # dark green
    (1.0, 0.4, 0.0),  # orange
    'saddlebrown',
    'navy',
]

# set mpl version dependent stuff
try:
    from matplotlib import cycler
except (ImportError, KeyError):  # mpl < 1.5
    GWPY_PLOT_PARAMS['axes.color_cycle'] = GWPY_COLOR_CYCLE
else:
    GWPY_PLOT_PARAMS.update({
        'axes.prop_cycle': cycler('color', GWPY_COLOR_CYCLE),
    })

# set latex options
if rcParams['text.usetex'] or USE_TEX:
    GWPY_PLOT_PARAMS.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": TEX_MACROS,
    })

# update matplotlib rcParams with new settings
rcParams.update(GWPY_PLOT_PARAMS)

# fix matplotlib issue #3470
if rcParams['font.family'] == 'serif':
    rcParams['font.family'] = u'serif'
