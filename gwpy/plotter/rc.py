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

"""Custom default figure configuration
"""

from matplotlib import (rcParams, rc_params)
from matplotlib.figure import SubplotParams

from .tex import (USE_TEX, MACROS as TEX_MACROS)

# record matplotlib's original rcParams
MPL_RCPARAMS = rc_params()

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- custom rc ----------------------------------------------------------------

# set default params
DEFAULT_PARAMS = {
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
DEFAULT_COLORS = [
    '#0066ff',  # blue
    '#ff0000',  # red
    '#33cc33',  # green
    '#ffb200',  # yellow(ish)
    '#8000bf',  # magenta
    '#808080',  # gray
    '#4cb2ff',  # light blue
    '#ffc0cb',  # pink
    '#232c16',  # dark green
    '#ff6600',  # orange
    '#8b4513',  # saddlebrown
    '#000080',  # navy
]

# set mpl version dependent stuff
try:
    from matplotlib import cycler
except (ImportError, KeyError):  # mpl < 1.5
    DEFAULT_PARAMS['axes.color_cycle'] = DEFAULT_COLORS
else:
    DEFAULT_PARAMS.update({
        'axes.prop_cycle': cycler('color', DEFAULT_COLORS),
    })

# set latex options
if rcParams['text.usetex'] or USE_TEX:
    DEFAULT_PARAMS.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern"],
        "text.latex.preamble": TEX_MACROS,
    })

# update matplotlib rcParams with new settings
rcParams.update(DEFAULT_PARAMS)

# -- dynamic subplot positioning ----------------------------------------------

SUBPLOT_WIDTH = {
    8.: (.15, .88),
    12.: (.1, .92),
}
SUBPLOT_HEIGHT = {
    4.: (.2, .85),
    6.: (.13, .9),
    8.: (.11, .92),
}


def get_subplot_params(figsize):
    """Return sensible default `SubplotParams` for a figure of the given size

    Returns
    -------
    params : `~matplotlib.figure.SubplotParams`
        formatted set of subplot parameters
    """
    w, h, = figsize
    try:
        l, r = SUBPLOT_WIDTH[w]
    except KeyError:
        l = r = None
    try:
        b, t = SUBPLOT_HEIGHT[h]
    except KeyError:
        b = t = None
    return SubplotParams(**{'left': l, 'bottom': b, 'right': r, 'top': t})
