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

import os

from matplotlib import (rcParams, rc_params, __version__ as mpl_version)
from matplotlib.figure import SubplotParams

from . import tex
from .colors import DEFAULT_COLORS

# record matplotlib's original rcParams
MPL_RCPARAMS = rc_params()

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- custom rc ----------------------------------------------------------------

# set default params
DEFAULT_PARAMS = {
    # axes boundary colours
    'axes.edgecolor': 'gray',
    # grid
    'axes.grid': True,
    'axes.axisbelow': False,
    'grid.linestyle': ':',
    'grid.linewidth': .5,
    # ticks
    'axes.formatter.limits': (-3, 4),
    # fonts
    'axes.labelpad': 5,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'font.sans-serif': ['FreeSans'] + rcParams['font.sans-serif'],
    # image
    'image.aspect': 'auto',
    'image.interpolation': 'nearest',
    'image.origin': 'lower',
    # legend (revert to mpl 1.5 formatting in parts)
    'legend.numpoints': 2,
    'legend.edgecolor': 'inherit',
    'legend.handlelength': 1,
    'legend.fancybox': False,
}

# select tex formatting (or not)
try:  # allow user to override from environment
    _USETEX = os.environ['GWPY_USETEX'].lower() in ['1', 'true', 'yes', 'y']
except KeyError:  # 'or' means default to tex
    _USETEX = rcParams['text.usetex'] or tex.has_tex()

# set latex options
rcParams['text.latex.preamble'].extend(tex.MACROS)
if _USETEX:
    DEFAULT_PARAMS.update({
        'text.usetex': True,
        'font.family': 'serif',
        'axes.labelpad': 4,
        'axes.labelsize': 20,
        'axes.titlesize': 24,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
    })
    if mpl_version < '2.0':
        DEFAULT_PARAMS['font.serif'] = ['Computer Modern']

# build better default colour cycle for matplotlib < 2
if mpl_version < '2.0':
    try:
        from matplotlib import cycler
    except (ImportError, KeyError):  # mpl < 1.5
        DEFAULT_PARAMS['axes.color_cycle'] = DEFAULT_COLORS
    else:  # mpl >= 1.5
        DEFAULT_PARAMS.update({
            'axes.prop_cycle': cycler('color', DEFAULT_COLORS),
        })

# remove rcParams for old matplotlib
# https://matplotlib.org/1.5.1/users/whats_new.html#configuration-rcparams
if mpl_version < '1.5':
    for key in (
            'axes.labelpad',
            'legend.edgecolor',
    ):
        DEFAULT_PARAMS.pop(key, None)

# update matplotlib rcParams with new settings
rcParams.update(DEFAULT_PARAMS)

# -- dynamic subplot positioning ----------------------------------------------

SUBPLOT_WIDTH = {
    6.4: (.1875, .87),
    8.: (.15, .85),
    12.: (.1, .90),
}
SUBPLOT_HEIGHT = {
    3.: (.25, .83),
    4.: (.2, .85),
    4.8: (.16, .88),
    6.: (.13, .9),
    8.: (.1, .93),
}


def get_subplot_params(figsize):
    """Return sensible default `SubplotParams` for a figure of the given size

    Parameters
    ----------
    figsize : `tuple` of `float`
         the ``(width, height)`` figure size (inches)

    Returns
    -------
    params : `~matplotlib.figure.SubplotParams`
        formatted set of subplot parameters
    """
    width, height, = figsize
    try:
        left, right = SUBPLOT_WIDTH[width]
    except KeyError:
        left = right = None
    try:
        bottom, top = SUBPLOT_HEIGHT[height]
    except KeyError:
        bottom = top = None
    return SubplotParams(left=left, bottom=bottom, right=right, top=top)
