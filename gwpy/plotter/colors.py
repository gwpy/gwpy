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

"""Colour customisations for visualisation in GWpy
"""

from matplotlib import (__version__ as mpl_version, rcParams)

try:
    from matplotlib.colors import _colors_full_map as color_map
except ImportError:  # mpl < 2
    from matplotlib.colors import cnames as color_map

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# build better default colour cycle for matplotlib < 2
if mpl_version < '2.0':
    DEFAULT_COLORS = [
        '#1f77b4',  # blue
        '#ffb200',  # yellow(ish)
        '#33cc33',  # green
        '#ff0000',  # red
        '#8000bf',  # magenta
        '#808080',  # gray
        '#4cb2ff',  # light blue
        '#ffc0cb',  # pink
        '#232c16',  # dark green
        '#ff7fe0',  # orange
        '#8b4513',  # saddlebrown
        '#000080',  # navy
    ]
else:  # just in case anyone expects DEFAULT_COLORS to exist for mpl 2
    DEFAULT_COLORS = rcParams['axes.prop_cycle'].by_key()['color']

# -- recommended defaults for current Gravitational-Wave Observatories --------
# the below colours are designed to work well for the colour-blind, as well
# as in grayscale, so are recommended for publications

GWPY_COLORS = {
    'geo600':          '#222222',  # dark gray
    'kagra':           '#ffb200',  # yellow/orange
    'ligo-hanford':    '#ee0000',  # red
    'ligo-india':      '#b0dd8b',  # light green
    'ligo-livingston': '#4ba6ff',  # blue
    'virgo':           '#9b59b6',  # magenta/purple
}  # nopep8

# provide user mapping by IFO prefix
_GWO_PREFIX = {
    'geo600':          'G1',
    'kagra':           'K1',
    'ligo-hanford':    'H1',
    'ligo-india':      'I1',
    'ligo-livingston': 'L1',
    'virgo':           'V1',
}  # nopep8
GW_OBSERVATORY_COLORS = {_GWO_PREFIX[n]: GWPY_COLORS[n] for n in GWPY_COLORS}

# set named colour upstream in matplotlib, so users can specify as
# e.g. plot(..., color='gwpy:ligo-hanford')
color_map.update({'gwpy:{}'.format(n): c for n, c in GWPY_COLORS.items()})
