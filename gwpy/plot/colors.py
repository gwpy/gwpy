# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2019)
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

import numpy

from matplotlib import (__version__ as mpl_version, rcParams)
from matplotlib import colors
try:
    from matplotlib.colors import (_colors_full_map as color_map, to_rgb)
except ImportError:  # mpl < 2
    from matplotlib.colors import (cnames as color_map, ColorConverter)
    to_rgb = ColorConverter().to_rgb

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


# -- colour utilities ---------------------------------------------------------

def tint(col, factor=1.0):
    """Tint a color (make it darker), returning a new RGB array
    """
    # this method is more complicated than it need be to
    # support matplotlib-1.x.
    # for matplotlib-2.x this would just be
    #     h, s, v = colors.rgb_to_hsv(colors.to_rgb(c))
    #     v *= factor
    #     return colors.hsv_to_rgb((h, s, v))
    rgb = numpy.array(to_rgb(col), ndmin=3)
    hsv = colors.rgb_to_hsv(rgb)
    hsv[-1][-1][2] *= factor
    return colors.hsv_to_rgb(hsv)[-1][-1]


def format_norm(kwargs, current=None):
    """Format a `~matplotlib.colors.Normalize` from a set of kwargs

    Returns
    -------
    norm, kwargs
        the formatted `Normalize` instance, and the remaining keywords
    """
    norm = kwargs.pop('norm', current) or 'linear'
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    clim = kwargs.pop('clim', (vmin, vmax)) or (None, None)
    clip = kwargs.pop('clip', None)

    if norm == 'linear':
        norm = colors.Normalize()
    elif norm == 'log':
        norm = colors.LogNorm()
    elif not isinstance(norm, colors.Normalize):
        raise ValueError("unrecognised value for norm {!r}".format(norm))

    for attr, value in (('vmin', clim[0]), ('vmax', clim[1]), ('clip', clip)):
        if value is not None:
            setattr(norm, attr, value)

    return norm, kwargs
