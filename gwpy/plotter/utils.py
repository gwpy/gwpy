# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Helper functions for plotting data with matplotlib and LAL
"""

import itertools
import re

from matplotlib import rcParams

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

rUNDERSCORE = re.compile(r'(?<!\\)_(?!.*{)')

# groups of input parameters (for passing to Plot() and subclasses)
FIGURE_PARAMS = [
    'figsize',
]
AXES_PARAMS = [
    'projection', 'sharex', 'sharey', 'xlim', 'ylim', 'xlabel', 'ylabel',
    'xscale', 'yscale', 'title', 'epoch',
]
LINE_PARAMS = [
    'linewidth', 'linestyle', 'color', 'label', 'alpha', 'rasterized',
]
COLLECTION_PARAMS = [
    'cmap', 'vmin', 'vmax', 'marker', 's', 'norm', 'rasterized',
]
IMAGE_PARAMS = [
    'imshow', 'cmap', 'vmin', 'vmax', 'norm', 'rasterized', 'extent',
    'origin', 'interpolation', 'aspect',
]
ARTIST_PARAMS = set(LINE_PARAMS + COLLECTION_PARAMS + IMAGE_PARAMS)
LEGEND_PARAMS = [
    'loc', 'borderaxespad', 'ncol',
]


def color_cycle(colors=None):
    """An infinite iterator of the given (or default) colors
    """
    if colors:
        return itertools.cycle(colors)
    return itertools.cycle(rcParams['axes.color_cycle'])


def marker_cycle(markers=None):
    """An infinite iterator of the given (or default) markers
    """
    if markers:
        return itertools.cycle(markers)
    return itertools.cycle(('o', 'x', '+', '^', 'D', 'H', '1'))
