# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

from matplotlib import rcParams

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# groups of input parameters (for passing to Plot())
FIGURE_PARAMS = [
    'figsize', 'dpi',
]
AXES_PARAMS = [
    'projection', 'title',  # standard options
    'sharex', 'xlim', 'xlabel', 'xscale',  # X-axis params
    'sharey', 'ylim', 'ylabel', 'yscale',  # Y-axis params
    'epoch', 'insetlabels',  # special GWpy extras
]


def color_cycle(colors=None):
    """An infinite iterator of the given (or default) colors
    """
    if colors:
        return itertools.cycle(colors)
    return itertools.cycle(p["color"] for p in rcParams["axes.prop_cycle"])


def marker_cycle(markers=None):
    """An infinite iterator of the given (or default) markers
    """
    if markers:
        return itertools.cycle(markers)
    return itertools.cycle(('o', 'x', '+', '^', 'D', 'H', '1'))
