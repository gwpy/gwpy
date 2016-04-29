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

import numpy
import itertools
import re

from . import rcParams

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

rUNDERSCORE = re.compile(r'(?<!\\)_')

# groups of input parameters (for passing to Plot() and subclasses)
FIGURE_PARAMS = [
    'figsize',
]
AXES_PARAMS = [
    'projection', 'sharex', 'sharey', 'xlim', 'ylim', 'xlabel', 'ylabel',
    'xscale', 'yscale', 'title',
]
LINE_PARAMS = [
    'linewidth', 'linestyle', 'color', 'label', 'alpha', 'rasterized',
]
COLLECTION_PARAMS = [
    'cmap', 'vmin', 'vmax', 'marker', 's', 'norm', 'rasterized',
]
ARTIST_PARAMS = set(LINE_PARAMS + COLLECTION_PARAMS)
LEGEND_PARAMS = [
    'loc', 'borderaxespad', 'ncol',
]


def float_to_latex(x, format="%.2g"):
    """Convert a floating point number into a TeX representation.

    In particular, scientific notation is handled gracefully: e -> 10^

    @code
    >>> float_to_latex(10)
    '$10$'
    >>> float_to_latex(1000)
    r'$10^{3}$'
    >>> float_to_latex(123456789)
    r'$1.2\times 10^{8}$'

    @returns a TeX format string (with math-mode dollars ($))
    """
    base_str = format % x
    if "e" not in base_str:
        return "$%s$" % base_str
    mantissa, exponent = base_str.split("e")
    exponent = exponent.lstrip("0+")
    if mantissa == "1":
        return r"$10^{%s}$" % exponent
    else:
        return r"$%s\times 10^{%s}$" % (mantissa, exponent)


def color_cycle(colors=None):
    """An infinite iterator of the given (or default) color cycle.
    """
    if colors:
        return itertools.cycle(colors)
    else:
        return itertools.cycle(rcParams['axes.color_cycle'])


def marker_cycle(markers=None):
    if markers:
        return itertools.cycle(markers)
    else:
        return itertools.cycle(('o', 'x', '+', '^', 'D', 'H', '1'))
