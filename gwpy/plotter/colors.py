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
