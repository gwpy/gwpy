# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Tests for `gwpy.plot.text`
"""

import itertools

from matplotlib import (
    colors as mpl_colors,
)

from .. import utils as plot_utils


def test_color_cycle():
    cyc = plot_utils.color_cycle()
    assert isinstance(cyc, itertools.cycle)
    assert next(cyc) == mpl_colors.to_hex("C0")


def test_color_cycle_arg():
    cyc = plot_utils.color_cycle(['1', '2', '3'])
    assert isinstance(cyc, itertools.cycle)
    assert next(cyc) == '1'
    assert next(cyc) == '2'
    assert next(cyc) == '3'
    assert next(cyc) == '1'


def test_marker_cycle():
    cyc = plot_utils.marker_cycle()
    assert isinstance(cyc, itertools.cycle)
    assert next(cyc) == 'o'


def test_marker_cycle_arg():
    cyc = plot_utils.marker_cycle(['1', '2', '3'])
    assert isinstance(cyc, itertools.cycle)
    assert next(cyc) == '1'
    assert next(cyc) == '2'
    assert next(cyc) == '3'
    assert next(cyc) == '1'
