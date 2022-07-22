# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2018-2021)
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

"""Tests for `gwpy.plot.rc`
"""

import pytest

from matplotlib import rcParams

from .. import rc as plot_rc


DEFAULT_LRTB = [
    rcParams[f"figure.subplot.{x}"]
    for x in ('left', 'right', 'bottom', 'top')
]


@pytest.mark.parametrize('figsize, lrbt', [
    ((6.4, 4.8), (.1875, .87, .16, .88)),
    ((0, 0), DEFAULT_LRTB),
])
def test_get_subplot_params(figsize, lrbt):
    params = plot_rc.get_subplot_params(figsize)
    for key, val in zip(('left', 'right', 'bottom', 'top'), lrbt):
        assert getattr(params, key) == val
