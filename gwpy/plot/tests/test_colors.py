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

"""Tests for `gwpy.plot.colors`
"""

import pytest

from numpy.testing import assert_array_equal

from matplotlib.colors import (Normalize, LogNorm)

from .. import colors as plot_colors


@pytest.mark.parametrize('in_, factor, out', [
    ('red', 1., (1., 0., 0.)),
    ((1., 0., 0.), 1., (1., 0., 0.)),
    ('green', .75, (0.0, 0.37647058823529411, 0.0)),
])
def test_tint(in_, factor, out):
    assert_array_equal(plot_colors.tint(in_, factor=factor), out)


def test_format_norm():
    # defaults
    norm, kwargs = plot_colors.format_norm({})
    assert isinstance(norm, Normalize)
    assert kwargs == {}

    # log norm
    norm, kwargs = plot_colors.format_norm(
        {'norm': 'log', 'vmin': 1, 'vmax': 10})
    assert isinstance(norm, LogNorm)
    assert norm.vmin == 1
    assert norm.vmax == 10

    # existing norm, change limits
    n = LogNorm()
    norm, kwargs = plot_colors.format_norm(
        {'norm': n, 'clim': (10, 1000)})
    assert norm is n
    assert norm.vmin == 10
    assert norm.vmax == 1000

    # check clim=None is honoured
    norm, kwargs = plot_colors.format_norm({'clim': None})
    assert norm.vmin is None and norm.vmax is None
