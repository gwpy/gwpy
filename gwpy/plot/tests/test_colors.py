# Copyright (c) 2018-2025 Cardiff University
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

"""Tests for `gwpy.plot.colors`."""

import pytest
from matplotlib.colors import (
    LogNorm,
    Normalize,
)
from numpy.testing import assert_array_equal

from .. import colors as plot_colors


@pytest.mark.parametrize(("in_", "factor", "out"), [
    pytest.param(
        "red",
        1.,
        (1., 0., 0.),
        id="red",
    ),
    pytest.param(
        (1., 0., 0.),
        1.,
        (1., 0., 0.),
        id="red-rgb",
    ),
    pytest.param(
        "green",
        .75,
        (0.0, 0.37647058823529411, 0.0),
        id="green",
    ),
])
def test_tint(in_, factor, out):
    """Test `tint()`."""
    assert_array_equal(
        plot_colors.tint(in_, factor=factor),
        out,
    )


def test_format_norm():
    """Test `format_norm()`."""
    norm, kwargs = plot_colors.format_norm({"blah": 1})
    assert isinstance(norm, Normalize)
    assert kwargs == {"blah": 1}


def test_format_norm_log():
    """Test `format_norm()` handling of ``norm="log"``."""
    norm, kwargs = plot_colors.format_norm({
        "norm": "log",
        "vmin": 1,
        "vmax": 10,
    })
    assert isinstance(norm, LogNorm)
    assert norm.vmin == 1
    assert norm.vmax == 10
    assert not kwargs


def test_format_norm_instance():
    """Test `format_norm()` handling of ``norm=<Normalize>``."""
    n = LogNorm()
    norm, kwargs = plot_colors.format_norm({
        "norm": n,
        "clim": (10, 1000),
    })
    assert norm is n
    assert norm.vmin == 10
    assert norm.vmax == 1000
    assert not kwargs


def test_format_norm_clim():
    """Test `format_norm()` handling of ``clim``."""
    norm, kwargs = plot_colors.format_norm({"clim": None})
    assert norm.vmin is None
    assert norm.vmax is None
    assert not kwargs
