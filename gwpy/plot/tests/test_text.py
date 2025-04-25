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

"""Tests for `gwpy.plot.text`."""

import pytest
from astropy.units import Unit
from matplotlib import (
    pyplot,
    rc_context,
)

from .. import text as plot_text


@pytest.mark.parametrize(("in_", "out", "texout"), [
    pytest.param(
        "test",
        "test",
        "test",
        id="str",
    ),
    pytest.param(
        4.0,
        "4.0",
        "4",
        id="float",
    ),
    pytest.param(
        8,
        "8",
        "8",
        id="int",
    ),
    pytest.param(
        Unit("m/Hz2"),
        "$\\mathrm{m\\,Hz^{-2}}$",
        "$\\mathrm{m\\,Hz^{-2}}$",
        id="unit",
    ),
])
def test_to_string(in_, out, texout):
    """Test `to_string()`."""
    with rc_context(rc={"text.usetex": False}):
        assert plot_text.to_string(in_) == out
    with rc_context(rc={"text.usetex": True}):
        assert plot_text.to_string(in_) == texout


@pytest.mark.parametrize(("unit", "label", "format_", "result"), [
    pytest.param(
        Unit("m"),
        None,
        "latex_inline_dimensional",
        r"Length [$\mathrm{m}$]",
        id="unit",
    ),
    pytest.param(
        Unit("m"),
        "My label",
        "latex_inline_dimensional",
        r"My label",
        id="label",
    ),
    pytest.param(
        Unit("m"),
        None,
        "console",
        "m",
        id="format",
    ),
])
def test_default_unit_label(unit, label, format_, result):
    """Test `default_unit_label()`."""
    fig = pyplot.figure()
    ax = fig.gca()
    if label:
        ax.set_xlabel(label)
    assert plot_text.default_unit_label(ax.xaxis, unit, format=format_) == result
    if not label:
        assert ax.xaxis.isDefault_label
