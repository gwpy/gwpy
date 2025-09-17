# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Unit tests for :mod:`gwpy.detector.units`."""

import pytest
from astropy import units

from ..units import parse_unit

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


@pytest.mark.parametrize(("arg", "unit"), [
    pytest.param(None, None),
    pytest.param(units.m, units.m),
    pytest.param("meter", units.m),
    pytest.param("Volts", units.V),
    pytest.param("Meters/Second", units.m / units.s),
    pytest.param("Amp", units.ampere),
    pytest.param("MPC", units.megaparsec),
    pytest.param("degrees_C", units.Unit("Celsius")),
    pytest.param("DegC", units.Unit("Celsius")),
    pytest.param("degrees_F", units.Unit("Fahrenheit")),
    pytest.param("time", units.second),  # LIGO default time 'unit'
    pytest.param("Time (sec)", units.second),  # Virgo default time 'unit'
    pytest.param("Seconds", units.second),  # GWOSC default time 'unit'
])
def test_parse_unit(arg, unit):
    """Test `parse_unit()`."""
    assert parse_unit(arg, parse_strict="silent") == unit


def test_parse_unit_parse_strict_raise():
    """Test that `parse_unit` raises exceptions appropriately."""
    # check that errors get raise appropriately
    with pytest.raises(
        ValueError,
        match=r"^'metre' did not parse as unit.* Did you mean meter?",
    ):
        parse_unit("metre", parse_strict="raise")


def test_parse_unit_parse_strict_warn():
    """Test that `parse_unit` emits warnings appropriately."""
    # check that warnings get posted, and a custom NamedUnit gets returned
    with pytest.warns(
        units.UnitsWarning,
        match=r"^'metre' did not parse as gwpy unit.* Did you mean meter?",
    ):
        u = parse_unit("metre", parse_strict="warn")
    assert isinstance(u, units.UnrecognizedUnit)
    assert str(u) == "metre"

    # assert that a newly-created unit only gets created once
    u2 = parse_unit("metre", parse_strict="warn")
    assert u2 is u  # same object


@pytest.mark.parametrize("name", [
    "NONE",
    "undef",
    "strain",
    "coherence",
    "sec",
    "torr",
    "cf",
    "cfm",
    "ptcls",
])
def test_detector_units(name):
    """Test that a bunch of custom units are registered."""
    assert units.Unit(name)
