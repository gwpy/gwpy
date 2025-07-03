# Copyright (c) 2025 Cardiff University
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

"""Tests for `gwpy.plot.units`."""

import pytest
from astropy.units import Unit

from .. import units as plot_units


class TestLatexInlineDimensional:
    """Test for `gwpy.plot.units.LatexInlineDimensional`."""

    Format = plot_units.LatexInlineDimensional

    @pytest.mark.parametrize(("unit", "out"), [
        pytest.param(
            Unit("m"),
            r"Length [$\mathrm{m}$]",
            id="length",
        ),
        pytest.param(
            Unit("m/s"),
            r"Speed/Velocity [$\mathrm{m\,s^{-1}}$]",
            id="speed",
        ),
        pytest.param(
            Unit(""),
            "Dimensionless",
            id="dimensionless",
        ),
        pytest.param(
            Unit("strain"),
            r"[$\mathrm{strain}$]",
            id="strain",
        ),
    ])
    def test_to_string(self, unit, out):
        """Test `to_string()`."""
        assert unit.to_string(self.Format.name) == out
