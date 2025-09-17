# Copyright (c) 2013-2017 Louisiana State University
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

"""Tests for :mod:`gwpy.signal.spectral._utils`."""

import pytest
from astropy.units import Unit

from ..spectral import _utils as fft_utils


@pytest.mark.parametrize(("unit", "kwargs", "result"), [
    (Unit("m"), {}, Unit("m^2/Hz")),
    (Unit("m"), {"scaling": "density"}, Unit("m^2/Hz")),
    (Unit("m"), {"scaling": "spectrum"}, Unit("m^2")),
    (None, {}, Unit("Hz^-1")),
])
def test_scale_timeseries_unit(unit, kwargs, result):
    """Test :func:`gwpy.signal.spectral._utils.scale_timeseries_units`."""
    assert fft_utils.scale_timeseries_unit(unit, **kwargs) == result


def test_scale_timeseries_unit_error():
    with pytest.raises(
        ValueError,
        match=r"^unknown scaling: 'other'$",
    ):
        fft_utils.scale_timeseries_unit(None, scaling="other")
