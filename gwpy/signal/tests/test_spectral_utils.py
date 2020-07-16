# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2020)
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

"""Unit test for signal module
"""

import pytest

from astropy import units

from ..spectral import _utils as fft_utils


def test_scale_timeseries_unit():
    """Test :func:`gwpy.signal.spectral.utils.scale_timeseries_units`
    """
    scale_ = fft_utils.scale_timeseries_unit
    u = units.Unit('m')
    # check default
    assert scale_(u) == units.Unit('m^2/Hz')
    # check scaling='density'
    assert scale_(u, scaling='density') == units.Unit('m^2/Hz')
    # check scaling='spectrum'
    assert scale_(u, scaling='spectrum') == units.Unit('m^2')
    # check anything else raises an exception
    with pytest.raises(ValueError):
        scale_(u, scaling='other')
    # check null unit
    assert scale_(None) == units.Unit('Hz^-1')
