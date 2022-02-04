# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2021-2022)
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

"""Tests for :mod:`gwpy.timeseries.io.losc`
"""

import pytest

from astropy.utils.data import download_file

from gwosc.locate import get_event_urls

from ...testing.errors import pytest_skip_network_error
from ...utils.env import bool_env
from .. import (
    StateVector,
    TimeSeries,
)

GWPY_CACHE = bool_env("GWPY_CACHE", False)


@pytest.fixture(scope="module")
@pytest_skip_network_error
def gw150914_hdf5():
    url, = get_event_urls(
        "GW150914",
        version=3,
        detector="L1",
        duration=32,
        sample_rate=4096,
        format="hdf5",
    )
    return download_file(url, cache=GWPY_CACHE)


def test_read_hdf5_gwosc(gw150914_hdf5):
    data = TimeSeries.read(
        gw150914_hdf5,
        format="hdf5.gwosc",
    )
    assert data.span == (1126259447, 1126259479)
    assert data.name == "Strain"
    assert data.max().value == pytest.approx(-4.60035111e-20)


def test_read_hdf5_gwosc_state(gw150914_hdf5):
    state = StateVector.read(
        gw150914_hdf5,
        format="hdf5.gwosc",
    )
    assert state.name == "Data quality"
    assert state.max().value == 127
