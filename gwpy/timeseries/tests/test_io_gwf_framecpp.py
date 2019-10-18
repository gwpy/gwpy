# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Tests for :mod:`gwpy.timeseries.io.gwf.framecpp`
"""

import pytest

import numpy

from ...detector import Channel
from ...testing.utils import (
    assert_quantity_sub_equal,
    TemporaryFilename,
)
from ...timeseries import TimeSeries

frameCPP = pytest.importorskip("LDAStools.frameCPP")  # noqa: F401


@pytest.fixture
def int32ts():
    ts = TimeSeries(
        numpy.arange(10, dtype="int32"),
        name="test",
        unit="m",
        channel=Channel(
            "test",
            sample_rate=1,
            dtype="int32",
            unit="m",
        )
    )
    return ts


def test_read_scaled_false(int32ts):
    with TemporaryFilename() as tmpf:
        int32ts.write(tmpf, format="gwf.framecpp", type="adc")
        new = type(int32ts).read(tmpf, "test", type="adc", scaled=False)
    assert new.dtype == int32ts.dtype
    assert new.unit == "ct"


def test_read_scaled_type_change(int32ts):
    with TemporaryFilename() as tmpf:
        int32ts.write(tmpf, format="gwf.framecpp", type="adc")
        new = type(int32ts).read(tmpf, "test", type="adc")
    assert new.dtype == numpy.dtype("float64")
    assert_quantity_sub_equal(int32ts, new)
