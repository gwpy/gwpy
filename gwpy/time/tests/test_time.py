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

"""Tests for :mod:`gwpy.time`."""

from datetime import (
    UTC,
    datetime,
)
from decimal import Decimal

import numpy
import pytest
from astropy.time import Time
from astropy.units import (
    Quantity,
    UnitConversionError,
)

from ... import time
from .. import LIGOTimeGPS

try:
    from glue.lal import LIGOTimeGPS as GlueGPS
except ImportError:
    GlueGPS = LIGOTimeGPS

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

GW150914 = LIGOTimeGPS(1126259462, 391000000)
GW150914_DT = datetime(2015, 9, 14, 9, 50, 45, 391000, tzinfo=UTC)
FREEZE = "2015-09-14 09:50:45.391"
NOW = 1126259462
TODAY = 1126224017
TOMORROW = 1126310417
YESTERDAY = 1126137617


@pytest.mark.freeze_time(FREEZE)
@pytest.mark.parametrize(("in_", "out"), [
    (1126259462, int(GW150914)),
    (1235635623.7500002, LIGOTimeGPS(1235635623, 750000200)),
    (LIGOTimeGPS(1126259462, 391000000), GW150914),
    ("0", 0),
    ("Jan 1 2017", 1167264018),
    ("Sep 14 2015 09:50:45.391", GW150914),
    ("Oct 30 2016 12:34 CST", 1161887657),
    ((2017, 1, 1), 1167264018),
    (datetime(2017, 1, 1, tzinfo=UTC), 1167264018),
    (Time(57754, format="mjd"), 1167264018),
    (Time(57754.0001, format="mjd"), LIGOTimeGPS(1167264026, 640000000)),
    (Quantity(1167264018, "s"), 1167264018),
    (Decimal("1126259462.391000000"), GW150914),
    pytest.param(
        GlueGPS(GW150914.gpsSeconds, GW150914.gpsNanoSeconds),
        GW150914,
        marks=pytest.mark.requires("glue"),
    ),
    (numpy.int32(NOW), NOW),  # fails with lal-6.18.0
    ("now", NOW),
    ("today", TODAY),
    ("tomorrow", TOMORROW),
    ("yesterday", YESTERDAY),
])
def test_to_gps(in_, out):
    """Test that :func:`to_gps` works."""
    assert time.to_gps(in_) == out


@pytest.mark.parametrize(("in_", "err"), [
    (Quantity(1, "m"), UnitConversionError),
    ("random string", ValueError),
])
def test_to_gps_error(in_, err):
    """Test that :func:`gwpy.time.to_gps` errors when it should."""
    with pytest.raises(err):
        time.to_gps(in_)


@pytest.mark.parametrize(("in_", "out"), [
    (1167264018, datetime(2017, 1, 1, tzinfo=UTC)),
    ("1167264018", datetime(2017, 1, 1, tzinfo=UTC)),
    (1126259462.391, datetime(2015, 9, 14, 9, 50, 45, 391000, tzinfo=UTC)),
    ("1.13e9", datetime(2015, 10, 27, 16, 53, 3, tzinfo=UTC)),
    pytest.param(
        GlueGPS(GW150914.gpsSeconds, GW150914.gpsNanoSeconds),
        GW150914_DT,
        marks=pytest.mark.requires("glue"),
    ),
])
def test_from_gps(in_, out):
    """Test that :func:`gwpy.time.from_gps` works."""
    assert time.from_gps(in_) == out


@pytest.mark.parametrize(("in_", "err"), [
    ("test", ValueError),
    (1167264017, ValueError),  # gwpy/gwpy#1021
])
def test_from_gps_error(in_, err):
    """Test that :func:`gwpy.time.from_gps` errors when it should."""
    with pytest.raises(err):
        time.from_gps(in_)


@pytest.mark.freeze_time(FREEZE)
@pytest.mark.parametrize(("in_", "out"), [
    (float(GW150914), GW150914_DT),
    (GW150914, GW150914_DT),
    (GW150914_DT, GW150914),
    pytest.param(
        GlueGPS(float(GW150914)),
        GW150914_DT,
        marks=pytest.mark.requires("glue"),
    ),
    ("now", NOW),
    ("today", TODAY),
    ("tomorrow", TOMORROW),
    ("yesterday", YESTERDAY),
])
def test_tconvert(in_, out):
    """Test :func:`gwpy.time.tconvert`."""
    assert time.tconvert(in_) == out


@pytest.mark.parametrize("gpsmod", [
    "glue.lal",
    "lal",
    "ligotimegps",
])
def test_gps_types(gpsmod):
    """Test that the module's `LIGOTimeGPS` matches the protocol."""
    mod = pytest.importorskip(gpsmod)
    gps = mod.LIGOTimeGPS(123, 456000000)
    assert isinstance(gps, time.LIGOTimeGPSLike)
    assert gps.gpsSeconds == 123
    assert gps.gpsNanoSeconds == 456000000
    assert gps == 123.456
