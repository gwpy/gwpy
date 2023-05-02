# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
#
# This file is part of PDpy.
#
# PDpy is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PDpy is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PDpy.  If not, see <http://www.gnu.org/licenses/>.

"""Tests for :mod:`pdpy.time`
"""

from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
from operator import attrgetter
from unittest import mock

import pytest

import numpy

from astropy.time import Time
from astropy.units import (UnitConversionError, Quantity)

from ... import time
from .. import LIGOTimeGPS

try:
    from glue.lal import LIGOTimeGPS as GlueGPS
except ImportError:
    GlueGPS = LIGOTimeGPS
    HAS_GLUE = False
else:
    HAS_GLUE = True

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

GW150914 = LIGOTimeGPS(1126259462, 391000000)
GW150914_DT = datetime(2015, 9, 14, 9, 50, 45, 391000)
FREEZE = '2015-09-14 09:50:45.391'
NOW = 1126259462
TODAY = 1126224017
TOMORROW = 1126310417
YESTERDAY = 1126137617


@pytest.mark.freeze_time(FREEZE)
@pytest.mark.parametrize("use_maya", (
    False,
    pytest.param(True, marks=pytest.mark.requires("maya")),
))
@pytest.mark.parametrize(("in_", "out"), [
    (1126259462, int(GW150914)),
    (1235635623.7500002, LIGOTimeGPS(1235635623, 750000200)),
    (LIGOTimeGPS(1126259462, 391000000), GW150914),
    ('0', 0),
    ('Jan 1 2017', 1167264018),
    ('Sep 14 2015 09:50:45.391', GW150914),
    ((2017, 1, 1), 1167264018),
    (datetime(2017, 1, 1), 1167264018),
    (Time(57754, format='mjd'), 1167264018),
    (Time(57754.0001, format='mjd'), LIGOTimeGPS(1167264026, 640000000)),
    (Quantity(1167264018, 's'), 1167264018),
    (Decimal('1126259462.391000000'), GW150914),
    pytest.param(
        GlueGPS(GW150914.gpsSeconds, GW150914.gpsNanoSeconds),
        GW150914,
        marks=pytest.mark.requires("glue"),
    ),
    (numpy.int32(NOW), NOW),  # fails with lal-6.18.0
    ('now', NOW),
    ('today', TODAY),
    ('tomorrow', TOMORROW),
    ('yesterday', YESTERDAY),
])
def test_to_gps(in_, out, use_maya):
    """Test that :func:`to_gps` works with and without maya.
    """
    if use_maya:  # do nothing
        ctx = nullcontext()
    else:  # force 'install maya' to error to use dateutil
        ctx = mock.patch.dict("sys.modules", {"maya": None})
    with ctx:
        assert time.to_gps(in_) == out


@pytest.mark.requires("maya")
@pytest.mark.parametrize(("in_", "out"), [
    ('Oct 30 2016 12:34 CST', 1161887657),
])
def test_to_gps_maya(in_, out):
    """Test that :func:`pdpy.time.to_gps` works with maya.
    """
    assert time.to_gps(in_) == out


@pytest.mark.parametrize(("in_", "err"), [
    (Quantity(1, 'm'), UnitConversionError),
    ('random string', (ValueError, TypeError)),
])
def test_to_gps_error(in_, err):
    """Test that :func:`pdpy.time.to_gps` errors when it should.
    """
    with pytest.raises(err):
        time.to_gps(in_)


@mock.patch.dict("sys.modules", {"maya": None})
def test_to_gps_dateparser_error_propagation_nomaya():
    with pytest.raises(
        RuntimeWarning,
        match=(
            "tzname CST identified but not understood.(.*) "
            "Try installing 'maya' (.*)"
        ),
    ):
        time.to_gps("Oct 30 2016 12:34 CST")


@pytest.mark.requires("maya")
@mock.patch("maya.when", side_effect=ValueError("test chain"))
def test_to_gps_dateparser_error_propagation_maya(_):
    with pytest.raises(RuntimeWarning) as exc:
        time.to_gps("Oct 30 2016 12:34 CST")
    # validate that the maya exception was chained to the dateutil one
    assert str(exc.value.__cause__) == "test chain"


@pytest.mark.parametrize('in_, out', [
    (1167264018, datetime(2017, 1, 1)),
    ('1167264018', datetime(2017, 1, 1)),
    (1126259462.391, datetime(2015, 9, 14, 9, 50, 45, 391000)),
    ('1.13e9', datetime(2015, 10, 27, 16, 53, 3)),
    pytest.param(
        GlueGPS(GW150914.gpsSeconds, GW150914.gpsNanoSeconds),
        GW150914_DT,
        marks=pytest.mark.requires("glue"),
    ),
])
def test_from_gps(in_, out):
    """Test that :func:`pdpy.time.from_gps` works.
    """
    assert time.from_gps(in_) == out


@pytest.mark.parametrize(("in_", "err"), [
    ('test', ValueError),
    (1167264017, ValueError),  # pdpy/pdpy#1021
])
def test_from_gps_error(in_, err):
    """Test that :func:`pdpy.time.from_gps` errors when it should.
    """
    with pytest.raises(err):
        time.from_gps(in_)


@pytest.mark.freeze_time(FREEZE)
@pytest.mark.parametrize('in_, out', [
    (float(GW150914), GW150914_DT),
    (GW150914, GW150914_DT),
    (GW150914_DT, GW150914),
    pytest.param(
        GlueGPS(float(GW150914)),
        GW150914_DT,
        marks=pytest.mark.requires("glue"),
    ),
    ('now', NOW),
    ('today', TODAY),
    ('tomorrow', TOMORROW),
    ('yesterday', YESTERDAY),
])
def test_tconvert(in_, out):
    """Test :func:`pdpy.time.tconvert`
    """
    assert time.tconvert(in_) == out


@pytest.mark.parametrize('gpstype', time.gps_types,
                         ids=attrgetter('__module__'))
def test_gps_types(gpstype):
    assert gpstype.__name__ == 'LIGOTimeGPS'
    gps = gpstype(123, 456000000)
    assert gps.gpsSeconds == 123
    assert gps.gpsNanoSeconds == 456000000
    assert gps == 123.456
