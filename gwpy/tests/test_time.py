# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Tests for :mod:`gwpy.time`
"""

from datetime import datetime

import pytest

from freezegun import freeze_time

from astropy.time import Time
from astropy.units import (UnitConversionError, Quantity)

from glue.lal import LIGOTimeGPS as GlueGPS

from gwpy import time

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_to_gps():
    """Test :func:`gwpy.time.to_gps`
    """
    # str conversion
    t = time.to_gps('Jan 1 2017')
    assert isinstance(t, time.LIGOTimeGPS)
    assert t == 1167264018
    assert time.to_gps('Sep 14 2015 09:50:45.391') == (
        time.LIGOTimeGPS(1126259462, 391000000))
    # datetime conversion
    assert time.to_gps(datetime(2017, 1, 1)) == 1167264018

    # astropy.time.Time conversion
    assert time.to_gps(Time(57754, format='mjd')) == 1167264018

    # tuple
    assert time.to_gps((2017, 1, 1)) == 1167264018

    # Quantity
    assert time.to_gps(Quantity(1167264018, 's')) == 1167264018

    # keywords
    with freeze_time('2015-09-14 09:50:45.391'):
        assert time.to_gps('now') == 1126259462
        assert time.to_gps('today') == 1126224017
        assert time.to_gps('tomorrow') == 1126310417
        assert time.to_gps('yesterday') == 1126137617

    # errors
    with pytest.raises(UnitConversionError):
        time.to_gps(Quantity(1, 'm'))
    with pytest.raises((ValueError, TypeError)) as exc:
        time.to_gps('random string')
    assert 'Cannot parse date string \'random string\': ' in str(exc.value)


def test_from_gps():
    """Test :func:`gwpy.time.from_gps`
   """
    # basic
    d = time.from_gps(1167264018)
    assert isinstance(d, datetime)
    assert d == datetime(2017, 1, 1)

    # str
    assert time.from_gps('1167264018') == datetime(2017, 1, 1)

    # float
    assert time.from_gps(1126259462.391) == (
        datetime(2015, 9, 14, 9, 50, 45, 391000))
    assert time.from_gps('1.13e9') == datetime(2015, 10, 27, 16, 53, 3)

    # errors
    with pytest.raises((RuntimeError, ValueError)):
        time.from_gps('test')


def test_tconvert():
    """Test :func:`gwpy.time.tconvert`
    """
    # from GPS
    assert time.tconvert(1126259462.391) == (
        datetime(2015, 9, 14, 9, 50, 45, 391000))

    # from GPS using LAL LIGOTimeGPS
    assert time.tconvert(time.LIGOTimeGPS(1126259462.391)) == (
        datetime(2015, 9, 14, 9, 50, 45, 391000))
    assert time.tconvert(GlueGPS(1126259462.391)) == (
        datetime(2015, 9, 14, 9, 50, 45, 391000))

    # to GPS
    assert time.tconvert(datetime(2015, 9, 14, 9, 50, 45, 391000)) == (
        time.LIGOTimeGPS(1126259462, 391000000))

    # special cases
    now = time.tconvert()
    now2 = time.tconvert('now')
    assert now == now2
    today = float(time.tconvert('today'))
    yesterday = float(time.tconvert('yesterday'))
    assert today - yesterday == pytest.approx(86400)
    assert now >= today
    tomorrow = float(time.tconvert('tomorrow'))
    assert tomorrow - today == pytest.approx(86400)
