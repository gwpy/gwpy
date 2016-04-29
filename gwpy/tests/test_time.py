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

"""Unit test for the time module
"""

import datetime

from compat import unittest

from astropy.units import (UnitConversionError, Quantity)

from gwpy import time

DATE = datetime.datetime(2000, 1, 1, 0, 0)
GPS = 630720013

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class TimeTests(unittest.TestCase):
    """`TestCase` for the time module
    """
    def test_to_gps(self):
        # test datetime conversion
        self.assertEqual(time.to_gps(DATE), GPS)
        # test Time
        self.assertEqual(
            time.to_gps(DATE, format='datetime', scale='utc'), GPS)
        # test tuple
        self.assertEqual(time.to_gps(tuple(DATE.timetuple())[:6]), GPS)
        # test Quantity
        self.assertEqual(time.to_gps(Quantity(GPS, 's')), GPS)
        # test errors
        self.assertRaises(UnitConversionError, time.to_gps, Quantity(1, 'm'))
        self.assertRaises(ValueError, time.to_gps, 'random string')

    def test_from_gps(self):
        date = time.from_gps(GPS)
        self.assertEqual(date, DATE)

    def test_tconvert(self):
        # from GPS
        date = time.tconvert(GPS)
        self.assertEqual(date, DATE)
        # from GPS using LAL LIGOTimeGPS
        try:
            from lal import LIGOTimeGPS
        except ImportError:
            pass
        else:
            d = time.tconvert(LIGOTimeGPS(GPS))
            self.assertEqual(d, DATE)
        # to GPS
        gps = time.tconvert(date)
        self.assertEqual(gps, GPS)
        # special cases
        now = time.tconvert()
        now2 = time.tconvert('now')
        self.assertEqual(now, now2)
        today = time.tconvert('today')
        yesterday = time.tconvert('yesterday')
        self.assertAlmostEqual(today - yesterday, 86400)
        self.assertTrue(now >= today)
        tomorrow = time.tconvert('tomorrow')
        self.assertAlmostEqual(tomorrow - today, 86400)


if __name__ == '__main__':
    unittest.main()
