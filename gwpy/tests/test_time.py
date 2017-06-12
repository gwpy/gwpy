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

from freezegun import freeze_time

from compat import (unittest, mock)

from astropy.time import Time
from astropy.units import (UnitConversionError, Quantity)

from glue.lal import LIGOTimeGPS as GlueGPS

from gwpy import time

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class TimeTests(unittest.TestCase):
    """`TestCase` for the time module
    """
    def test_to_gps(self):
        # str conversion
        t = time.to_gps('Jan 1 2017')
        self.assertIsInstance(t, time.LIGOTimeGPS)
        self.assertEqual(t, 1167264018)
        self.assertEqual(time.to_gps('Sep 14 2015 09:50:45.391'),
                         time.LIGOTimeGPS(1126259462, 391000000))

        # datetime conversion
        self.assertEqual(time.to_gps(datetime.datetime(2017, 1, 1)), 1167264018)

        # astropy.time.Time conversion
        self.assertEqual(time.to_gps(Time(57754, format='mjd')), 1167264018)

        # tuple
        self.assertEqual(time.to_gps((2017, 1, 1)), 1167264018)

        # Quantity
        self.assertEqual(time.to_gps(Quantity(1167264018, 's')), 1167264018)

        # keywords
        with freeze_time('2015-09-14 09:50:45.391'):
            self.assertEqual(time.to_gps('now'), 1126259462)
            self.assertEqual(time.to_gps('today'), 1126224017)
            self.assertEqual(time.to_gps('tomorrow'), 1126310417)
            self.assertEqual(time.to_gps('yesterday'), 1126137617)

        # errors
        self.assertRaises(UnitConversionError, time.to_gps, Quantity(1, 'm'))
        with self.assertRaises(ValueError) as exc:
            time.to_gps('random string')
        self.assertIn('Cannot parse date string \'random string\': ',
                      str(exc.exception))

    def test_from_gps(self):
        # basic
        d = time.from_gps(1167264018)
        self.assertIsInstance(d, datetime.datetime)
        self.assertEqual(d, datetime.datetime(2017, 1, 1))

        # str
        self.assertEqual(time.from_gps('1167264018'),
                         datetime.datetime(2017, 1, 1))

        # float
        self.assertEqual(time.from_gps(1126259462.391),
                         datetime.datetime(2015, 9, 14, 9, 50, 45, 391000))
        self.assertEqual(time.from_gps('1.13e9'),
                         datetime.datetime(2015, 10, 27, 16, 53, 3))

        # errors
        self.assertRaises((RuntimeError, ValueError), time.from_gps, 'test')

    def test_tconvert(self):
        # from GPS
        self.assertEqual(time.tconvert(1126259462.391),
                         datetime.datetime(2015, 9, 14, 9, 50, 45, 391000))

        # from GPS using LAL LIGOTimeGPS
        self.assertEqual(time.tconvert(time.LIGOTimeGPS(1126259462.391)),
                         datetime.datetime(2015, 9, 14, 9, 50, 45, 391000))
        self.assertEqual(time.tconvert(GlueGPS(1126259462.391)),
                         datetime.datetime(2015, 9, 14, 9, 50, 45, 391000))

        # to GPS
        self.assertEqual(
            time.tconvert(datetime.datetime(2015, 9, 14, 9, 50, 45, 391000)),
            time.LIGOTimeGPS(1126259462, 391000000))

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
