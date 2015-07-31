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

from gwpy import time

from gwpy import version

DATE = datetime.datetime(2000, 1, 1, 0, 0)
GPS = 630720013

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class TimeTests(unittest.TestCase):
    """`TestCase` for the time module
    """
    def test_to_gps(self):
        gps = time.to_gps(DATE)
        self.assertEqual(gps, GPS)

    def test_from_gps(self):
        date = time.from_gps(GPS)
        self.assertEqual(date, DATE)

    def test_tconvert(self):
        # from GPS
        date = time.tconvert(GPS)
        self.assertEqual(date, DATE)
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


if __name__ == '__main__':
    unittest.main()
