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

"""Unit test for timeseries module
"""

import os
import os.path
import unittest
import tempfile

from numpy import random

from astropy import units

from gwpy.time import Time

from gwpy import version
from gwpy.timeseries import TimeSeries

SEED = 1
GPS_EPOCH = Time(0, format='gps', scale='utc')
ONE_HZ = units.Quantity(1, 'Hz')
ONE_SECOND = units.Quantity(1, 'second')

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class TimeSeriesTests(unittest.TestCase):
    """`TestCase` for the timeseries module
    """
    framefile = os.path.join(os.path.split(__file__)[0], 'data',
                             'HLV-GW100916-968654552-1.gwf')

    def setUp(self):
        random.seed(SEED)
        self.data = random.random(100)

    def test_creation(self):
        TimeSeries(self.data)

    def test_creation_with_metadata(self):
        self.ts = TimeSeries(self.data, sample_rate=1, name='TEST CASE',
                             epoch=0, channel='TEST CASE')
        repr(self.ts)
        self.assertTrue(self.ts.epoch == GPS_EPOCH)
        self.assertTrue(self.ts.sample_rate == ONE_HZ)
        self.assertTrue(self.ts.dt == ONE_SECOND)

    def frame_read(self, format=None):
        ts = TimeSeries.read(self.framefile, 'L1:LDAS-STRAIN', format=format)
        self.assertTrue(ts.epoch == Time(968654552, format='gps',
                                              scale='utc'))
        self.assertTrue(ts.sample_rate == units.Quantity(16384, 'Hz'))
        self.assertTrue(ts.unit == units.Unit('strain'))

    def test_frame_read_lalframe(self):
        try:
            self.frame_read(format='lalframe')
        except ImportError as e:
            raise unittest.SkipTest(str(e))

    def test_frame_read_framecpp(self):
        try:
            self.frame_read(format='framecpp')
        except ImportError as e:
            raise unittest.SkipTest(str(e))

    def test_hdf5_write(self, delete=True):
        self.ts = TimeSeries(self.data, sample_rate=1, name='TEST CASE',
                             epoch=0, channel='TEST CASE')
        hdfout = tempfile.mktemp(prefix='gwpy_test_')
        try:
            self.ts.write(hdfout, format='hdf')
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        finally:
            if delete and os.path.isfile(hdfout):
                os.remove(hdfout)
        return hdfout

    def test_hdf5_read(self):
        try:
            hdfout = self.test_hdf5_write(delete=False)
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        else:
            try:
                ts = TimeSeries.read(hdfout, 'TEST CASE', format='hdf')
            finally:
                if os.path.isfile(hdfout):
                    os.remove(hdfout)


if __name__ == '__main__':
    unittest.main()
