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
import sys
import tempfile

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

import numpy
from numpy import testing as nptest

from astropy import units

from ..time import Time

from .. import version
from ..timeseries import (TimeSeries, StateVector)
from .array import SeriesTestCase

SEED = 1
GPS_EPOCH = Time(0, format='gps', scale='utc')
ONE_HZ = units.Quantity(1, 'Hz')
ONE_SECOND = units.Quantity(1, 'second')

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


# -----------------------------------------------------------------------------

class TimeSeriesTestCase(SeriesTestCase):
    """`~unittest.TestCase` for the `~gwpy.timeseries.TimeSeries` class
    """
    channel = 'L1:LDAS-STRAIN'
    framefile = os.path.join(os.path.split(__file__)[0], 'data',
                             'HLV-GW100916-968654552-1.gwf')
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_')
    TEST_CLASS = TimeSeries

    def test_creation_with_metadata(self):
        self.ts = self.create()
        repr(self.ts)
        self.assertTrue(self.ts.epoch == GPS_EPOCH)
        self.assertTrue(self.ts.sample_rate == ONE_HZ)
        self.assertTrue(self.ts.dt == ONE_SECOND)

    def frame_read(self, format=None):
        ts = self.TEST_CLASS.read(
            self.framefile, self.channel, format=format)
        self.assertTrue(ts.epoch == Time(968654552, format='gps', scale='utc'))
        self.assertTrue(ts.sample_rate == units.Quantity(16384, 'Hz'))
        self.assertTrue(ts.unit == units.Unit('strain'))

    def test_epoch(self):
        array = self.create()
        self.assertEquals(array.epoch.gps, array.x0.value)

    def test_frame_read_lalframe(self):
        try:
            self.frame_read(format='lalframe')
        except ImportError as e:
            self.skipTest(str(e))

    def test_frame_read_framecpp(self):
        try:
            self.frame_read(format='framecpp')
        except ImportError as e:
            self.skipTest(str(e))

    def test_ascii_write(self, delete=True):
        self.ts = self.create()
        asciiout = self.tmpfile % 'txt'
        self.ts.write(asciiout)
        if delete and os.path.isfile(asciiout):
            os.remove(asciiout)
        return asciiout

    def test_ascii_read(self):
        fp = self.test_ascii_write(delete=False)
        try:
            self.TEST_CLASS.read(fp)
        finally:
            if os.path.isfile(fp):
                os.remove(fp)

    def test_hdf5_write(self, delete=True):
        self.ts = self.create(name=self.channel)
        hdfout = self.tmpfile % 'hdf'
        try:
            self.ts.write(hdfout)
        except ImportError as e:
            self.skipTest(str(e))
        finally:
            if delete and os.path.isfile(hdfout):
                os.remove(hdfout)
        return hdfout

    def test_hdf5_read(self):
        try:
            hdfout = self.test_hdf5_write(delete=False)
        except ImportError as e:
            self.skipTest(str(e))
        else:
            try:
                self.TEST_CLASS.read(hdfout, self.channel)
            finally:
                if os.path.isfile(hdfout):
                    os.remove(hdfout)

    def test_resample(self):
        """Test the `TimeSeries.resample` method
        """
        ts1 = self.create(sample_rate=100)
        ts2 = ts1.resample(10)
        self.assertEquals(ts2.sample_rate, ONE_HZ*10)

    def test_to_from_lal(self):
        ts = self.create()
        try:
            lalts = ts.to_lal()
        except (NotImplementedError, ImportError) as e:
            self.skipTest(str(e))
        ts2 = type(ts).from_lal(lalts)
        self.assertEqual(ts, ts2)


# -----------------------------------------------------------------------------

class StateVectorTestCase(TimeSeriesTestCase):
    """`~unittest.TestCase` for the `~gwpy.timeseries.StateVector` object
    """
    TEST_CLASS = StateVector

    def setUp(self):
        super(StateVectorTestCase, self).setUp(dtype='uint32')


if __name__ == '__main__':
    unittest.main()
