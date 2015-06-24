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

from numpy import (random, testing as nptest)

from astropy import units

from gwpy.time import Time

from gwpy import version
from gwpy.timeseries.core import TimeSeries

SEED = 1
GPS_EPOCH = Time(0, format='gps', scale='utc')
ONE_HZ = units.Quantity(1, 'Hz')
ONE_SECOND = units.Quantity(1, 'second')

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


# -----------------------------------------------------------------------------

class TimeSeriesTests(unittest.TestCase):
    """`~unittest.TestCase` for the `~gwpy.timeseries.TimeSeries` class
    """
    channel = 'L1:LDAS-STRAIN'
    framefile = os.path.join(os.path.split(__file__)[0], 'data',
                             'HLV-GW100916-968654552-1.gwf')
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_')
    SeriesClass = TimeSeries

    def setUp(self):
        random.seed(SEED)
        self.data = random.random(100)
        self._ts = None

    def create_with_random_data(self, sample_rate=1, name='TEST CASE',
                                epoch=0, channel='TEST CASE', copy=False,
                                **kwargs):
        return self.SeriesClass(self.data, sample_rate=sample_rate, name=name,
                                epoch=epoch, channel=channel, **kwargs)

    def get_test_series(self):
        if self._ts is None:
            self._ts = self.create_with_random_data()
        return self._ts

    def test_creation(self):
        self.SeriesClass(self.data)

    def test_creation_with_metadata(self):
        self.ts = self.get_test_series()
        repr(self.ts)
        self.assertTrue(self.ts.epoch == GPS_EPOCH)
        self.assertTrue(self.ts.sample_rate == ONE_HZ)
        self.assertTrue(self.ts.dt == ONE_SECOND)

    def frame_read(self, format=None):
        ts = self.SeriesClass.read(
            self.framefile, self.channel, format=format)
        self.assertTrue(ts.epoch == Time(968654552, format='gps', scale='utc'))
        self.assertTrue(ts.sample_rate == units.Quantity(16384, 'Hz'))
        self.assertTrue(ts.unit == units.Unit('strain'))

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
        self.ts = self.get_test_series()
        asciiout = self.tmpfile % 'txt'
        self.ts.write(asciiout)
        if delete and os.path.isfile(asciiout):
            os.remove(asciiout)
        return asciiout

    def test_ascii_read(self):
        fp = self.test_ascii_write(delete=False)
        try:
            ts = self.SeriesClass.read(fp)
        finally:
            if os.path.isfile(fp):
                os.remove(fp)

    def test_hdf5_write(self, delete=True):
        self.ts = self.get_test_series()
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
                ts = self.SeriesClass.read(hdfout, 'TEST CASE')
            finally:
                if os.path.isfile(hdfout):
                    os.remove(hdfout)

    def test_pickle(self):
        """Check pickle-unpickle yields unchanged data
        """
        import cPickle
        ts = self.get_test_series()
        pickle = ts.dumps()
        ts2 = cPickle.loads(pickle)
        self.assertEqual(ts, ts2)

    def test_crop(self):
        """Test cropping `TimeSeries` by GPS times
        """
        ts = self.get_test_series()
        ts2 = ts.crop(10, 20)
        self.assertAlmostEqual(ts2.epoch.gps, 10)
        self.assertEqual(ts2.x0.value, 10)
        self.assertEqual(ts2.span[1], 20)
        nptest.assert_array_equal(ts2.data, ts.data[10:20])

    def test_is_compatible(self):
        """Test the `TimeSeries.is_compatible` method
        """
        ts1 = self.get_test_series()
        ts2 = self.create_with_random_data(name='TEST CASE 2')
        self.assertTrue(ts1.is_compatible(ts2))
        ts3 = self.create_with_random_data(sample_rate=2)
        self.assertRaises(ValueError, ts1.is_compatible, ts3)
        ts4 = self.create_with_random_data(unit='m')
        self.assertRaises(ValueError, ts1.is_compatible, ts4)

    def test_is_contiguous(self):
        """Test the `TimeSeries.is_contiguous` method
        """
        ts1 = self.get_test_series()
        ts2 = self.create_with_random_data(epoch=ts1.span[1])
        self.assertEquals(ts1.is_contiguous(ts2), 1)
        self.assertEquals(ts1.is_contiguous(ts2.value), 1)
        ts3 = self.create_with_random_data(epoch=ts1.span[1]+1)
        self.assertEquals(ts1.is_contiguous(ts3), 0)
        ts4 = self.create_with_random_data(epoch=-ts1.span[1])
        self.assertEquals(ts1.is_contiguous(ts4), -1)

    def test_append(self):
        """Test the `TimeSeries.append` method
        """
        ts1 = self.get_test_series()
        ts2 = self.create_with_random_data(epoch=ts1.span[1])
        ts3 = ts1.append(ts2, inplace=False)
        self.assertEquals(ts3.epoch, ts1.epoch)
        self.assertEquals(ts3.size, ts1.size+ts2.size)
        self.assertEquals(ts3.span, ts1.span+ts2.span)
        self.assertRaises(ValueError, ts3.append, ts1)

    def test_prepend(self):
        """Test the `TimeSeries.prepend` method
        """
        ts1 = self.get_test_series()
        ts2 = self.create_with_random_data(epoch=ts1.span[1])
        ts3 = ts2.prepend(ts1, inplace=False)
        self.assertEquals(ts3.epoch, ts1.epoch)
        self.assertEquals(ts3.size, ts1.size+ts2.size)
        self.assertEquals(ts3.span, ts1.span+ts2.span)
        self.assertRaises(ValueError, ts3.prepend, ts1)

    def test_update(self):
        """Test the `TimeSeries.update method`
        """
        ts1 = self.get_test_series()
        ts2 = self.create_with_random_data(epoch=ts1.span[1])[:ts1.size//2]
        ts3 = ts1.update(ts2, inplace=False)
        self.assertEquals(ts3.x0, ts1.x0 + ts2.duration)
        self.assertEquals(ts3.size, ts1.size)
        self.assertRaises(ValueError, ts3.update, ts1)


if __name__ == '__main__':
    unittest.main()
