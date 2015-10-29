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

"""Unit test for data classes
"""

import abc
import os
import tempfile

from compat import unittest

from numpy import testing as nptest
import numpy

from astropy import units
from astropy.time import Time

from gwpy import version
from gwpy.data import (Array, Series)
from gwpy.detector import Channel

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

SEED = 1
GPS_EPOCH = 12345
TIME_EPOCH = Time(12345, format='gps', scale='utc')
CHANNEL_NAME = 'X1:TEST-CHANNEL'
CHANNEL = Channel(CHANNEL_NAME)


class CommonTests(object):
    __metaclass_ = abc.ABCMeta
    TEST_CLASS = Array
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_')

    def setUp(self, dtype=None):
        numpy.random.seed(SEED)
        self.data = (numpy.random.random(100) * 1e5).astype(dtype=dtype)
        self.datasq = self.data ** 2

    @property
    def TEST_ARRAY(self):
        try:
            return self._TEST_ARRAY
        except AttributeError:
            self._TEST_ARRAY = self.create(name='test', unit='meter',
                                           channel=CHANNEL_NAME,
                                           epoch=TIME_EPOCH)
            return self.TEST_ARRAY

    def create(self, *args, **kwargs):
        kwargs.setdefault('copy', False)
        return self.TEST_CLASS(self.data, *args, **kwargs)

    def assertArraysEqual(self, ts1, ts2, *args):
        nptest.assert_array_equal(ts1.value, ts2.value)
        if not args:
            args = ['units'] + self.TEST_CLASS._metadata_slots
        for attr in args:
            self.assertEqual(getattr(ts1, attr, None),
                             getattr(ts2, attr, None))

    # -- test methods ---------------------------

    def test_init(self):
        """Test Array creation
        """
        # test basic empty contructor
        self.assertRaises(TypeError, self.TEST_CLASS)
        self.assertRaises(IndexError, self.TEST_CLASS, [])
        # test with some data
        array = self.create()
        nptest.assert_array_equal(array.value, self.data)

    def test_unit(self):
        array = self.create()
        self.assertIsNone(array.unit)
        array = self.create(unit='m')
        self.assertEquals(array.unit, units.m)

    def test_name(self):
        array = self.create(name=None)
        self.assertIsNone(array.name)
        array = self.create(name='TEST CASE')
        self.assertEquals(array.name, 'TEST CASE')

    def test_epoch(self):
        array = self.create()
        self.assertIsNone(array.epoch)
        array = self.create(epoch=GPS_EPOCH)
        self.assertEquals(array.epoch, TIME_EPOCH)

    def test_channel(self):
        array = self.create()
        self.assertIsNone(array.channel)
        array = self.create(channel=CHANNEL_NAME)
        self.assertEquals(array.channel, CHANNEL)

    def test_math(self):
        """Test Array math operations
        """
        array = self.create(unit='Hz')
        # test basic operations
        arraysq = array ** 2
        nptest.assert_array_equal(arraysq.value, self.datasq)
        self.assertEqual(arraysq.unit, units.Hz ** 2)
        self.assertEqual(arraysq.name, array.name)
        self.assertEqual(arraysq.epoch, array.epoch)
        self.assertEqual(arraysq.channel, array.channel)

    def test_copy(self):
        """Test Array.copy
        """
        array = self.create(unit='Hz')
        array2 = array.copy()
        nptest.assert_array_equal(array.value, array2.value)
        self.assertEqual(array.unit, array2.unit)

    # -- test I/O -------------------------------

    def test_hdf5_write(self, delete=True, format=[None, 'hdf5', 'hdf']):
        if isinstance(format, (list, tuple)):
            formats = format
        else:
            formats = [format]
        hdfout = self.tmpfile % 'hdf'
        for format in formats:
            try:
                self.TEST_ARRAY.write(hdfout, format=format)
            except ImportError as e:
                self.skipTest(str(e))
            finally:
                if delete and os.path.isfile(hdfout):
                    os.remove(hdfout)
        return hdfout

    def test_hdf5_read(self, format=[None, 'hdf5', 'hdf']):
        try:
            hdfout = self.test_hdf5_write(delete=False, format='hdf5')
        except ImportError as e:
            self.skipTest(str(e))
        else:
            if isinstance(format, (list, tuple)):
                formats = format
            else:
                formats = [format]
            try:
                for format in formats:
                    ts = self.TEST_CLASS.read(hdfout, format=format)
                    self.assertArraysEqual(self.TEST_ARRAY, ts)
            finally:
                if os.path.isfile(hdfout):
                    os.remove(hdfout)


class ArrayTestCase(CommonTests, unittest.TestCase):
    pass


class SeriesTestCase(CommonTests, unittest.TestCase):
    TEST_CLASS = Series

    def test_init(self):
        super(SeriesTestCase, self).test_init()
        series = self.create(x0=0, dx=1)
        self.assertEqual(series.x0, units.Quantity(0, series._default_xunit))
        self.assertEqual(series.dx, units.Quantity(1, series._default_xunit))

    def test_xunit(self, unit=None):
        if unit is None:
            unit = self.TEST_CLASS._default_xunit
        series = self.create(unit='Hz', dx=4*unit)
        self.assertEqual(series.x0, 0*unit)
        self.assertEqual(series.dx, 4*unit)
        # for series only, test arbitrary xunit
        if self.TEST_CLASS == Series:
            series = self.create(unit='Hz', dx=4*units.m)
            self.assertEqual(series.x0, 0*units.m)
            self.assertEqual(series.dx, 4*units.m)

    def test_index(self):
        series = self.create()
        self.assertFalse(hasattr(series, '_xindex'))
        nptest.assert_array_equal(
            series.xindex, numpy.arange(series.size) * series.dx + series.x0)

    def test_pickle(self):
        """Check pickle-unpickle yields unchanged data
        """
        import cPickle
        ts = self.create()
        pickle = ts.dumps()
        ts2 = cPickle.loads(pickle)
        self.assertArraysEqual(ts, ts2)

    def test_crop(self):
        """Test cropping `Series` by GPS times
        """
        ts = self.create()
        ts2 = ts.crop(10, 20)
        self.assertEqual(ts2.x0.value, 10)
        self.assertEqual(ts2.xspan[1], 20)
        nptest.assert_array_equal(ts2.value, ts.value[10:20])

    def test_is_compatible(self):
        """Test the `Series.is_compatible` method
        """
        ts1 = self.create()
        ts2 = self.create(name='TEST CASE 2')
        self.assertTrue(ts1.is_compatible(ts2))
        ts3 = self.create(dx=2)
        self.assertRaises(ValueError, ts1.is_compatible, ts3)
        ts4 = self.create(unit='m')
        self.assertRaises(ValueError, ts1.is_compatible, ts4)

    def test_is_contiguous(self):
        """Test the `Series.is_contiguous` method
        """
        ts1 = self.create()
        ts2 = self.create(x0=ts1.xspan[1])
        self.assertEquals(ts1.is_contiguous(ts2), 1)
        self.assertEquals(ts1.is_contiguous(ts2.value), 1)
        ts3 = self.create(x0=ts1.xspan[1]+1)
        self.assertEquals(ts1.is_contiguous(ts3), 0)
        ts4 = self.create(x0=-ts1.xspan[1])
        self.assertEquals(ts1.is_contiguous(ts4), -1)

    def test_append(self):
        """Test the `Series.append` method
        """
        # create arrays
        ts1 = self.create()
        ts2 = self.create(x0=ts1.xspan[1])
        # test basic append
        ts3 = ts1.append(ts2, inplace=False)
        self.assertEquals(ts3.epoch, ts1.epoch)
        self.assertEquals(ts3.x0, ts1.x0)
        self.assertEquals(ts3.size, ts1.size+ts2.size)
        self.assertEquals(ts3.xspan, ts1.xspan+ts2.xspan)
        self.assertRaises(ValueError, ts3.append, ts1)
        nptest.assert_array_equal(ts3.value[:ts1.size], ts1.value)
        nptest.assert_array_equal(ts3.value[-ts2.size:], ts2.value)
        # test appending with one xindex deletes it in the output
        ts1.xindex
        ts4 = ts1.append(ts2, inplace=False)
        self.assertTrue(hasattr(ts4, '_xindex'))
        nptest.assert_array_equal(
            ts4.xindex.value,
            numpy.concatenate((ts1.xindex.value, ts2.xindex.value)))
        # test appending with both xindex appends as well
        ts1.xindex
        ts2.xindex
        ts5 = ts1.append(ts2, inplace=False)
        self.assertTrue(hasattr(ts5, '_xindex'))
        nptest.assert_array_equal(
            ts5.xindex.value,
            numpy.concatenate((ts1.xindex.value, ts2.xindex.value)))

    def test_prepend(self):
        """Test the `Series.prepend` method
        """
        ts1 = self.create()
        ts2 = self.create(x0=ts1.xspan[1]) * 2
        ts3 = ts2.prepend(ts1, inplace=False)
        self.assertEquals(ts3.x0, ts1.x0)
        self.assertEquals(ts3.size, ts1.size+ts2.size)
        self.assertEquals(ts3.xspan, ts1.xspan+ts2.xspan)
        self.assertRaises(ValueError, ts3.prepend, ts1)
        nptest.assert_array_equal(ts3.value[:ts1.size], ts1.value)
        nptest.assert_array_equal(ts3.value[-ts2.size:], ts2.value)

    def test_update(self):
        """Test the `Series.update` method
        """
        ts1 = self.create()
        ts2 = self.create(x0=ts1.xspan[1])[:ts1.size//2]
        ts3 = ts1.update(ts2, inplace=False)
        self.assertEquals(ts3.x0, ts1.x0 + abs(ts2.xspan)*ts1.x0.unit)
        self.assertEquals(ts3.size, ts1.size)
        self.assertRaises(ValueError, ts3.update, ts1)

    def test_pad(self):
        """Test the `Series.pad` method
        """
        ts1 = self.create()
        ts2 = ts1.pad(10)
        self.assertEquals(ts2.size, ts1.size + 20)
        nptest.assert_array_equal(
            ts2.value,
            numpy.concatenate((numpy.zeros(10), ts1.value, numpy.zeros(10))))
        self.assertEquals(ts2.x0, ts1.x0 - 10*ts1.x0.unit)
        # test pre-pad
        ts3 = ts1.pad((20, 10))
        self.assertEquals(ts3.size, ts1.size + 30)
        nptest.assert_array_equal(
            ts3.value,
            numpy.concatenate((numpy.zeros(20), ts1.value, numpy.zeros(10))))
        self.assertEquals(ts3.x0, ts1.x0 - 20*ts1.x0.unit)
        # test bogus input
        self.assertRaises(ValueError, ts1.pad, -1)

    def test_diff(self):
        """Test the `Series.diff` method

        This just ensures that the returned `Series` has the right length
        and the right x0
        """
        ts1 = self.create()
        diff = ts1.diff()
        self.assertIsInstance(diff, type(ts1))
        self.assertEqual(ts1.size - 1, diff.size)
        self.assertEqual(diff.x0, ts1.x0 + ts1.dx)
        self.assertEqual(diff.xspan[1], ts1.xspan[1])
        self.assertEqual(diff.channel, ts1.channel)
        # test n=3
        diff = ts1.diff(n=3)
        self.assertEqual(ts1.size - 3, diff.size)
        self.assertEqual(diff.x0, ts1.x0 + ts1.dx * 3)


if __name__ == '__main__':
    unittest.main()
