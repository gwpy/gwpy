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

"""Unit test for gwpy.types classes
"""

import abc
import os
import tempfile
import pickle

import pytest
from compat import unittest

from numpy import testing as nptest
import numpy

from astropy import (units, __version__ as astropy_version)
from astropy.time import Time

from gwpy.types import (Array, Series, Array2D)
from gwpy.detector import Channel
from gwpy.segments import Segment
from gwpy.time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

SEED = 1
GPS_EPOCH = 12345
TIME_EPOCH = Time(12345, format='gps', scale='utc')
CHANNEL_NAME = 'G1:TEST-CHANNEL'
CHANNEL = Channel(CHANNEL_NAME)


class CommonTests(object):
    __metaclass_ = abc.ABCMeta
    TEST_CLASS = Array
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_')
    if astropy_version >= '1.3':
        EMPTY_ARRAY_ERROR = None
    else:
        EMPTY_ARRAY_ERROR = IndexError

    @classmethod
    def setUpClass(cls, dtype=None):
        numpy.random.seed(SEED)
        cls.data = (numpy.random.random(100) * 1e5).astype(dtype=dtype)
        cls.datasq = cls.data ** 2

    @property
    def TEST_ARRAY(self):
        try:
            return self._TEST_ARRAY
        except AttributeError:
            self._TEST_ARRAY = self.create(name=CHANNEL_NAME, unit='meter',
                                           channel=CHANNEL_NAME,
                                           epoch=GPS_EPOCH)
            return self.TEST_ARRAY

    def create(self, *args, **kwargs):
        kwargs.setdefault('copy', False)
        return self.TEST_CLASS(self.data, *args, **kwargs)

    def assertQuantityEqual(self, q1, q2, *args):
        nptest.assert_array_equal(q1.value, q2.value)
        self.assertEqual(q1.unit, q2.unit)

    def assertArraysEqual(self, ts1, ts2, *args, **kwargs):
        exclude = kwargs.pop('exclude', [])
        if kwargs:
            raise TypeError("assertArraysEqual has no keyword argument %r"
                            % list(kwargs.keys())[0])
        self.assertQuantityEqual(ts1, ts2)
        if not args:
            args = self.TEST_CLASS._metadata_slots
        for attr in args:
            if attr in exclude:
                continue
            a = getattr(ts1, attr, None)
            b = getattr(ts2, attr, None)
            msg="%r attribute doesn't match: %r != %r" % (attr, a, b)
            if isinstance(a, numpy.ndarray) and isinstance(b, numpy.ndarray):
                nptest.assert_array_equal(a, b, err_msg=msg)
            else:
                self.assertEqual(a, b, msg=msg)

    # -- test basic construction ----------------

    def test_init(self):
        """Test Array creation
        """
        # test basic empty contructor
        self.assertRaises(TypeError, self.TEST_CLASS)
        if self.EMPTY_ARRAY_ERROR:
            self.assertRaises(self.EMPTY_ARRAY_ERROR, self.TEST_CLASS, [])
        # test with some data
        array = self.create()
        nptest.assert_array_equal(array.value, self.data)

    def test_unit(self):
        # test default unit is dimensionless
        array = self.create()
        self.assertEqual(array.unit, units.dimensionless_unscaled)
        # test deleter and recovery
        del array.unit
        del array.unit  # test twice to make sure Attribute Error isn't raised
        self.assertIsNone(array.unit)
        # test unit gets passed properly
        array = self.create(unit='m')
        self.assertIs(array.unit, units.m)
        # test unrecognised units
        with pytest.warns(units.UnitsWarning):
            array = self.create(unit='blah')
        self.assertIsInstance(array.unit, units.UnrecognizedUnit)
        self.assertEqual(str(array.unit), 'blah')
        # test setting unit doesn't work
        def _set_unit():
            array.unit = 'm'
        self.assertRaises(AttributeError, _set_unit)
        del array.unit
        array.unit = 'm'
        self.assertIs(array.unit, units.m)

    def test_name(self):
        # test default is no name
        array = self.create()
        self.assertIsNone(array.name)
        # test deleter and recovery
        del array.name
        del array.name
        self.assertIsNone(array.name)
        # test simple name
        array = self.create(name='TEST CASE')
        self.assertEquals(array.name, 'TEST CASE')

    def test_epoch(self):
        # test default is no epoch
        array = self.create()
        self.assertIsNone(array.epoch)
        # test deleter and recovery
        del array.epoch
        del array.epoch
        self.assertIsNone(array.epoch)
        # test epoch gets parsed properly
        array = self.create(epoch=GPS_EPOCH)
        self.assertIsInstance(array.epoch, Time)
        self.assertEquals(array.epoch, TIME_EPOCH)
        # test epoch in different formats
        array = self.create(epoch=LIGOTimeGPS(GPS_EPOCH))
        self.assertEquals(array.epoch, TIME_EPOCH)
        # test precision at high GPS times (to millisecond)
        gps = LIGOTimeGPS(1234567890, 123456000)
        array = self.create(epoch=gps)
        self.assertEqual(array.epoch.gps, float(gps))

    def test_channel(self):
        # test default channl is None
        array = self.create()
        self.assertIsNone(array.channel)
        # test deleter and recovery
        del array.channel
        del array.channel
        self.assertIsNone(array.channel)
        # test simple channel
        array = self.create(channel=CHANNEL_NAME)
        self.assertIsInstance(array.channel, Channel)
        self.assertEquals(array.channel, CHANNEL)
        # test existing channel doesn't get modified
        array = self.create(channel=CHANNEL)
        self.assertIs(array.channel, CHANNEL)
        # test preserves None
        array.channel = None
        self.assertIsNone(array.channel)

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

    def test_repr(self):
        # just test that it runs
        array = self.create()
        repr(array)

    def test_str(self):
        # just test that it runs
        array = self.create()
        str(array)

    def test_pickle(self):
        """Check pickle-unpickle yields unchanged data
        """
        ts = self.create()
        pkl = ts.dumps()
        ts2 = pickle.loads(pkl)
        self.assertArraysEqual(ts, ts2)

    # -- test I/O -------------------------------

    def _test_read_write(self, format, extension=None, auto=True, exclude=[],
                         readkwargs={}, writekwargs={}):
        if extension is None:
            extension = format
        extension = '.%s' % extension.lstrip('.')
        try:
            fp = tempfile.mktemp(suffix=extension)
            self.TEST_ARRAY.write(fp, format=format, **writekwargs)
            if auto:
                self.TEST_ARRAY.write(fp, **writekwargs)
            b = self.TEST_ARRAY.read(fp, self.TEST_ARRAY.name,
                                     format=format, **readkwargs)
            if auto:
                self.TEST_ARRAY.read(fp, self.TEST_ARRAY.name,
                                     **readkwargs)
            self.assertArraysEqual(self.TEST_ARRAY, b, exclude=exclude)
            return b
        finally:
            if os.path.exists(fp):
                os.remove(fp)


class ArrayTestCase(CommonTests, unittest.TestCase):
    pass


class SeriesTestCase(CommonTests, unittest.TestCase):
    TEST_CLASS = Series

    def test_init(self):
        super(SeriesTestCase, self).test_init()
        series = self.create(x0=0, dx=1)
        self.assertEqual(series.x0, units.Quantity(0, series._default_xunit))
        self.assertEqual(series.dx, units.Quantity(1, series._default_xunit))

    # -- test properties ------------------------

    def test_x0(self):
        # test simple
        series = self.create(x0=5)
        self.assertEqual(series.x0,
                         units.Quantity(5, self.TEST_CLASS._default_xunit))
        # test deleter
        del series.x0
        del series.x0
        self.assertEqual(series.x0,
                         units.Quantity(0, self.TEST_CLASS._default_xunit))
        # test quantity
        series.x0 = units.Quantity(5, 'm')
        self.assertEqual(series.x0, units.Quantity(5, 'm'))

    def test_dx(self):
        # test simple
        series = self.create(dx=5)
        self.assertEqual(series.dx,
                         units.Quantity(5, self.TEST_CLASS._default_xunit))
        # test deleter
        del series.dx
        del series.dx
        self.assertEqual(series.dx,
                         units.Quantity(1, self.TEST_CLASS._default_xunit))
        # test quantity
        series.dx = units.Quantity(5, 'm')
        self.assertEqual(series.dx, units.Quantity(5, 'm'))

    def test_xindex(self):
        x = numpy.linspace(0, 100, num=self.data.shape[0])
        # test simple
        series = self.create(xindex=x)
        self.assertQuantityEqual(
            series.xindex, units.Quantity(x, self.TEST_CLASS._default_xunit))
        # test deleter
        del series.xindex
        del series.xindex
        x1 = series.x0.value + series.shape[0] * series.dx.value
        x_default = numpy.linspace(series.x0.value, x1, num=series.shape[0],
                                   endpoint=False)
        self.assertQuantityEqual(
            series.xindex,
            units.Quantity(x_default, self.TEST_CLASS._default_xunit))
        # test setting of x0 and dx
        series = self.create(xindex=units.Quantity(x, 'Farad'))
        self.assertEqual(series.x0, units.Quantity(x[0], 'Farad'))
        self.assertEqual(series.dx, units.Quantity(x[1] - x[0], 'Farad'))
        self.assertEqual(series.xunit, units.Farad)
        self.assertEqual(series.xspan, (x[0], x[-1] + x[1] - x[0]))
        # test that setting xindex warns about ignoring dx or x0
        with pytest.warns(UserWarning):
            series = self.create(xindex=units.Quantity(x, 'Farad'), dx=1)
        with pytest.warns(UserWarning):
            series = self.create(xindex=units.Quantity(x, 'Farad'), x0=0)
        # test non-regular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=units.Quantity(x, 'Mpc'))
        def _get_dx():
            series.dx
        self.assertRaises(AttributeError, _get_dx)
        self.assertEqual(series.x0, units.Quantity(1, 'Mpc'))
        self.assertEqual(series.xspan, (x[0], x[-1] + x[-1] - x[-2]))

    def test_xunit(self, unit=None):
        if unit is None:
            unit = self.TEST_CLASS._default_xunit
        series = self.create(unit='Hz', dx=4*unit)
        self.assertEqual(series.xunit, unit)
        self.assertEqual(series.x0, 0*unit)
        self.assertEqual(series.dx, 4*unit)
        # for series only, test arbitrary xunit
        if self.TEST_CLASS is Series:
            series = self.create(unit='Hz', dx=4, xunit=units.m)
            self.assertEqual(series.x0, 0*units.m)
            self.assertEqual(series.dx, 4*units.m)

    def test_xspan(self):
        # test normal
        series = self.create(x0=1, dx=1)
        self.assertEqual(series.xspan, (1, 1 + 1 * series.shape[0]))
        self.assertIsInstance(series.xspan, Segment)
        # test from irregular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=x)
        self.assertEqual(series.xspan, (x[0], x[-1] + x[-1] - x[-2]))

    # -- test methods ---------------------------

    def test_getitem(self):
        a = self.create()
        self.assertEqual(a[0].value, a.value[0])
        self.assertIsInstance(a[0], units.Quantity)
        self.assertEqual(a[0].unit, a.unit)

    def test_zip(self):
        series = self.create()
        z = series.zip()
        nptest.assert_array_equal(
            z, numpy.column_stack((series.xindex.value, series.value)))

    def test_diff(self):
        # test simple
        series = self.create()
        d = series.diff()
        d2 = self.TEST_CLASS(numpy.diff(series.value),
                             x0=series.x0 + series.dx)
        self.assertArraysEqual(d, d2)
        d = series.diff(n=2)
        d2 = self.TEST_CLASS(numpy.diff(series.value, n=2),
                             x0=series.x0 + series.dx * 2)
        self.assertArraysEqual(d, d2)
        # test irregular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=x)
        d = series.diff(n=4)
        d2 = self.TEST_CLASS(numpy.diff(series.value, n=4), x0=x[4])
        self.assertArraysEqual(d, d2)

    def test_crop(self):
        """Test cropping `Series` by GPS times
        """
        ts = self.create()
        ts2 = ts.crop(10, 20)
        self.assertEqual(ts2.x0.value, 10)
        self.assertEqual(ts2.xspan[1], 20)
        nptest.assert_array_equal(ts2.value, ts.value[10:20])
        # check that warnings are printed for out-of-bounds
        with pytest.warns(UserWarning):
            ts.crop(ts.xspan[0]-1, ts.xspan[1])
            ts.crop(ts.xspan[0], ts.xspan[1]+1)

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
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        ts5 = self.create(xindex=x)
        self.assertRaises(ValueError, ts1.is_compatible, ts5)

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
        ts3 = ts1.append(ts2, inplace=False)
        self.assertFalse(hasattr(ts3, '_xindex'))
        # test appending with both xindex appends as well
        ts1.xindex
        ts2.xindex
        ts3 = ts1.append(ts2, inplace=False)
        self.assertTrue(hasattr(ts3, '_xindex'))
        nptest.assert_array_equal(
            ts3.xindex.value,
            numpy.concatenate((ts1.xindex.value, ts2.xindex.value)))
        # test appending with one only and not resize
        del ts2.xindex
        ts3 = ts1.append(ts2, inplace=False, resize=False)
        self.assertEqual(ts3.x0, ts1.x0 + ts1.dx * ts2.size)
        # test discontiguous appends - gap='raise'
        ts3 = self.create(x0=ts1.xspan[1] + 1)
        ts4 = ts1.copy()
        self.assertRaises(ValueError, ts1.append, ts3)
        # gap='ignore'
        ts4.append(ts3, gap='ignore')
        self.assertEqual(ts4.shape[0], ts1.shape[0] + ts3.shape[0])
        nptest.assert_array_equal(
            ts4.value, numpy.concatenate((ts1.value, ts3.value)))
        # gap='pad'
        ts4 = ts1.copy()
        ts4.append(ts3, gap='pad', pad=0)
        self.assertEqual(ts4.shape[0], ts1.shape[0] + 1 + ts3.shape[0])
        nptest.assert_array_equal(
            ts4.value, numpy.concatenate((ts1.value, [0], ts3.value)))

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

    def test_value_at(self):
        ts1 = self.TEST_CLASS([1, 2, 3, 4, 5, 4, 3, 2, 1], dx=.5, unit='m')
        self.assertEqual(ts1.value_at(1.5), 4 * ts1.unit)
        self.assertEqual(ts1.value_at(1.5 * ts1.xunit), 4 * units.m)
        self.assertRaises(IndexError, ts1.value_at, 1.6)
        # test TimeSeries unit conversion
        if ts1.xunit == units.s:
            self.assertEqual(ts1.value_at(1500 * units.millisecond),
                             4 * units.m)
        # test FrequencySeries unit conversion
        elif ts1.xunit == units.Hz:
            self.assertEqual(ts1.value_at(1500 * units.milliHertz),
                             4 * units.m)

    # -- test I/O -------------------------------

    def _test_read_write_ascii(self, format='txt'):
        extension = '.%s' % format.lstrip('.')
        try:
            with tempfile.NamedTemporaryFile(suffix=extension,
                                             delete=False) as f:
                self.TEST_ARRAY.write(f.name, format=format)
                self.TEST_ARRAY.write(f.name)
                b = self.TEST_ARRAY.read(f.name, format=format)
                self.TEST_ARRAY.read(f.name)
                nptest.assert_array_equal(self.TEST_ARRAY.value, b.value)
        finally:
            if os.path.exists(f.name):
               os.remove(f.name)


class Array2DTestCase(CommonTests, unittest.TestCase):
    TEST_CLASS = Array2D
    EMPTY_ARRAY_ERROR = ValueError

    @classmethod
    def setUpClass(cls, dtype=None):
        numpy.random.seed(SEED)
        cls.data = (numpy.random.random(100)
                     * 1e5).astype(dtype=dtype).reshape((10, 10))
        cls.datasq = cls.data ** 2

    def test_getitem(self):
        a = self.create()
        self.assertEqual(a[0, 0], a[0][0])
        nptest.assert_array_equal(a[0].value, a.value[0])
        self.assertIsInstance(a[0], self.TEST_CLASS._columnclass)
        self.assertIsInstance(a[0][0], units.Quantity)
        self.assertEqual(a[0].unit, a.unit)
        self.assertEqual(a[0][0].unit, a.unit)
        self.assertIsInstance(a[:,0], self.TEST_CLASS._rowclass)

    def test_value_at(self):
        ts1 = self.create(dx=.5, dy=.25, unit='m')
        self.assertEqual(ts1.value_at(1.5, .75), self.data[3][3] * ts1.unit)
        self.assertEqual(ts1.value_at(1.0 * ts1.xunit, .25 * ts1.yunit),
                         self.data[2][1] * units.m)
        self.assertRaises(IndexError, ts1.value_at, 1.6, 5.8)
        # test Spectrogram unit conversion
        if ts1.xunit == units.s and ts1.yunit == units.Hz:
            self.assertEqual(ts1.value_at(1500 * units.millisecond,
                                          750 * units.milliHertz),
                             self.data[3][3] * units.m)


if __name__ == '__main__':
    unittest.main()
