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

import os
import tempfile
import pickle
import warnings

import pytest

import numpy

from astropy import units
from astropy.time import Time

from gwpy.types import (Array, Series, Array2D, Index)
from gwpy.detector import Channel
from gwpy.segments import Segment
from gwpy.time import LIGOTimeGPS

from . import utils

warnings.filterwarnings('always', category=units.UnitsWarning)
warnings.filterwarnings('always', category=UserWarning)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

SEED = 1
GPS_EPOCH = 12345
TIME_EPOCH = Time(12345, format='gps', scale='utc')
CHANNEL_NAME = 'G1:TEST-CHANNEL'
CHANNEL = Channel(CHANNEL_NAME)


class TestArray(object):
    """Test `gwpy.types.Array`
    """
    TEST_CLASS = Array
    DTYPE = None

    # -- setup ----------------------------------

    @classmethod
    def setup_class(cls):
        numpy.random.seed(SEED)
        cls.data = (numpy.random.random(100) * 1e5).astype(dtype=cls.DTYPE)

    @classmethod
    def create(cls, *args, **kwargs):
        kwargs.setdefault('copy', False)
        return cls.TEST_CLASS(cls.data, *args, **kwargs)

    @classmethod
    @pytest.fixture()
    def array(cls):
        return cls.create()

    @property
    def TEST_ARRAY(self):
        try:
            return self._TEST_ARRAY
        except AttributeError:
            # create array
            self._TEST_ARRAY = self.create(name=CHANNEL_NAME, unit='meter',
                                           channel=CHANNEL_NAME,
                                           epoch=GPS_EPOCH)
            # customise channel a wee bit
            #    used to test pickle/unpickle when storing channel as
            #    dataset attr in HDF5
            self._TEST_ARRAY.channel.sample_rate = 128
            self._TEST_ARRAY.channel.unit = 'm'
            return self.TEST_ARRAY

    # -- test basic construction ----------------

    def test_new(self):
        """Test Array creation
        """
        # test basic empty contructor
        with pytest.raises(TypeError):
            self.TEST_CLASS()

        # test with some data
        array = self.create()
        utils.assert_array_equal(array.value, self.data)

        # test that copy=True ensures owndata
        a = self.create(copy=False)
        assert self.create(copy=False).flags.owndata is False
        assert self.create(copy=True).flags.owndata is True

        # return array for subclasses to use
        return array

    def test_unit(self, array):
        # test default unit is dimensionless
        assert array.unit is units.dimensionless_unscaled

        # test deleter and recovery
        del array.unit
        del array.unit  # test twice to make sure AttributeError isn't raised
        assert array.unit is None

        # test unit gets passed properly
        array = self.create(unit='m')
        assert array.unit is units.m

        # test unrecognised units
        with pytest.warns(units.UnitsWarning):
            array = self.create(unit='blah')
        assert isinstance(array.unit, units.IrreducibleUnit)
        assert str(array.unit) == 'blah'

        # test setting unit doesn't work
        with pytest.raises(AttributeError):
            array.unit = 'm'
        del array.unit
        array.unit = 'm'
        assert array.unit is units.m

    def test_name(self, array):
        # test default is no name
        assert array.name is None

        # test deleter and recovery
        del array.name
        del array.name
        assert array.name is None

        # test simple name
        array = self.create(name='TEST CASE')
        assert array.name == 'TEST CASE'

        # test None gets preserved
        array.name = None
        assert array.name is None

        # but everything else gets str()
        array.name = 4
        assert array.name == '4'

    def test_epoch(self, array):
        # test default is no epoch
        assert array.epoch is None

        # test deleter and recovery
        del array.epoch
        del array.epoch
        assert array.epoch is None

        # test epoch gets parsed properly
        array = self.create(epoch=GPS_EPOCH)
        assert isinstance(array.epoch, Time)
        assert array.epoch == TIME_EPOCH

        # test epoch in different formats
        array = self.create(epoch=LIGOTimeGPS(GPS_EPOCH))
        assert array.epoch == TIME_EPOCH

        # test precision at high GPS times (to millisecond)
        gps = LIGOTimeGPS(1234567890, 123456000)
        array = self.create(epoch=gps)
        assert array.epoch.gps == float(gps)

        # test None gets preserved
        array.epoch = None
        assert array.epoch is None

    def test_channel(self, array):
        # test default channl is None
        assert array.channel is None

        # test deleter and recovery
        del array.channel
        del array.channel
        assert array.channel is None

        # test simple channel
        array = self.create(channel=CHANNEL_NAME)
        assert isinstance(array.channel, Channel)
        assert array.channel == CHANNEL

        # test existing channel doesn't get modified
        array = self.create(channel=CHANNEL)
        assert array.channel is CHANNEL

        # test preserves None
        array.channel = None
        assert array.channel is None

    def test_math(self, array):
        array.override_unit('Hz')
        # test basic operations
        arraysq = array ** 2
        utils.assert_array_equal(arraysq.value, self.data ** 2)
        assert arraysq.unit == units.Hz ** 2
        assert arraysq.name == array.name
        assert arraysq.epoch == array.epoch
        assert arraysq.channel == array.channel

    def test_copy(self, array):
        utils.assert_quantity_sub_equal(array, array.copy())

    def test_repr(self, array):
        # just test that it runs
        repr(array)

    def test_str(self, array):
        # just test that it runs
        str(array)

    def test_pickle(self, array):
        # check pickle-unpickle yields unchanged data
        pkl = array.dumps()
        a2 = pickle.loads(pkl)
        utils.assert_quantity_sub_equal(array, a2)

    # -- test methods ---------------------------

    def test_tostring(self, array):
        assert array.tostring() == array.value.tostring()

    def test_abs(self, array):
        utils.assert_quantity_equal(array.abs(), numpy.abs(array))

    def test_median(self, array):
        utils.assert_quantity_equal(
            array.median(), numpy.median(array.value) * array.unit)

    def test_override_unit(self, array):
        assert array.unit is units.dimensionless_unscaled

        # check basic override works
        array.override_unit('m')
        assert array.unit is units.meter

        # check parse_strict works for each of 'raise' (default), 'warn',
        # and 'silent'
        with pytest.raises(ValueError):
            array.override_unit('blah', parse_strict='raise')

        with pytest.warns(units.UnitsWarning):
            array.override_unit('blah', parse_strict='warn')

        array.override_unit('blah', parse_strict='silent')
        assert isinstance(array.unit, units.IrreducibleUnit)
        assert str(array.unit) == 'blah'

    # -- test I/O -------------------------------

    def _test_read_write(self, format, extension=None, auto=True, exclude=[],
                         readkwargs={}, writekwargs={}):
        """Helper method for testing unified I/O for `Array` instances
        """
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
            utils.assert_array_equal(self.TEST_ARRAY, b, exclude=exclude)
            return b
        finally:
            if os.path.exists(fp):
                os.remove(fp)


class TestSeries(TestArray):
    TEST_CLASS = Series

    def test_new(self):
        array = super(TestSeries, self).test_new()
        assert array.x0 == units.Quantity(0, self.TEST_CLASS._default_xunit)
        assert array.dx == units.Quantity(1, self.TEST_CLASS._default_xunit)
        return array

    # -- test properties ------------------------

    def test_x0(self, array):
        array.x0 = 5
        # test simple
        assert array.x0 == 5 * self.TEST_CLASS._default_xunit

        # test deleter
        del array.x0
        del array.x0
        assert array.x0 == 0 * self.TEST_CLASS._default_xunit

        # test quantity
        array.x0 = 5 * units.m
        assert array.x0 == units.Quantity(5, 'm')

    def test_dx(self, array):
        array.dx = 5 * self.TEST_CLASS._default_xunit
        # test simple
        assert array.dx == units.Quantity(5, self.TEST_CLASS._default_xunit)

        # test deleter
        del array.dx
        del array.dx
        assert array.dx == units.Quantity(1, self.TEST_CLASS._default_xunit)

        # test quantity
        array.dx = 5 * units.m
        assert array.dx == units.Quantity(5, 'm')

    def test_xindex(self):
        x = numpy.linspace(0, 100, num=self.data.shape[0])

        # test simple
        series = self.create(xindex=x)
        utils.assert_quantity_equal(
            series.xindex, units.Quantity(x, self.TEST_CLASS._default_xunit))

        # test deleter
        del series.xindex
        del series.xindex
        x1 = series.x0.value + series.shape[0] * series.dx.value
        x_default = numpy.linspace(series.x0.value, x1, num=series.shape[0],
                                   endpoint=False)
        utils.assert_quantity_equal(
            series.xindex,
            units.Quantity(x_default, self.TEST_CLASS._default_xunit))

        # test setting of x0 and dx
        series = self.create(xindex=units.Quantity(x, 'Farad'))
        assert series.x0 == units.Quantity(x[0], 'Farad')
        assert series.dx == units.Quantity(x[1] - x[0], 'Farad')
        assert series.xunit == units.Farad
        assert series.xspan == (x[0], x[-1] + x[1] - x[0])

        # test that setting xindex warns about ignoring dx or x0
        with pytest.warns(UserWarning):
            series = self.create(xindex=units.Quantity(x, 'Farad'), dx=1)
        with pytest.warns(UserWarning):
            series = self.create(xindex=units.Quantity(x, 'Farad'), x0=0)

        # test non-regular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=units.Quantity(x, 'Mpc'))
        with pytest.raises(AttributeError):
            series.dx
        assert series.x0 == units.Quantity(1, 'Mpc')
        assert series.xspan == (x[0], x[-1] + x[-1] - x[-2])

    def test_xunit(self, unit=None):
        if unit is None:
            unit = self.TEST_CLASS._default_xunit
        series = self.create(dx=4*unit)
        assert series.xunit == unit
        assert series.x0 == 0*unit
        assert series.dx == 4*unit
        # for series only, test arbitrary xunit
        if self.TEST_CLASS is (Series, Array2D):
            series = self.create(dx=4, xunit=units.m)
            assert series.x0 == 0*units.m
            assert series.dx == 4*units.m

    def test_xspan(self):
        # test normal
        series = self.create(x0=1, dx=1)
        assert series.xspan == (1, 1 + 1 * series.shape[0])
        assert isinstance(series.xspan, Segment)
        # test from irregular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=x)
        assert series.xspan == (x[0], x[-1] + x[-1] - x[-2])

    # -- test methods ---------------------------

    def test_getitem(self, array):
        # item access
        utils.assert_quantity_equal(
            array[0], units.Quantity(array.value[0], array.unit))

        # slice
        utils.assert_quantity_equal(array[1::2], self.TEST_CLASS(
            array.value[1::2], x0=array.x0+array.dx, dx=array.dx*2,
            name=array.name, epoch=array.epoch, unit=array.unit),
        )

        # index array
        a = numpy.array([3, 4, 1, 2])
        utils.assert_quantity_equal(array[a], self.TEST_CLASS(
            array.value[a], xindex=array.xindex[a],
            name=array.name, epoch=array.epoch, unit=array.unit),
        )

    def test_zip(self, array):
        z = array.zip()
        utils.assert_array_equal(
            z, numpy.column_stack((array.xindex.value, array.value)))

    def test_crop(self, array):
        a2 = array.crop(10, 20)
        utils.assert_quantity_equal(array[10:20], a2)
        # check that warnings are printed for out-of-bounds
        with pytest.warns(UserWarning):
            array.crop(array.xspan[0]-1, array.xspan[1])
        with pytest.warns(UserWarning):
            array.crop(array.xspan[0], array.xspan[1]+1)

    def test_is_compatible(self, array):
        """Test the `Series.is_compatible` method
        """
        a2 = self.create(name='TEST CASE 2')
        assert array.is_compatible(a2)

        a3 = self.create(dx=2)
        with pytest.raises(ValueError):
            array.is_compatible(a3)

        a4 = self.create(unit='m')
        with pytest.raises(ValueError):
            array.is_compatible(a4)

        x = numpy.logspace(0, 2, num=self.data.shape[0])
        a5 = self.create(xindex=x)
        with pytest.raises(ValueError):
            array.is_compatible(a5)

    def test_is_contiguous(self, array):
        a2 = self.create(x0=array.xspan[1])
        assert array.is_contiguous(a2) == 1
        assert array.is_contiguous(a2.value) == 1

        ts3 = self.create(x0=array.xspan[1]+1)
        assert array.is_contiguous(ts3) == 0

        ts4 = self.create(x0=-array.xspan[1])
        assert array.is_contiguous(ts4) == -1

    def test_append(self, array):
        a2 = self.create(x0=array.xspan[1])

        # test basic append
        a3 = array.append(a2, inplace=False)
        assert a3.epoch == array.epoch
        assert a3.x0 == array.x0
        assert a3.size == array.size+a2.size
        assert a3.xspan == array.xspan+a2.xspan
        utils.assert_array_equal(a3.value[:array.shape[0]], array.value)
        utils.assert_array_equal(a3.value[-a2.shape[0]:], a2.value)

        # check that appending again causes a problem
        with pytest.raises(ValueError):
            a3.append(array)

        # test appending with one xindex deletes it in the output
        array.xindex
        a3 = array.append(a2, inplace=False)
        assert hasattr(a3, '_xindex') is False

        # test appending with both xindex appends as well
        array.xindex
        a2.xindex
        a3 = array.append(a2, inplace=False)
        assert hasattr(a3, '_xindex')
        utils.assert_array_equal(
            a3.xindex.value,
            numpy.concatenate((array.xindex.value, a2.xindex.value)))

        # test appending with one only and not resize
        del a2.xindex
        a3 = array.append(a2, inplace=False, resize=False)
        assert a3.x0 == array.x0 + array.dx * a2.shape[0]

        # test discontiguous appends - gap='raise'
        a3 = self.create(x0=array.xspan[1] + 1)
        ts4 = array.copy()
        with pytest.raises(ValueError):
            array.append(a3)

        # gap='ignore'
        ts4.append(a3, gap='ignore')
        assert ts4.shape[0] == array.shape[0] + a3.shape[0]
        utils.assert_array_equal(
            ts4.value, numpy.concatenate((array.value, a3.value)))

        # gap='pad'
        ts4 = array.copy()
        ts4.append(a3, gap='pad', pad=0)
        assert ts4.shape[0] == array.shape[0] + 1 + a3.shape[0]
        z = numpy.zeros((1,) + array.shape[1:])
        utils.assert_array_equal(
            ts4.value, numpy.concatenate((array.value, z, a3.value)))

    def test_prepend(self, array):
        """Test the `Series.prepend` method
        """
        a2 = self.create(x0=array.xspan[1]) * 2
        a3 = a2.prepend(array, inplace=False)
        assert a3.x0 == array.x0
        assert a3.size == array.size + a2.size
        assert a3.xspan == array.xspan + a2.xspan
        with pytest.raises(ValueError):
            a3.prepend(array)
        utils.assert_array_equal(a3.value[:array.shape[0]], array.value)
        utils.assert_array_equal(a3.value[-a2.shape[0]:], a2.value)

    def test_update(self):
        """Test the `Series.update` method
        """
        ts1 = self.create()
        ts2 = self.create(x0=ts1.xspan[1])[:ts1.size//2]
        ts3 = ts1.update(ts2, inplace=False)
        assert ts3.x0 == ts1.x0 + abs(ts2.xspan)*ts1.x0.unit
        assert ts3.size == ts1.size
        with pytest.raises(ValueError):
            ts3.update(ts1)

    def test_pad(self):
        """Test the `Series.pad` method
        """
        ts1 = self.create()
        ts2 = ts1.pad(10)
        assert ts2.shape[0] == ts1.shape[0] + 20
        utils.assert_array_equal(
            ts2.value,
            numpy.concatenate((numpy.zeros(10), ts1.value, numpy.zeros(10))))
        assert ts2.x0 == ts1.x0 - 10*ts1.x0.unit
        # test pre-pad
        ts3 = ts1.pad((20, 10))
        assert ts3.size == ts1.size + 30
        utils.assert_array_equal(
            ts3.value,
            numpy.concatenate((numpy.zeros(20), ts1.value, numpy.zeros(10))))
        assert ts3.x0 == ts1.x0 - 20*ts1.x0.unit
        # test bogus input
        with pytest.raises(ValueError):
            ts1.pad(-1)

    def test_diff(self, array):
        """Test the `Series.diff` method

        This just ensures that the returned `Series` has the right length
        and the right x0
        """
        diff = array.diff(axis=0)
        assert isinstance(diff, type(array))
        assert array.shape[0] - 1 == diff.shape[0]
        assert diff.x0 == array.x0 + array.dx
        assert diff.xspan[1] == array.xspan[1]
        assert diff.channel == array.channel
        # test n=3
        diff = array.diff(n=3)
        assert array.shape[-1] - 3 == diff.shape[-1]
        assert diff.x0 == array.x0 + array.dx * 3

    def test_value_at(self):
        ts1 = self.create(dx=.5)
        assert ts1.value_at(1.5) == self.data[3] * ts1.unit
        assert ts1.value_at(1.5 * ts1.xunit) == self.data[3] * ts1.unit
        with pytest.raises(IndexError):
            ts1.value_at(1.6)
        # test TimeSeries unit conversion
        if ts1.xunit == units.s:
            assert ts1.value_at(1500 * units.millisecond) == (
                self.data[3] * ts1.unit)
        # test FrequencySeries unit conversion
        elif ts1.xunit == units.Hz:
            assert ts1.value_at(1500 * units.milliHertz) == (
                self.data[3] * ts1.unit)

    # -- test I/O -------------------------------

    def _test_read_write_ascii(self, format='txt'):
        extension = '.%s' % format.lstrip('.')
        try:
            with tempfile.NamedTemporaryFile(suffix=extension, mode='w',
                                             delete=False) as f:
                self.TEST_ARRAY.write(f.name, format=format)
                self.TEST_ARRAY.write(f.name)
                b = self.TEST_ARRAY.read(f.name, format=format)
                self.TEST_ARRAY.read(f.name)
                utils.assert_array_equal(self.TEST_ARRAY.value, b.value)
        finally:
            if os.path.exists(f.name):
                os.remove(f.name)


class TestArray2D(TestSeries):
    TEST_CLASS = Array2D

    @classmethod
    def setup_class(cls, dtype=None):
        numpy.random.seed(SEED)
        cls.data = (numpy.random.random(100) * 1e5).astype(
            dtype=dtype).reshape(
            (20, 5))
        cls.datasq = cls.data ** 2

    # -- test properties ------------------------

    def test_y0(self, array):
        array.y0 = 5
        # test simple
        assert array.y0 == 5 * self.TEST_CLASS._default_yunit

        # test deleter
        del array.y0
        del array.y0
        assert array.y0 == 0 * self.TEST_CLASS._default_yunit

        # test quantity
        array.y0 = 5 * units.m
        assert array.y0 == units.Quantity(5, 'm')

    def test_dy(self, array):
        array.dy = 5 * self.TEST_CLASS._default_yunit
        # test simple
        assert array.dy == units.Quantity(5, self.TEST_CLASS._default_yunit)

        # test deleter
        del array.dy
        del array.dy
        assert array.dy == units.Quantity(1, self.TEST_CLASS._default_yunit)

        # test quantity
        array.dy = 5 * units.m
        assert array.dy == units.Quantity(5, 'm')

    def test_yindex(self):
        y = numpy.linspace(0, 100, num=self.data.shape[1])

        # test simple
        series = self.create(yindex=y)
        utils.assert_quantity_equal(
            series.yindex, units.Quantity(y, self.TEST_CLASS._default_yunit))

        # test deleter
        del series.yindex
        del series.yindex
        y1 = series.y0.value + series.shape[1] * series.dy.value
        y_default = numpy.linspace(series.y0.value, y1, num=series.shape[1],
                                   endpoint=False)
        utils.assert_quantity_equal(
            series.yindex,
            units.Quantity(y_default, self.TEST_CLASS._default_yunit))

        # test setting of y0 and dy
        series = self.create(yindex=units.Quantity(y, 'Farad'))
        assert series.y0 == units.Quantity(y[0], 'Farad')
        assert series.dy == units.Quantity(y[1] - y[0], 'Farad')
        assert series.yunit == units.Farad
        assert series.yspan == (y[0], y[-1] + y[1] - y[0])

        # test that setting yindex warns about ignoring dy or y0
        with pytest.warns(UserWarning):
            series = self.create(yindex=units.Quantity(y, 'Farad'), dy=1)
        with pytest.warns(UserWarning):
            series = self.create(yindex=units.Quantity(y, 'Farad'), y0=0)

        # test non-regular yindex
        y = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(yindex=units.Quantity(y, 'Mpc'))
        with pytest.raises(AttributeError):
            series.dy
        assert series.y0 == units.Quantity(1, 'Mpc')
        assert series.yspan == (y[0], y[-1] + y[-1] - y[-2])

    def test_yunit(self, unit=None):
        if unit is None:
            unit = self.TEST_CLASS._default_yunit
        series = self.create(dy=4*unit)
        assert series.yunit == unit
        assert series.y0 == 0*unit
        assert series.dy == 4*unit
        # for series only, test arbitrary yunit
        if self.TEST_CLASS in (Series, Array2D):
            series = self.create(dy=4, yunit=units.m)
            assert series.y0 == 0*units.m
            assert series.dy == 4*units.m

    def test_yspan(self):
        # test normal
        series = self.create(y0=1, dy=1)
        assert series.yspan == (1, 1 + 1 * series.shape[1])
        assert isinstance(series.yspan, Segment)
        # test from irregular yindex
        y = numpy.logspace(0, 2, num=self.data.shape[1])
        series = self.create(yindex=y)
        assert series.yspan == (y[0], y[-1] + y[-1] - y[-2])

    # -- test methods ---------------------------

    @pytest.mark.parametrize('create_kwargs', [
        {'x0': 0, 'dx': 1, 'y0': 100, 'dy': 2},
        {'xindex': numpy.arange(20), 'yindex': numpy.linspace(0, 100, 5)},
        {'x0': 0, 'dx': 1, 'yindex': numpy.linspace(0, 100, 5)},
        {'xindex': numpy.arange(20), 'y0': 100, 'dy': 2},
    ])
    def test_getitem(self, array, create_kwargs):
        array = self.create(name='test_getitem', **create_kwargs)

        # test element returns as quantity
        element = array[0, 0]
        assert element == array[0][0]
        assert isinstance(element, units.Quantity)
        utils.assert_quantity_equal(element, array.value[0, 0] * array.unit)

        # test column slice returns as _columnclass
        utils.assert_quantity_sub_equal(array[2], array[2, :])
        column = array[0, 0::2]
        utils.assert_quantity_sub_equal(column, self.TEST_CLASS._columnclass(
            array.value[0, 0::2], x0=array.y0, dx=array.dy*2, name=array.name,
            channel=array.channel, unit=array.unit, epoch=array.epoch))

        # test row slice returns as _rowclass
        row = array[1:10:3, 0]
        utils.assert_array_equal(row.value, array.value[1:10:3, 0])
        utils.assert_quantity_sub_equal(row, self.TEST_CLASS._rowclass(
                array.value[1:10:3, 0],
                x0=array.x0+array.dx, dx=array.dx*3,
                name=array.name, channel=array.channel, unit=array.unit),
            exclude=['epoch'])

        # test dual slice returns type(self) with metadata
        subarray = array[1:5:2, 1:5:2]
        utils.assert_quantity_sub_equal(subarray, self.TEST_CLASS(
                array.value[1:5:2, 1:5:2],
                x0=array.x0+array.dx, dx=array.dx*2,
                y0=array.y0+array.dy, dy=array.dy*2,
                name=array.name, channel=array.channel, unit=array.unit),
            exclude=['epoch'])

    def test_is_compatible(self, array):
        super(TestArray2D, self).test_is_compatible(array)

        a2 = self.create(dy=2)
        with pytest.raises(ValueError):
            array.is_compatible(a2)

        y = numpy.logspace(0, 2, num=self.data.shape[0])
        a2 = self.create(yindex=y)
        with pytest.raises(ValueError):
            array.is_compatible(a2)

    def test_value_at(self, array):
        assert array.value_at(2, 3) == self.data[2][3] * array.unit
        assert array.value_at(5 * array.xunit, 2 * array.yunit) == (
            self.data[5][2] * array.unit)
        with pytest.raises(IndexError):
            array.value_at(1.6, 4.8)

    def test_pad(self):
        return NotImplemented


class TestIndex(object):
    TEST_CLASS = Index

    def test_is_regular(self):
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], 's')
        assert a.is_regular()
        assert a[::-1].is_regular()

        b = self.TEST_CLASS([1, 2, 4, 5, 7, 8, 9])
        assert not b.is_regular()

    def test_regular(self):
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], 's')
        assert a.regular
        assert a.regular is a.info.meta['regular']

    def test_getitem(self):
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], 'Hz')
        assert type(a[0]) is units.Quantity
        assert a[0] == 1 * units.Hz
        assert isinstance(a[:2], type(a))
