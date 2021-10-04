# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Unit tests for gwpy.types.series
"""

import warnings

import numpy

import pytest

from astropy import units

from ...segments import Segment
from ...testing import utils
from .. import (Series, Array2D, Index)
from .test_array import TestArray as _TestArray


class TestSeries(_TestArray):
    TEST_CLASS = Series

    def test_new(self):
        array = super().test_new()
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

    def test_xindex_dtype(self):
        x0 = numpy.longdouble(100)
        dx = numpy.float32(1e-4)
        series = self.create(x0=x0, dx=dx)
        assert series.xindex.dtype is x0.dtype

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
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='xindex was given',
                                    category=UserWarning)
            utils.assert_quantity_equal(array[a], self.TEST_CLASS(
                array.value[a], xindex=array.xindex[a],
                name=array.name, epoch=array.epoch, unit=array.unit),
            )

    def test_empty_slice(self, array):
        """Check that we can slice a `Series` into nothing

        This tests against a bug in sliceutils.py.
        """
        a2 = array[:0]
        assert a2.x0 == array.x0
        assert a2.dx == array.dx

        idx = numpy.array([False]*array.shape[0])  # False slice array
        a3 = array[idx]
        utils.assert_quantity_sub_equal(a2, a3)

        a4 = array[idx[:0]]  # empty slice array
        utils.assert_quantity_sub_equal(a2, a4)

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

    def test_crop_irregular(self):
        """The cropping on an irregularly spaced series.
        """
        x = numpy.linspace(0, 100, num=self.data.shape[0])

        # add shift to second half of times to create irregular space
        x[x > 50] += 1

        series = self.create(xindex=x)

        cropped = series.crop(end=75)
        utils.assert_quantity_equal(series[x < 75], cropped)

        cropped = series.crop(start=25)
        utils.assert_quantity_equal(series[x > 25], cropped)

        cropped = series.crop(start=25, end=75)
        utils.assert_quantity_equal(series[(x > 25) & (x < 75)], cropped)

    def test_is_compatible(self, array):
        """Test the `Series.is_compatible` method
        """
        other = self.create(name='TEST CASE 2')
        assert array.is_compatible(other)

    def test_is_compatible_error_dx(self, array):
        """Check that `Series.is_compatible` errors with mismatching ``dx``
        """
        other = self.create(dx=2)
        with pytest.raises(ValueError) as exc:
            array.is_compatible(other)
        assert "sample sizes do not match" in str(exc.value)

    def test_is_compatible_error_unit(self, array):
        """Check that `Series.is_compatible` errors with mismatching ``unit``
        """
        other = self.create(unit='m')
        with pytest.raises(ValueError) as exc:
            array.is_compatible(other)
        assert "units do not match" in str(exc.value)

    def test_is_compatible_xindex(self):
        """Check that irregular arrays are compatible if their xindexes match
        """
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        a = self.create(xindex=x)
        b = self.create(xindex=x)
        assert a.is_compatible(b)

    def test_is_compatible_error_xindex(self, array):
        """Check that `Series.is_compatible` errors with mismatching indexes
        """
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        other = self.create(xindex=x)
        with pytest.raises(ValueError) as exc:
            array.is_compatible(other)
        assert "indexes do not match" in str(exc.value)

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

    def test_pad_index(self):
        """Check that `Series.pad` correctly returns a padded index array
        """
        ts1 = self.create()
        ts1.xindex  # <- create the index for ts1
        ts2 = ts1.pad(10)
        utils.assert_array_equal(
            ts2.xindex,
            Index(
                numpy.linspace(*ts2.xspan, num=ts2.shape[0], endpoint=False),
                unit=ts1.xindex.unit,
            ),
        )

    def test_pad_asymmetric(self):
        ts1 = self.create()
        ts2 = ts1.pad((20, 10))
        assert ts2.shape[0] == ts1.shape[0] + 30
        utils.assert_array_equal(
            ts2.value,
            numpy.concatenate((numpy.zeros(20), ts1.value, numpy.zeros(10))))
        assert ts2.x0 == ts1.x0 - 20*ts1.x0.unit

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

    def test_shift(self):
        a = self.create(x0=0, dx=1, xunit='s')
        x0 = a.x0.copy()
        a.shift(5)
        assert a.x0 == x0 + 5 * x0.unit

        a.shift('1 hour')
        assert a.x0 == x0 + 3605 * x0.unit

        a.shift(-0.007)
        assert a.x0 == x0 + (3604.993) * x0.unit

        with pytest.raises(ValueError):
            a.shift('1 Hz')
