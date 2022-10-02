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

"""Unit tests for :mod:`gwpy.types.array2d`
"""

import pytest

import numpy

from astropy import units

from ...segments import Segment
from ...testing import utils
from .. import (Series, Array2D)
from .test_series import TestSeries as _TestSeries

SEED = 1


class TestArray2D(_TestSeries):
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

    def test_transpose(self, array):
        trans = array.T
        utils.assert_array_equal(trans.value, array.value.T)
        assert trans.unit is array.unit
        utils.assert_array_equal(trans.xindex, array.yindex)
        utils.assert_array_equal(trans.yindex, array.xindex)

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
        utils.assert_quantity_sub_equal(
            row,
            self.TEST_CLASS._rowclass(
                array.value[1:10:3, 0],
                x0=array.x0+array.dx, dx=array.dx*3,
                name=array.name, channel=array.channel, unit=array.unit,
            ),
            exclude=['epoch'],
        )

        # test dual slice returns type(self) with metadata
        subarray = array[1:5:2, 1:5:2]
        utils.assert_quantity_sub_equal(
            subarray,
            self.TEST_CLASS(
                array.value[1:5:2, 1:5:2],
                x0=array.x0+array.dx, dx=array.dx*2,
                y0=array.y0+array.dy, dy=array.dy*2,
                name=array.name, channel=array.channel, unit=array.unit,
            ),
            exclude=['epoch'],
        )

    def test_single_column_slice(self):
        """Check that we can slice an `Array2D` into a single column.

        But still represent the output as an `Array2D` with `Index` arrays.

        This tests regression of https://github.com/gwpy/gwpy/issues/1504.
        """
        # create an array with indices
        a = self.create()
        a.xindex
        a.yindex

        # select a slice of width 1 (as opposed to indexing a single column)
        b = a[0:1]

        # and check that the index arrays were correctly preserved
        assert isinstance(b, self.TEST_CLASS)
        for attr in ("x0", "dx", "xunit", "y0", "dy", "yunit"):
            assert getattr(a, attr) == getattr(b, attr)
        utils.assert_array_equal(b[0], a[0])
        utils.assert_array_equal(b.xindex, a.xindex[0:1])
        utils.assert_array_equal(b.yindex, a.yindex)

    def test_is_compatible_yindex(self):
        """Check that irregular arrays are compatible if their yindexes match
        """
        y = numpy.logspace(0, 2, num=self.data.shape[1])
        a = self.create(yindex=y)
        b = self.create(yindex=y)
        assert a.is_compatible(b)

    def test_is_compatible_error_yindex(self, array):
        """Check that `Array2D.is_compatible` errors with mismatching indexes
        """
        y = numpy.logspace(0, 2, num=self.data.shape[1])
        other = self.create(yindex=y)
        with pytest.raises(ValueError) as exc:
            array.is_compatible(other)
        assert "indexes do not match" in str(exc.value)

    def test_value_at(self, array):
        assert array.value_at(2, 3) == self.data[2][3] * array.unit
        assert array.value_at(5 * array.xunit, 2 * array.yunit) == (
            self.data[5][2] * array.unit)
        with pytest.raises(IndexError):
            array.value_at(1.6, 4.8)

    @pytest.mark.skip("not implemented for >1D arrays")
    def test_pad(self):
        return NotImplemented

    @pytest.mark.skip("not implemented for >1D arrays")
    def test_pad_index(self):
        return NotImplemented

    @pytest.mark.skip("not implemented for >1D arrays")
    def test_pad_asymmetric(self):
        return NotImplemented
