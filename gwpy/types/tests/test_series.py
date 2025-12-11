# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Unit tests for gwpy.types.series."""

from __future__ import annotations

import warnings
from typing import (
    Generic,
    TypeVar,
    cast,
)

import numpy
import pytest
from astropy import units
from numpy.testing import assert_array_equal

from ...segments import Segment
from ...testing import utils
from .. import Array2D, Index, Series
from .test_array import TestArray as _TestArray

SeriesType = TypeVar("SeriesType", bound=Series)


class TestSeries(_TestArray[SeriesType], Generic[SeriesType]):
    """Test the `Series` class."""

    TEST_CLASS: type[SeriesType] = Series

    def assert_new(self, array: SeriesType):
        """Assert properties of a new `Series`."""
        super().assert_new(array)
        assert array.x0 == units.Quantity(0, self.TEST_CLASS._default_xunit)
        assert array.dx == units.Quantity(1, self.TEST_CLASS._default_xunit)

    # -- test properties -------------

    def test_x0(self, array: SeriesType):
        """Test `Series.x0`."""
        array.x0 = 5
        # test simple
        assert array.x0 == 5 * self.TEST_CLASS._default_xunit

        # test deleter
        del array.x0
        del array.x0
        assert array.x0 == 0 * self.TEST_CLASS._default_xunit

        # test quantity
        array.x0 = 5 * units.m
        assert array.x0 == units.Quantity(5, "m")

    def test_dx(self, array: SeriesType):
        """Test `Series.dx`."""
        array.dx = 5 * self.TEST_CLASS._default_xunit
        # test simple
        assert array.dx == units.Quantity(5, self.TEST_CLASS._default_xunit)

        # test deleter
        del array.dx
        del array.dx
        assert array.dx == units.Quantity(1, self.TEST_CLASS._default_xunit)

        # test quantity
        array.dx = 5 * units.m
        assert array.dx == units.Quantity(5, "m")

    def test_xindex(self):
        """Test `Series.xindex`."""
        x = numpy.linspace(0, 100, num=self.data.shape[0])

        # test simple
        series = self.create(xindex=x)
        utils.assert_quantity_equal(
            series.xindex,
            units.Quantity(x, self.TEST_CLASS._default_xunit),
        )

        # test deleter
        del series.xindex
        del series.xindex
        x1 = series.x0.value + series.shape[0] * series.dx.value
        x_default = numpy.linspace(
            series.x0.value,
            x1,
            num=series.shape[0],
            endpoint=False,
        )
        utils.assert_quantity_equal(
            series.xindex,
            units.Quantity(x_default, self.TEST_CLASS._default_xunit),
        )

        # test setting of x0 and dx
        series = self.create(xindex=units.Quantity(x, "Farad"))
        assert series.x0 == units.Quantity(x[0], "Farad")
        assert series.dx == units.Quantity(x[1] - x[0], "Farad")
        assert series.xunit == units.Farad
        assert series.xspan == (x[0], x[-1] + x[1] - x[0])

        # test that setting xindex warns about ignoring dx or x0
        with pytest.warns(
            UserWarning,
            match="xindex was given",
        ):
            series = self.create(xindex=units.Quantity(x, "Farad"), dx=1)
        with pytest.warns(
            UserWarning,
            match="xindex was given",
        ):
            series = self.create(xindex=units.Quantity(x, "Farad"), x0=0)

        # test non-regular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=units.Quantity(x, "Mpc"))
        with pytest.raises(AttributeError):
            series.dx  # noqa: B018
        assert series.x0 == units.Quantity(1, "Mpc")
        assert series.xspan == (x[0], x[-1] + x[-1] - x[-2])

    def test_xindex_length_exception(self):
        """Test that xindex must match data length."""
        with pytest.raises(
            ValueError,
            match="xindex must have the same length as data",
        ):
            Series([1, 2, 3], xindex=[0])

    def test_xindex_dtype(self):
        """Test that xindex dtype matches x0 and dx."""
        x0 = numpy.longdouble(100)
        dx = numpy.float32(1e-4)
        series = self.create(x0=x0, dx=dx)
        assert series.xindex.dtype is x0.dtype

    def test_xunit(self):
        """Test `Series.xunit`."""
        unit = self.TEST_CLASS._default_xunit
        series = self.create(dx=4 * unit)
        assert series.xunit == unit
        assert series.x0 == 0 * unit
        assert series.dx == 4 * unit
        # for series only, test arbitrary xunit
        if self.TEST_CLASS is (Series, Array2D):
            series = self.create(dx=4, xunit=units.m)
            assert series.x0 == 0 * units.m
            assert series.dx == 4 * units.m

    def test_xspan(self):
        """Test `Series.xspan`."""
        # test normal
        series = self.create(x0=1, dx=1)
        assert series.xspan == (1, 1 + 1 * series.shape[0])
        assert isinstance(series.xspan, Segment)
        # test from irregular xindex
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        series = self.create(xindex=x)
        assert series.xspan == (x[0], x[-1] + x[-1] - x[-2])

    # -- test i/o --------------------

    def test_read_write_csv(self, array: SeriesType):
        """Test reading and writing a `Series` to CSV."""
        utils.test_read_write(
            array,
            "csv",
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={
                "exclude": [
                    "name",
                    "channel",
                    "unit",
                ],
            },
        )

    # -- test methods ----------------

    def test_getitem(self, array: SeriesType):
        """Test `Series[...]` item access."""
        # item access
        utils.assert_quantity_equal(
            array[0],
            units.Quantity(array.value[0], array.unit),
        )

        # slice
        utils.assert_quantity_equal(
            array[1::2],
            self.TEST_CLASS(
                array.value[1::2],
                x0=array.x0 + array.dx,
                dx=array.dx * 2,
                name=array.name,
                epoch=array.epoch,
                unit=array.unit,
            ),
        )

        # index array
        a = numpy.array([3, 4, 1, 2])
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="xindex was given",
                category=UserWarning,
            )
            utils.assert_quantity_equal(
                array[a],
                self.TEST_CLASS(
                    array.value[a],
                    xindex=array.xindex[a],
                    name=array.name,
                    epoch=array.epoch,
                    unit=array.unit,
                ),
            )

    def test_getitem_index(self, array: SeriesType):
        """Test that __getitem__ also applies to an xindex.

        When subsetting a Series with an iterable of integer indices,
        make sure that the xindex, if it exists, is also subsetted. Tests
        regression against https://gitlab.com/gwpy/gwpy/-/issues/1680.
        """
        array.xindex  # create xindex  # noqa: B018
        indices = numpy.array([0, 1, len(array) - 1])
        newarray = array[indices]

        assert len(newarray) == 3
        assert len(newarray) == len(newarray.value)
        assert len(newarray.value) == len(newarray.xindex)

    def test_getitem_list_index(self, array: SeriesType):
        """Test that __getitem__ works with list and numpy.array."""
        indices = numpy.array([0, 1, len(array) - 1])
        lindices = [0, 1, len(array) - 1]
        utils.assert_quantity_sub_equal(array[indices], array[lindices])

    def test_single_getitem_not_created(self, array: SeriesType):
        """Test that array[i] does not return an object with a new _xindex."""
        # check that there is no xindex when a single value is accessed
        with pytest.raises(
            AttributeError,
            match=r"'.*' object has no",
        ):
            array[0].xindex  # noqa: B018

        # we don't need this, we don't want it accidentally injected
        with pytest.raises(
            AttributeError,
            match=r"'.*' object has no",
        ):
            array[0]._xindex  # noqa: B018

    def test_empty_slice(self, array: SeriesType):
        """Check that we can slice a `Series` into nothing.

        This tests against a bug in sliceutils.py.
        """
        a2 = array[:0]
        assert a2.x0 == array.x0
        assert a2.dx == array.dx

        idx = numpy.array([False] * array.shape[0])  # False slice array
        a3 = array[idx]
        utils.assert_quantity_sub_equal(a2, a3)

        a4 = array[idx[:0]]  # empty slice array
        utils.assert_quantity_sub_equal(a2, a4)

    def test_zip(self, array: SeriesType):
        """Test `Series.zip`."""
        z = array.zip()
        assert_array_equal(
            z,
            numpy.column_stack((array.xindex.value, array.value)),
        )

    def test_crop(self, array: SeriesType):
        """Test basic functionality of `Series.crop`."""
        # all defaults
        utils.assert_quantity_equal(array, array.crop())

        # normal operation
        a2 = array.crop(10, 20)
        utils.assert_quantity_equal(array[10:20], a2)

    def test_crop_warnings(self, array: SeriesType):
        """Test that `Series.crop` emits warnings when it is supposed to."""
        with pytest.warns(
            UserWarning,
            match="crop given start",
        ):
            array.crop(array.xspan[0] - 1, array.xspan[1])
        with pytest.warns(
            UserWarning,
            match="crop given end",
        ):
            array.crop(array.xspan[0], array.xspan[1] + 1)

    def test_crop_irregular(self):
        """Test `Series.crop` with an irregular index."""
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

    def test_crop_float_precision_last_value(self):
        """Verify the float precision of Series.crop given the last index.

        This tests against regression of
        https://gitlab.com/gwpy/gwpy/-/issues/1601.
        """
        # construct empty data array with the right shape for this array object
        shape = (101,) * self.TEST_CLASS._ndim
        series = self.TEST_CLASS(numpy.empty(shape), dx=0.01)

        # assert that when we crop it, we only crop a single sample
        cropped = series.crop(end=1.0)
        utils.assert_quantity_equal(series[:-1], cropped)

    def test_crop_float_precision_last_value_float(self):
        """Verify the float precision of the crop function with float end.

        This tests against regression of
        https://gitlab.com/gwpy/gwpy/-/issues/1656.
        """
        arrlen = 500
        xmax = 0.508463154883984
        x_series = numpy.linspace(0, xmax, arrlen)
        series = Series([0] * arrlen, xindex=x_series)
        expected = series.xindex[-2]
        assert series.crop(end=xmax).xindex[-1] == expected

    def test_crop_between_grid_points_is_floored(self):
        """Test that when we crop between xindex values, the result is floored.

        This tests against regression of
        https://gitlab.com/gwpy/gwpy/-/issues/1656.
        """
        # e.g. x = [1, 2, 3], end = 2.5, result = [1, 2]
        series = Series([0] * 3, xindex=[1, 2, 3])
        assert_array_equal(series.crop(end=2.5).xindex, [1, 2])

        series = Series([0] * 3, xindex=[1, 2, 3])
        assert_array_equal(series.crop(start=2.5).xindex, [3])

        series = Series([0] * 5, xindex=[1, 2, 3, 4, 5])
        assert_array_equal(series.crop(start=2.5, end=4.5).xindex, [3, 4])

    def test_crop_float_precision_near_last_value_float(self):
        """Test the float precision of Series.crop with arg just under end.

        This tests against regression of
        https://gitlab.com/gwpy/gwpy/-/issues/1656.
        """
        arrlen = 500
        xmax = 0.508463154883984
        x_series = numpy.linspace(0, xmax, arrlen)
        series = Series([0] * arrlen, xindex=x_series)
        expected = series.xindex[-2]
        mid = 0.6 * series.xindex[-2] + 0.4 * series.xindex[-1]
        assert series.crop(end=mid).xindex[-1] == expected

    def test_crop_float_precision_first_value_float(self):
        """Verify the float precision of the crop function with float start.

        This tests against regression of
        https://gitlab.com/gwpy/gwpy/-/issues/1656.
        """
        arrlen = 500
        xmin = 0.508463154883984
        x_series = numpy.linspace(xmin, 1.0, arrlen)
        series = Series([0] * arrlen, xindex=x_series)
        expected = series.xindex[0]
        assert series.crop(start=xmin).xindex[0] == expected

    def test_is_compatible(self, array: SeriesType):
        """Test the `Series.is_compatible` method."""
        other = self.create(name="TEST CASE 2")
        assert array.is_compatible(other)

    def test_is_compatible_error_dx(self, array: SeriesType):
        """Check that `Series.is_compatible` errors with mismatching ``dx``."""
        other = self.create(dx=2)
        with pytest.raises(ValueError, match="sample sizes do not match"):
            array.is_compatible(other)

    def test_is_compatible_error_unit(self, array: SeriesType):
        """Check that `Series.is_compatible` errors with mismatching ``unit``."""
        other = self.create(unit="m")
        with pytest.raises(ValueError, match="units do not match"):
            array.is_compatible(other)

    def test_is_compatible_xindex(self):
        """Check that irregular arrays are compatible if their xindexes match."""
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        a = self.create(xindex=x)
        b = self.create(xindex=x)
        assert a.is_compatible(b)

    def test_is_compatible_error_xindex(self, array: SeriesType):
        """Check that `Series.is_compatible` errors with mismatching indexes."""
        x = numpy.logspace(0, 2, num=self.data.shape[0])
        other = self.create(xindex=x)
        with pytest.raises(ValueError, match="indexes do not match"):
            array.is_compatible(other)

    def test_is_contiguous(self, array: SeriesType):
        """Test `Series.is_contiguous`."""
        a2 = self.create(x0=array.xspan[1])
        assert array.is_contiguous(a2) == 1
        assert array.is_contiguous(a2.value) == 1

        ts3 = self.create(x0=array.xspan[1] + 1)
        assert array.is_contiguous(ts3) == 0

        ts4 = self.create(x0=-array.xspan[1])
        assert array.is_contiguous(ts4) == -1

    def test_append(self, array: SeriesType):
        """Test `Series.append`."""
        a2 = self.create(x0=array.xspan[1])

        # test basic append
        a3 = array.append(a2, inplace=False)
        assert a3.epoch == array.epoch
        assert a3.x0 == array.x0
        assert a3.size == array.size + a2.size
        assert a3.xspan == array.xspan + a2.xspan
        assert_array_equal(a3.value[: array.shape[0]], array.value)
        assert_array_equal(a3.value[-a2.shape[0] :], a2.value)

        # check that appending again causes a problem
        with pytest.raises(
            ValueError,
            match=f"Cannot append discontiguous {type(a3).__name__}",
        ):
            a3.append(array)

        # test appending with one xindex deletes it in the output
        array.xindex  # noqa: B018
        a3 = array.append(a2, inplace=False)
        assert hasattr(a3, "_xindex") is False

        # test appending with both xindex appends as well
        array.xindex  # noqa: B018
        a2.xindex  # noqa: B018
        a3 = array.append(a2, inplace=False)
        assert hasattr(a3, "_xindex")
        assert_array_equal(
            a3.xindex.value,
            numpy.concatenate((array.xindex.value, a2.xindex.value)),
        )

        # test appending with one only and not resize
        del a2.xindex
        a3 = array.append(a2, inplace=False, resize=False)
        assert a3.x0 == array.x0 + array.dx * a2.shape[0]

        # test discontiguous appends - gap='raise'
        a3 = self.create(x0=array.xspan[1] + 1)
        ts4 = array.copy()
        with pytest.raises(
            ValueError,
            match=f"Cannot append discontiguous {type(array).__name__}",
        ):
            array.append(a3)

        # test gap='ignore'
        ts4.append(a3, gap="ignore")
        assert ts4.shape[0] == array.shape[0] + a3.shape[0]
        assert_array_equal(ts4.value, numpy.concatenate((array.value, a3.value)))

        # test gap='pad'
        ts4 = array.copy()
        ts4.append(a3, gap="pad", pad=0)
        assert ts4.shape[0] == array.shape[0] + 1 + a3.shape[0]
        z = numpy.zeros((1, *array.shape[1:]))
        assert_array_equal(
            ts4.value,
            numpy.concatenate((array.value, z, a3.value)),
        )

    def test_prepend(self, array: SeriesType):
        """Test `Series.prepend`."""
        a2 = cast("SeriesType", self.create(x0=array.xspan[1]) * 2)
        a3 = a2.prepend(array, inplace=False)
        assert a3.x0 == array.x0
        assert a3.size == array.size + a2.size
        assert a3.xspan == array.xspan + a2.xspan
        with pytest.raises(
            ValueError,
            match=f"Cannot append discontiguous {type(a3).__name__}",
        ):
            a3.prepend(array)
        assert_array_equal(a3.value[: array.shape[0]], array.value)
        assert_array_equal(a3.value[-a2.shape[0] :], a2.value)

    def test_update(self):
        """Test `Series.update`."""
        ts1 = self.create()
        ts2 = self.create(x0=ts1.xspan[1])[: ts1.size // 2]
        ts3 = ts1.update(ts2, inplace=False)
        assert ts3.x0 == ts1.x0 + abs(ts2.xspan) * ts1.x0.unit
        assert ts3.size == ts1.size
        with pytest.raises(
            ValueError,
            match=f"Cannot append discontiguous {type(ts3).__name__}",
        ):
            ts3.update(ts1)

    def test_pad(self):
        """Test `Series.pad`."""
        ts1 = self.create()
        ts2 = ts1.pad(10)
        assert ts2.shape[0] == ts1.shape[0] + 20
        assert_array_equal(
            ts2.value,
            numpy.concatenate((numpy.zeros(10), ts1.value, numpy.zeros(10))),
        )
        assert ts2.x0 == ts1.x0 - 10 * ts1.x0.unit

    def test_pad_index(self):
        """Check that `Series.pad` correctly returns a padded index array."""
        ts1 = self.create()
        ts1.xindex  # <- create the index for ts1  # noqa: B018

        ts2 = ts1.pad(10)

        start: float
        end: float
        start, end = ts2.xspan
        assert_array_equal(
            ts2.xindex,
            Index(
                numpy.linspace(start, end, num=ts2.shape[0], endpoint=False),
                unit=ts1.xindex.unit,
            ),
        )

    def test_pad_asymmetric(self):
        """Test `Series.pad` with asymmetric padding."""
        ts1 = self.create()
        ts2 = ts1.pad((20, 10))
        assert ts2.shape[0] == ts1.shape[0] + 30
        assert_array_equal(
            ts2.value,
            numpy.concatenate((numpy.zeros(20), ts1.value, numpy.zeros(10))),
        )
        assert ts2.x0 == ts1.x0 - 20 * ts1.x0.unit

    def test_diff(self, array: SeriesType):
        """Test `Series.diff`."""
        """Test the `Series.diff` method.

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
        """Test `Series.value_at`."""
        ts1 = self.create(dx=0.5)
        assert ts1.value_at(1.5) == self.data[3] * ts1.unit
        assert ts1.value_at(1.5 * ts1.xunit) == self.data[3] * ts1.unit
        with pytest.raises(IndexError):
            ts1.value_at(1.6)
        # test TimeSeries unit conversion
        if ts1.xunit == units.s:
            assert ts1.value_at(1500 * units.millisecond) == (self.data[3] * ts1.unit)
        # test FrequencySeries unit conversion
        elif ts1.xunit == units.Hz:
            assert ts1.value_at(1500 * units.milliHertz) == (self.data[3] * ts1.unit)

    def test_shift(self):
        """Test `Series.shift`."""
        a = self.create(x0=0, dx=1, xunit="s")
        x0 = a.x0.copy()
        a.shift(5)
        assert a.x0 == x0 + 5 * x0.unit

        a.shift("1 hour")
        assert a.x0 == x0 + 3605 * x0.unit

        a.shift(-0.007)
        assert a.x0 == x0 + (3604.993) * x0.unit

        with pytest.raises(
            ValueError,
            match=r"'Hz' \(frequency\) and 's' \(time\) are not convertible",
        ):
            a.shift("1 Hz")

    @pytest.mark.parametrize("inplace", [True, False])
    def test_inject(self, inplace: bool):
        """Test `Series.inject()`."""
        # Create a series of zeros (supporting multi-dim series)
        duration = 16
        ndim = self.TEST_CLASS._ndim
        nsamp = duration * ndim
        shape = () if ndim == 1 else (ndim,)
        zshape = (duration, *shape)
        data = self.TEST_CLASS(numpy.zeros(zshape))

        # Create a second series to inject into the first
        ishape = (duration // 2, *shape)
        start = duration // 4
        injection = self.TEST_CLASS(
            numpy.linspace(1, 2, num=nsamp // 2).reshape(ishape),
            x0=start + .1,  # non-grid-aligned start time
        )

        # Test that we recover this injection when we add it to data
        with pytest.warns(
            UserWarning,
            match="will round",
        ):
            new = data.inject(injection, inplace=inplace)
        assert new.unit == data.unit
        assert new.size == data.size

        # Check that the injection is where we expect it to be
        nzind = new.value.nonzero()
        assert nzind[0].size == injection.size
        first = tuple(nzind[i][0] for i in range(len(nzind)))
        assert first[0] == start * data.dx.value

        # Check that the injected values are correct
        utils.assert_allclose(new.value[nzind], injection.value.flatten())

        # Check that inplace does what we expect
        if inplace:
            assert new is data
        else:
            utils.assert_allclose(data.value, numpy.zeros(zshape))
