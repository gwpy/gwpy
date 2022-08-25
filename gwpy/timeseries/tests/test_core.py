# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

import operator
from functools import reduce
from io import BytesIO

import pytest

import numpy
from numpy import shares_memory

from matplotlib import rc_context

from astropy import units

from ...detector import Channel
from ...segments import (Segment, SegmentList)
from ...testing import (mocks, utils)
from ...time import Time
from ...types.tests.test_series import TestSeries as _TestSeries
from .. import (TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList)

numpy.random.seed(1)
GPS_EPOCH = Time(0, format='gps', scale='utc')

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- TimeSeriesBase -----------------------------------------------------------

class TestTimeSeriesBase(_TestSeries):
    TEST_CLASS = TimeSeriesBase

    def test_new(self):
        """Test `gwpy.timeseries.TimeSeriesBase` constructor
        """
        array = super().test_new()

        # check time-domain metadata
        assert array.epoch == GPS_EPOCH
        assert array.sample_rate == units.Quantity(1, 'Hertz')
        assert array.dt == units.Quantity(1, 'second')

        # check handling of epoch vs t0
        a = self.create(epoch=10)
        b = self.create(t0=10)
        utils.assert_quantity_sub_equal(a, b)
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS(self.data, epoch=1, t0=1)
        assert str(exc.value) == 'give only one of epoch or t0'

        # check handling of sample_rate vs dt
        a = self.create(sample_rate=100)
        b = self.create(dt=0.01)
        utils.assert_quantity_sub_equal(a, b)
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS(self.data, sample_rate=1, dt=1)
        assert str(exc.value) == 'give only one of sample_rate or dt'

    def test_epoch(self):
        """Test `gwpy.timeseries.TimeSeriesBase.epoch`
        """
        # check basic conversion from t0 -> epoch
        a = self.create(t0=1126259462)
        assert a.epoch == Time('2015-09-14 09:50:45', format='iso')

        # test that we can't delete epoch
        with pytest.raises(AttributeError):
            del a.epoch

        # check None gets preserved
        a.epoch = None
        with pytest.raises(AttributeError):
            a._t0

        # check other types
        a.epoch = Time('2015-09-14 09:50:45', format='iso')
        utils.assert_quantity_almost_equal(
            a.t0, units.Quantity(1126259462, 's'))

    def test_sample_rate(self):
        """Test `gwpy.timeseries.TimeSeriesBase.sample_rate`
        """
        # check basic conversion from dt -> sample_rate
        a = self.create(dt=0.5)
        assert a.sample_rate == 2 * units.Hz

        # test that we can't delete sample_rate
        with pytest.raises(AttributeError):
            del a.sample_rate

        # check None gets preserved
        a.sample_rate = None
        with pytest.raises(AttributeError):
            a._t0

        # check other types
        a.sample_rate = units.Quantity(128, units.Hz)
        utils.assert_quantity_equal(a.dt, units.s / 128.)
        a.sample_rate = units.Quantity(16.384, units.kiloHertz)
        utils.assert_quantity_equal(a.dt, units.s / 16384)

    def test_duration(self, array):
        assert array.duration == array.t0 + array.shape[0] * array.dt

    # -- test methods ---------------------------

    def test_plot(self, array):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot()
            line = plot.gca().lines[0]
            utils.assert_array_equal(line.get_xdata(), array.xindex.value)
            utils.assert_array_equal(line.get_ydata(), array.value)
            plot.save(BytesIO(), format='png')
            plot.close()

    @pytest.mark.requires("nds2")
    def test_from_nds2_buffer(self):
        # build fake buffer
        nds_buffer = mocks.nds2_buffer(
            'X1:TEST',
            self.data,
            1000000000,
            self.data.shape[0],
            'm',
            name='test',
            slope=2,
            offset=1,
        )

        # convert to TimeSeries
        a = self.TEST_CLASS.from_nds2_buffer(nds_buffer)

        # check everything works (including default dynamic scaling)
        assert isinstance(a, self.TEST_CLASS)
        assert not shares_memory(a.value, nds_buffer.data)
        utils.assert_array_equal(a.value, nds_buffer.data * 2 + 1)
        assert a.t0 == 1000000000 * units.s
        assert a.dt == units.s / nds_buffer.data.shape[0]
        assert a.name == 'test'
        assert a.channel == Channel(
            'X1:TEST',
            sample_rate=self.data.shape[0],
            unit='m',
            type='raw',
            dtype='float32',
        )

        # check that we can use keywords to override settings
        b = self.TEST_CLASS.from_nds2_buffer(nds_buffer, scaled=False,
                                             copy=False, sample_rate=128)
        assert b.dt == 1/128. * units.s
        assert shares_memory(nds_buffer.data, b.value)

    @pytest.mark.requires("lal")
    def test_to_from_lal(self, array):
        # check that to + from returns the same array
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts)
        utils.assert_quantity_sub_equal(array, a2, exclude=['channel'])

    @pytest.mark.requires("lal")
    @pytest.mark.parametrize("copy", (False, True))
    def test_to_from_lal_no_copy(self, array, copy):
        """Check that copy=False shares data
        """
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts, copy=copy)
        assert shares_memory(a2.value, lalts.data.data) is not copy

    @pytest.mark.requires("lal")
    def test_to_from_lal_unrecognised_units(self, array):
        """Test that unrecognised units get warned, but the operation continues
        """
        import lal
        array.override_unit('undef')
        with pytest.warns(UserWarning):
            lalts = array.to_lal()
        assert lalts.sampleUnits == lal.DimensionlessUnit
        a2 = self.TEST_CLASS.from_lal(lalts)
        assert a2.unit == units.dimensionless_unscaled

    @pytest.mark.requires("lal")
    def test_to_from_lal_pow10_units(self, array):
        """Test that normal scaled units scale the data properly
        """
        import lal
        array.override_unit("km")
        lalts = array.to_lal()
        utils.assert_array_equal(lalts.data.data, array.value)
        assert lalts.sampleUnits == lal.MeterUnit * 1000.

    @pytest.mark.requires("lal")
    def test_to_from_lal_scaled_units(self, array):
        """Test that weird scaled units scale the data properly
        """
        import lal
        array.override_unit("123 m")
        lalts = array.to_lal()
        utils.assert_array_equal(lalts.data.data, array.value * 123.)
        assert lalts.sampleUnits == lal.MeterUnit

    @pytest.mark.requires("lal", "pycbc")
    def test_to_from_pycbc(self, array):
        from pycbc.types import TimeSeries as PyCBCTimeSeries

        # test default conversion
        pycbcts = array.to_pycbc()
        assert isinstance(pycbcts, PyCBCTimeSeries)
        utils.assert_array_equal(array.value, pycbcts.data)
        assert array.t0.value == pycbcts.start_time
        assert array.dt.value == pycbcts.delta_t

        # go back and check we get back what we put in in the first place
        a2 = type(array).from_pycbc(pycbcts)
        utils.assert_quantity_sub_equal(
            array, a2, exclude=['name', 'unit', 'channel'])

        # test copy=False
        a2 = type(array).from_pycbc(array.to_pycbc(copy=False), copy=False)
        assert shares_memory(array.value, a2.value)


# -- TimeSeriesBaseDict -------------------------------------------------------

class TestTimeSeriesBaseDict(object):
    TEST_CLASS = TimeSeriesBaseDict
    ENTRY_CLASS = TimeSeriesBase
    DTYPE = None

    @classmethod
    def create(cls):
        new = cls.TEST_CLASS()
        new['a'] = cls.ENTRY_CLASS(numpy.random.normal(size=200),
                                   name='a', x0=0, dx=1, dtype=cls.DTYPE)
        new['b'] = cls.ENTRY_CLASS(numpy.random.normal(size=2000),
                                   name='b', x0=0, dx=.1, dtype=cls.DTYPE)
        return new

    @pytest.fixture()
    def instance(self):
        return self.create()

    def test_series_link(self):
        assert self.ENTRY_CLASS.DictClass is self.TEST_CLASS
        assert self.TEST_CLASS.EntryClass is self.ENTRY_CLASS

    def test_span(self, instance):
        assert isinstance(instance.span, Segment)
        assert instance.span == reduce(
            operator.or_, (ts.span for ts in instance.values()), Segment(0, 0),
        )
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS().span
        assert 'cannot calculate span for empty ' in str(exc.value)

    def test_copy(self, instance):
        copy = instance.copy()
        assert isinstance(copy, self.TEST_CLASS)
        for key in copy:
            assert not shares_memory(copy[key].value, instance[key].value)
            utils.assert_quantity_sub_equal(copy[key], instance[key])

    def test_append(self, instance):
        # test appending from empty (with and without copy)
        for copy in (True, False):
            new = type(instance)()
            new.append(instance, copy=copy)
            for key in new:
                assert shares_memory(new[key].value,
                                     instance[key].value) is not copy
                utils.assert_quantity_sub_equal(new[key], instance[key])

        # create copy of dict that is contiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)([1, 2, 3, 4, 5], x0=a.xspan[1], dx=a.dx,
                               dtype=a.dtype)

        # append and test
        b = instance.copy()
        b.append(new)
        for key in b:
            utils.assert_array_equal(
                b[key].value,
                numpy.concatenate((instance[key].value, new[key].value)))

        # create copy of dict that is discontiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)([1, 2, 3, 4, 5], x0=a.xspan[1]+a.dx.value,
                               dx=a.dx, dtype=a.dtype)
        # check error
        with pytest.raises(ValueError):
            instance.append(new)

        # check padding works (don't validate too much, that is tested
        # elsewhere)
        b = instance.copy()
        b.append(new, pad=0, gap='pad')

    def test_prepend(self, instance):
        # test appending from empty (with and without copy)
        new = type(instance)()
        new.prepend(instance)
        for key in new:
            assert shares_memory(new[key].value, instance[key].value)
            utils.assert_quantity_sub_equal(new[key], instance[key])

        # create copy of dict that is contiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)([1, 2, 3, 4, 5], x0=a.xspan[1], dx=a.dx,
                               dtype=a.dtype)
        # append and test
        b = new.copy()
        b.prepend(instance)
        for key in b:
            utils.assert_array_equal(
                b[key].value,
                numpy.concatenate((instance[key].value, new[key].value)))

        # create copy of dict that is discontiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)([1, 2, 3, 4, 5], x0=a.xspan[1], dx=a.dx,
                               dtype=a.dtype)
        # check error
        with pytest.raises(ValueError):
            new.append(instance)
        # check padding works (don't validate too much, that is tested
        # elsewhere)
        b = new.copy()
        b.prepend(instance, pad=0)

    def test_crop(self, instance):
        """Test :meth:`TimeSeriesBaseDict.crop`
        """
        a = instance.copy().crop(10, 20)  # crop() modifies in-place
        for key in a:
            utils.assert_quantity_sub_equal(a[key], instance[key].crop(10, 20))

    def test_resample(self, instance):
        if self.ENTRY_CLASS is TimeSeriesBase:  # currently only for subclasses
            return NotImplemented
        a = instance.resample(.5)
        for key in a:
            assert a[key].dx == 1/.5 * a[key].xunit

    def test_fetch(self):
        return NotImplemented

    def test_find(self):
        return NotImplemented

    def test_get(self):
        return NotImplemented

    @pytest.mark.requires("nds2")
    def test_from_nds2_buffers(self):
        buffers = [
            mocks.nds2_buffer('X1:TEST', numpy.arange(100), 1000000000,
                              1, 'm'),
            mocks.nds2_buffer('X1:TEST2', numpy.arange(100, 200), 1000000100,
                              1, 'm'),
        ]
        a = self.TEST_CLASS.from_nds2_buffers(buffers)
        assert isinstance(a, self.TEST_CLASS)
        assert a['X1:TEST'].x0.value == 1000000000
        assert a['X1:TEST2'].dx.value == 1
        assert a['X1:TEST2'].x0.value == 1000000100

        a = self.TEST_CLASS.from_nds2_buffers(buffers, sample_rate=.01)
        assert a['X1:TEST'].dx.value == 100

    def test_plot(self, instance):
        with rc_context(rc={'text.usetex': False}):
            plot = instance.plot()
            for line, key in zip(plot.gca().lines, instance):
                utils.assert_array_equal(line.get_xdata(),
                                         instance[key].xindex.value)
                utils.assert_array_equal(line.get_ydata(),
                                         instance[key].value)
            plot.save(BytesIO(), format='png')
            plot.close()


# -- TimeSeriesBaseList -------------------------------------------------------

class TestTimeSeriesBaseList(object):
    TEST_CLASS = TimeSeriesBaseList
    ENTRY_CLASS = TimeSeriesBase
    DTYPE = None

    @classmethod
    def create(cls):
        new = cls.TEST_CLASS()
        new.append(cls.ENTRY_CLASS(numpy.random.normal(size=100),
                                   x0=0, dx=1, dtype=cls.DTYPE))
        new.append(cls.ENTRY_CLASS(numpy.random.normal(size=1000),
                                   x0=101, dx=1, dtype=cls.DTYPE))
        return new

    @pytest.fixture()
    def instance(self):
        return self.create()

    def test_series_link(self):
        assert self.TEST_CLASS.EntryClass is self.ENTRY_CLASS

    def test_segments(self, instance):
        """Test :attr:`gwpy.timeseries.TimeSeriesBaseList.segments`
        """
        sl = instance.segments
        assert isinstance(sl, SegmentList)
        assert all(isinstance(s, Segment) for s in sl)
        assert sl == [(0, 100), (101, 1101)]

    def test_append(self):
        tsl = self.create()

        # test simple append
        new = self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=1102, dx=1)
        tsl.append(new)

        # test mismatched type raises error
        with pytest.raises(TypeError) as exc:
            tsl.append([1, 2, 3, 4, 5])
        assert str(exc.value) == (
            "Cannot append type 'list' to %s" % type(tsl).__name__)

    def test_extend(self):
        a = self.create()
        b = a.copy()
        new = self.ENTRY_CLASS([1, 2, 3, 4, 5])
        a.append(new)
        b.extend([new])
        for i in range(max(map(len, (a, b)))):
            utils.assert_quantity_sub_equal(a[i], b[i])

    def test_coalesce(self):
        a = self.TEST_CLASS()
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=0, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=11, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=5, dx=1))
        a.coalesce()
        assert len(a) == 2
        assert a[0].span == (0, 10)
        utils.assert_array_equal(a[0].value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

    def test_join(self):
        a = self.TEST_CLASS()
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=0, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=5, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=11, dx=1))

        # disjoint list should throw error
        with pytest.raises(ValueError):
            a.join()

        # but we can pad to get rid of the errors
        t = a.join(gap='pad')
        assert isinstance(t, a.EntryClass)
        assert t.span == (0, 16)
        utils.assert_array_equal(
            t.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])

        # check that joining empty list produces something sensible
        t = self.TEST_CLASS().join()
        assert isinstance(t, self.TEST_CLASS.EntryClass)
        assert t.size == 0

    def test_slice(self, instance):
        s = instance[:2]
        assert type(s) is type(instance)

    def test_copy(self, instance):
        a = instance.copy()
        assert type(a) is type(instance)
        for x, y in zip(instance, a):
            utils.assert_quantity_sub_equal(x, y)
            assert not shares_memory(x.value, y.value)
