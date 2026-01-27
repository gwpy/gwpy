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

"""Unit test for timeseries module."""

from __future__ import annotations

import operator
from functools import reduce
from io import BytesIO
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

import numpy
import pytest
from astropy import units
from matplotlib import rc_context
from numpy import shares_memory

from ...detector import Channel
from ...segments import Segment, SegmentList
from ...testing import mocks, utils
from ...time import Time
from ...types.tests.test_series import TestSeries as _TestSeries
from .. import TimeSeriesBase, TimeSeriesBaseDict, TimeSeriesBaseList

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

GPS_EPOCH = Time(0, format="gps", scale="utc")
RNG = numpy.random.default_rng(seed=1)

TimeSeriesBaseType = TypeVar("TimeSeriesBaseType", bound=TimeSeriesBase)
TimeSeriesBaseDictType = TypeVar(
    "TimeSeriesBaseDictType",
    bound=TimeSeriesBaseDict[TimeSeriesBase],
)
TimeSeriesBaseListType = TypeVar("TimeSeriesBaseListType", bound=TimeSeriesBaseList)
EntryType = TypeVar("EntryType", bound=TimeSeriesBase)


# -- TimeSeriesBase ------------------

class TestTimeSeriesBase(_TestSeries[TimeSeriesBaseType], Generic[TimeSeriesBaseType]):
    """Test `TimeSeriesBase`."""

    TEST_CLASS: type[TimeSeriesBaseType] = TimeSeriesBase

    def test_new(self):
        """Test `gwpy.timeseries.TimeSeriesBase` constructor."""
        array = self.create()
        super().test_new()

        # check time-domain metadata
        assert array.epoch == GPS_EPOCH
        assert array.sample_rate == units.Quantity(1, "Hertz")
        assert array.dt == units.Quantity(1, "second")

    def test_new_epoch_t0(self):
        """Test `gwpy.timeseries.TimeSeriesBase` handling of epoch vs t0."""
        a = self.create(epoch=10)
        b = self.create(t0=10)
        utils.assert_quantity_sub_equal(a, b)
        with pytest.raises(
            ValueError,
            match=r"^give only one of epoch or t0$",
        ):
            self.TEST_CLASS(self.data, epoch=1, t0=1)

    def test_new_sample_rate_dt(self):
        """Test `gwpy.timeseries.TimeSeriesBase` handling of sample_rate vs dt."""
        # check handling of sample_rate vs dt
        a = self.create(sample_rate=100)
        b = self.create(dt=0.01)
        utils.assert_quantity_sub_equal(a, b)
        with pytest.raises(
            ValueError,
            match=r"^give only one of sample_rate or dt$",
        ):
            self.TEST_CLASS(self.data, sample_rate=1, dt=1)

    def test_epoch(self):  # type: ignore[override]
        """Test `gwpy.timeseries.TimeSeriesBase.epoch`."""
        # check basic conversion from t0 -> epoch
        a = self.create(t0=1126259462)
        assert a.epoch == Time("2015-09-14 09:50:45", format="iso")

        # test that we can't delete epoch
        with pytest.raises(AttributeError):
            del a.epoch

        # check None gets preserved
        a.epoch = None
        with pytest.raises(AttributeError):
            a._t0  # noqa: B018

        # check other types
        a.epoch = Time("2015-09-14 09:50:45", format="iso")
        utils.assert_quantity_almost_equal(
            a.t0,
            units.Quantity(1126259462, "s"),
        )

    def test_sample_rate(self):
        """Test `gwpy.timeseries.TimeSeriesBase.sample_rate`."""
        # check basic conversion from dt -> sample_rate
        a = self.create(dt=0.5)
        assert a.sample_rate == 2 * units.Hz

    def test_sample_rate_del(self, array):
        """Test that `sample_rate` cannot be deleted."""
        # test that we can't delete sample_rate
        with pytest.raises(
            AttributeError,
            match=r"(can't delete attribute|has no deleter)",
        ):
            del array.sample_rate

    def test_sample_rate_none(self, array):
        """Test that `sample_rate = None` is effectively a deletion."""
        # check None gets preserved
        array.sample_rate = None
        with pytest.raises(AttributeError, match="_t0"):
            array._t0  # noqa: B018

    @pytest.mark.parametrize(("samp", "dt"), [
        (128 * units.Hz, units.s / 128.),
        (16.384 * units.kiloHertz, units.s / 16384),
        (10 / units.s, units.s / 10),
    ])
    def test_sample_rate_type(self, array, samp, dt):
        """Test that units and types are handled when setting `sample_rate`."""
        array.sample_rate = samp
        utils.assert_quantity_equal(array.dt, dt)

    def test_sample_rate_ghz(self, array):
        """Test that very large sample rates don't get rounded to dt=0.

        Regression: https://gitlab.com/gwpy/gwpy/-/issues/1646
        """
        array.sample_rate = 1e9
        assert array.dt.value > 0.

    def test_duration(self, array):
        """Test `TimeSeriesBase.duration`."""
        assert array.duration == array.t0 + array.shape[0] * array.dt

    # -- test i/o ---------------------

    def test_read_write_csv(self, array: TimeSeriesBaseType):  # noqa: ARG002
        """Test reading and writing CSV files."""
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")

    # -- test methods -----------------

    def test_plot(self, array):
        """Test `TimeSeriesBase.plot`."""
        with rc_context(rc={"text.usetex": False}):
            plot = array.plot()
            line = plot.gca().lines[0]
            utils.assert_array_equal(line.get_xdata(), array.xindex.value)
            utils.assert_array_equal(line.get_ydata(), array.value)
            plot.save(BytesIO(), format="png")
            plot.close()

    @pytest.mark.requires("arrakis")
    @pytest.mark.parametrize("copy", [False, True])
    @pytest.mark.parametrize("dtype", [
        pytest.param(numpy.float64, id="float64"),
        pytest.param(numpy.int32, id="int32"),
        pytest.param(bool, id="bool"),
        pytest.param(numpy.dtype('bool'), id="np_bool"),
    ])
    def test_from_arrakis(self, copy, dtype):
        """Test `TimeSeriesBase.from_arrakis`."""
        from arrakis import Channel as ArrakisChannel
        from arrakis.block import Series as ArrakisSeries

        data = self.data.astype(dtype)

        # create arrakis objects
        achan = ArrakisChannel(
            "X1:TEST-CHANNEL",
            dtype,
            128,
        )
        aseries = ArrakisSeries(
            data=data,
            time_ns=int(1e18),
            channel=achan,
        )

        # convert to TimeSeries
        ts = self.TEST_CLASS.from_arrakis(aseries, copy=copy)

        # check data
        utils.assert_array_equal(ts.value, data)
        assert shares_memory(ts.value, aseries.data) is not copy

        # check metadata
        assert ts.name == aseries.name
        assert ts.t0 == aseries.t0 * units.s
        assert ts.dt == aseries.dt * units.s
        assert ts.sample_rate == aseries.sample_rate * units.Hz
        assert ts.dtype == numpy.dtype(dtype)
        assert abs(ts.span) == aseries.duration
        utils.assert_array_equal(ts.times.value, aseries.times)
        # see https://git.ligo.org/ngdd/arrakis-python/-/issues/22:
        assert ts.unit == units.dimensionless_unscaled

    @pytest.mark.requires("nds2")
    def test_from_nds2_buffer(self):
        """Test `TimeSeriesBase.from_nds2_buffer`."""
        # build fake buffer
        nds_buffer = mocks.nds2_buffer(
            "X1:TEST",
            self.data,
            1000000000,
            self.data.shape[0],
            "m",
            name="test",
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
        assert a.name == "test"
        assert a.channel == Channel(
            "X1:TEST",
            sample_rate=self.data.shape[0],
            unit="m",
            type="raw",
            dtype="float32",
        )

        # check that we can use keywords to override settings
        b = self.TEST_CLASS.from_nds2_buffer(
            nds_buffer,
            scaled=False,
            copy=False,
            sample_rate=128,
        )
        assert b.dt == 1/128. * units.s
        assert shares_memory(nds_buffer.data, b.value)

    @pytest.mark.requires("lal")
    def test_to_from_lal(self, array):
        """Test `TimeSeriesBase.to_lal` and `TimeSeriesBase.from_lal`."""
        # check that to + from returns the same array
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts)
        utils.assert_quantity_sub_equal(array, a2, exclude=["channel"])

    @pytest.mark.requires("lal")
    @pytest.mark.parametrize("copy", [False, True])
    def test_to_from_lal_no_copy(self, array, copy):
        """Check that copy=False shares data."""
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts, copy=copy)
        assert shares_memory(a2.value, lalts.data.data) is not copy

    @pytest.mark.requires("lal")
    def test_to_from_lal_unrecognised_units(self, array):
        """Test that unrecognised units get warned, but the operation continues."""
        import lal
        array.override_unit("undef")
        with pytest.warns(
            UserWarning,
            match="defaulting to lal.DimensionlessUnit",
        ):
            lalts = array.to_lal()
        assert lalts.sampleUnits == lal.DimensionlessUnit
        a2 = self.TEST_CLASS.from_lal(lalts)
        assert a2.unit == units.dimensionless_unscaled

    def test_to_from_lal_pow10_units(self, array):
        """Test that normal scaled units scale the data properly."""
        lal = pytest.importorskip("lal")
        array.override_unit("km")
        lalts = array.to_lal()
        utils.assert_array_equal(lalts.data.data, array.value)
        assert lalts.sampleUnits == lal.MeterUnit * 1000.

    def test_to_from_lal_scaled_units(self, array):
        """Test that weird scaled units scale the data properly."""
        lal = pytest.importorskip("lal")
        array.override_unit("123 m")
        lalts = array.to_lal()
        utils.assert_array_equal(lalts.data.data, array.value * 123.)
        assert lalts.sampleUnits == lal.MeterUnit

    @pytest.mark.requires("lal", "pycbc")
    def test_to_from_pycbc(self, array):
        """Test `TimeSeriesBase.to_pycbc` and `TimeSeriesBase.from_pycbc`."""
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
            array, a2, exclude=["name", "unit", "channel"])

        # test copy=False
        a2 = type(array).from_pycbc(array.to_pycbc(copy=False), copy=False)
        assert shares_memory(array.value, a2.value)


# -- TimeSeriesBaseDict --------------

class TestTimeSeriesBaseDict(Generic[TimeSeriesBaseDictType, EntryType]):
    """Test base class for TimeSeriesBaseDict."""

    TEST_CLASS: type[TimeSeriesBaseDictType] = TimeSeriesBaseDict
    ENTRY_CLASS: type[EntryType] = TimeSeriesBase
    DTYPE: DTypeLike = None

    @classmethod
    def create(cls) -> TimeSeriesBaseDictType:
        """Create a new `TimeSeriesBaseDict` instance."""
        new = cls.TEST_CLASS()
        new["a"] = cls.ENTRY_CLASS(
            RNG.normal(size=200),
            name="a",
            x0=0,
            dx=1,
            dtype=cls.DTYPE,
        )
        new["b"] = cls.ENTRY_CLASS(
            RNG.normal(size=2000),
            name="b",
            x0=0,
            dx=.1,
            dtype=cls.DTYPE,
        )
        return new

    @pytest.fixture
    def instance(self) -> TimeSeriesBaseDictType:
        """Create an instance of a `TimeSeriesBaseDict`."""
        return self.create()

    def test_series_link(self):
        """Test the links between `DictClass` and `EntryClass`."""
        assert self.ENTRY_CLASS.DictClass is self.TEST_CLASS
        assert self.TEST_CLASS.EntryClass is self.ENTRY_CLASS

    def test_span(self, instance):
        """Test `TimeSeriesBaseDict.span`."""
        assert isinstance(instance.span, Segment)
        assert instance.span == reduce(
            operator.or_, (ts.span for ts in instance.values()), Segment(0, 0),
        )

    def test_span_error_empty(self):
        """Test that `TimeSeriesBaseDict.span` raises for empty dict."""
        with pytest.raises(
            ValueError,
            match="cannot calculate span for empty ",
        ):
            self.TEST_CLASS().span  # noqa: B018

    def test_copy(self, instance):
        """Test `TimeSeriesBaseDict.copy`."""
        copy = instance.copy()
        assert isinstance(copy, self.TEST_CLASS)
        for key in copy:
            assert not shares_memory(copy[key].value, instance[key].value)
            utils.assert_quantity_sub_equal(copy[key], instance[key])

    def test_append(self, instance):
        """Test `TimeSeriesBaseDict.append`."""
        # test appending from empty (with and without copy)
        for copy in (True, False):
            new = type(instance)()
            new.append(instance, copy=copy)
            for key in new:
                assert shares_memory(
                    new[key].value,
                    instance[key].value,
                ) is not copy
                utils.assert_quantity_sub_equal(new[key], instance[key])

        # create copy of dict that is contiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)(
                [1, 2, 3, 4, 5],
                x0=a.xspan[1],
                dx=a.dx,
                dtype=a.dtype,
            )

        # append and test
        b = instance.copy()
        b.append(new)
        for key in b:
            utils.assert_array_equal(
                b[key].value,
                numpy.concatenate((instance[key].value, new[key].value)),
            )

        # create copy of dict that is discontiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)(
                [1, 2, 3, 4, 5],
                x0=a.xspan[1]+a.dx.value,
                dx=a.dx,
                dtype=a.dtype,
            )
        # check error
        with pytest.raises(
            ValueError,
            match="Cannot append discontiguous",
        ):
            instance.append(new)

        # check padding works (don't validate too much, that is tested
        # elsewhere)
        b = instance.copy()
        b.append(new, pad=0, gap="pad")

    def test_prepend(self, instance):
        """Test `TimeSeriesBaseDict.prepend`."""
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
            new[key] = type(a)(
                [1, 2, 3, 4, 5],
                x0=a.xspan[1],
                dx=a.dx,
                dtype=a.dtype,
            )
        # append and test
        b = new.copy()
        b.prepend(instance)
        for key in b:
            utils.assert_array_equal(
                b[key].value,
                numpy.concatenate((instance[key].value, new[key].value)),
            )

        # create copy of dict that is discontiguous
        new = type(instance)()
        for key in instance:
            a = instance[key]
            new[key] = type(a)(
                [1, 2, 3, 4, 5],
                x0=a.xspan[1],
                dx=a.dx,
                dtype=a.dtype,
            )
        # check error
        with pytest.raises(
            ValueError,
            match="Cannot append discontiguous",
        ):
            new.append(instance)
        # check padding works (don't validate too much, that is tested
        # elsewhere)
        b = new.copy()
        b.prepend(instance, pad=0)

    def test_crop(self, instance):
        """Test `TimeSeriesBaseDict.crop`."""
        a = instance.copy().crop(10, 20)  # crop() modifies in-place
        for key in a:
            utils.assert_quantity_sub_equal(a[key], instance[key].crop(10, 20))

    def test_resample(self, instance):
        """Test `TimeSeriesBaseDict.resample`."""
        if self.ENTRY_CLASS is TimeSeriesBase:  # currently only for subclasses
            pytest.skip(f"not implemented for {type(instance).__name__}")

        # for all subclasses
        a = instance.resample(.5)
        for key in a:
            assert a[key].dx == 1/.5 * a[key].xunit

    @pytest.mark.requires("arrakis")
    @pytest.mark.parametrize("copy", [False, True])
    def test_from_arrakis(self, copy):
        """Test :meth:`TimeSeriesBaseDict.from_arrakis`."""
        from arrakis import (
            Channel as ArrakisChannel,
            SeriesBlock as ArrakisBlock,
        )

        # arrakis metadata (channels)
        channels = {
            "X1:TEST-CHANNEL_1": ArrakisChannel(
                "X1:TEST-CHANNEL_1",
                data_type=numpy.dtype(self.DTYPE),
                sample_rate=64,
            ),
            "X1:TEST-CHANNEL_2": ArrakisChannel(
                "X1:TEST-CHANNEL_2",
                data_type=numpy.dtype(self.DTYPE),
                sample_rate=128,
            ),
        }

        # arrakis block
        block = ArrakisBlock(
            int(1e18),
            {
                chan.name: RNG.random(
                    size=int(10 * chan.sample_rate),
                ).astype(self.DTYPE)
                for chan in channels.values()
            },
            channels,
        )

        # convert to TimeSeries
        tsd = self.TEST_CLASS.from_arrakis(block, copy=copy)

        # check keys match
        assert list(tsd) == list(map(str, block.channels))

        for key in tsd:
            # check data
            aseries = block[str(key)]
            ts = tsd[key]
            utils.assert_array_equal(ts.value, aseries.data)
            assert shares_memory(ts.value, aseries.data) is not copy

            # check metadata
            assert ts.name == aseries.name
            assert ts.t0 == aseries.t0 * units.s
            assert ts.dt == aseries.dt * units.s
            assert ts.sample_rate == aseries.sample_rate * units.Hz
            assert abs(ts.span) == aseries.duration
            utils.assert_array_equal(ts.times.value, aseries.times)
            # see https://git.ligo.org/ngdd/arrakis-python/-/issues/22:
            assert ts.unit == units.dimensionless_unscaled

    @pytest.mark.requires("nds2")
    def test_from_nds2_buffers(self):
        """Test `TimeSeriesBaseDict.from_nds2_buffers`."""
        buffers = [
            mocks.nds2_buffer("X1:TEST", numpy.arange(100), 1000000000,
                              1, "m"),
            mocks.nds2_buffer("X1:TEST2", numpy.arange(100, 200), 1000000100,
                              1, "m"),
        ]
        a = self.TEST_CLASS.from_nds2_buffers(buffers)
        assert isinstance(a, self.TEST_CLASS)
        assert a["X1:TEST"].x0.value == 1000000000
        assert a["X1:TEST2"].dx.value == 1
        assert a["X1:TEST2"].x0.value == 1000000100

        a = self.TEST_CLASS.from_nds2_buffers(buffers, sample_rate=.01)
        assert a["X1:TEST"].dx.value == 100

    def test_plot(self, instance):
        """Test `TimeSeriesBaseDict.plot`."""
        with rc_context(rc={"text.usetex": False}):
            plot = instance.plot()
            for line, key in zip(plot.gca().lines, instance, strict=True):
                utils.assert_array_equal(line.get_xdata(),
                                         instance[key].xindex.value)
                utils.assert_array_equal(line.get_ydata(),
                                         instance[key].value)
            plot.save(BytesIO(), format="png")
            plot.close()

    def test_plot_separate(self, instance):
        """Test plotting `TimeSeriesDict` on separate axes.

        See https://gitlab.com/gwpy/gwpy/-/issues/1609
        """
        with rc_context(rc={"text.usetex": False}):
            plot = instance.plot(separate=True)
            assert len(plot.axes) == len(instance.keys())
            for ax, key in zip(plot.axes, instance, strict=True):
                utils.assert_array_equal(ax.lines[-1].get_xdata(),
                                         instance[key].xindex.value)
                utils.assert_array_equal(ax.lines[-1].get_ydata(),
                                         instance[key].value)
            plot.save(BytesIO(), format="png")
            plot.close()


# -- TimeSeriesBaseList --------------

class TestTimeSeriesBaseList(Generic[TimeSeriesBaseListType, EntryType]):
    """Test base class for TimeSeriesBaseList."""

    TEST_CLASS: type[TimeSeriesBaseListType] = TimeSeriesBaseList
    ENTRY_CLASS: type[EntryType] = TimeSeriesBase
    DTYPE: DTypeLike = None

    @classmethod
    def create(cls) -> TimeSeriesBaseListType:
        """Create a new `TimeSeriesList`."""
        new = cls.TEST_CLASS()
        new.append(
            cls.ENTRY_CLASS(
                RNG.normal(size=100),
                x0=0,
                dx=1,
                dtype=cls.DTYPE,
            ),
        )
        new.append(
            cls.ENTRY_CLASS(
                RNG.normal(size=1000),
                x0=101,
                dx=1,
                dtype=cls.DTYPE,
            ),
        )
        return new

    @pytest.fixture
    def instance(self) -> TimeSeriesBaseListType:
        """Create an instance of a `TimeSeriesList`."""
        return self.create()

    def test_series_link(self):
        """Test that `TimeSeriesList.EntryClass` is set properly."""
        assert self.TEST_CLASS.EntryClass is self.ENTRY_CLASS

    def test_segments(self, instance):
        """Test :attr:`TimeSeriesBaseList.segments`."""
        sl = instance.segments
        assert isinstance(sl, SegmentList)
        assert all(isinstance(s, Segment) for s in sl)
        assert sl == [(0, 100), (101, 1101)]

    def test_append(self, instance):
        """Test `TimeSeriesList.append`."""
        new = self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=1102, dx=1)
        instance.append(new)
        assert len(instance) == 3
        assert instance[-1] is new

    def test_append_typeerror(self, instance):
        """Test `TimeSeriesList.append` errors on type differences."""
        with pytest.raises(
            TypeError,
            match=fr"^Cannot append type 'list' to {self.TEST_CLASS.__name__}$",
        ):
            instance.append([1, 2, 3, 4, 5])

    def test_extend(self):
        """Test `TimeSeriesList.extend`."""
        a = self.create()
        b = a.copy()
        new = self.ENTRY_CLASS([1, 2, 3, 4, 5])
        a.append(new)
        b.extend([new])
        for i in range(max(map(len, (a, b)))):
            utils.assert_quantity_sub_equal(a[i], b[i])

    def test_coalesce(self):
        """Test `TimeSeriesList.coalesce`."""
        a = self.TEST_CLASS()
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=0, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=11, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=5, dx=1))
        a.coalesce()
        assert len(a) == 2
        assert a[0].span == (0, 10)
        utils.assert_array_equal(a[0].value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5])

    def test_join(self):
        """Test `TimeSeriesList.join`."""
        a = self.TEST_CLASS()
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=0, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=5, dx=1))
        a.append(self.ENTRY_CLASS([1, 2, 3, 4, 5], x0=11, dx=1))

        # disjoint list should throw error
        with pytest.raises(
            ValueError,
            match="Cannot append",
        ):
            a.join()

        # but we can pad to get rid of the errors
        t = a.join(gap="pad")
        assert isinstance(t, a.EntryClass)
        assert t.span == (0, 16)
        utils.assert_array_equal(
            t.value, [1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5])

    def test_join_empty(self):
        """Test `TimeSeriesList.join` with an empty list."""
        # check that joining empty list produces something sensible
        t = self.TEST_CLASS().join()
        assert isinstance(t, self.TEST_CLASS.EntryClass)
        assert t.size == 0

    def test_slice(self, instance):
        """Test `TimeSeriesList` slicing."""
        s = instance[:2]
        assert type(s) is type(instance)

    def test_copy(self, instance):
        """Test `TimeSeriesList.copy`."""
        a = instance.copy()
        assert type(a) is type(instance)
        for x, y in zip(instance, a, strict=True):
            utils.assert_quantity_sub_equal(x, y)
            assert not shares_memory(x.value, y.value)
