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
import pytest
import tempfile

from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError

import pytest

import numpy
from numpy import testing as nptest
try:
    from numpy import shares_memory
except ImportError:  # old numpy
    from numpy import may_share_memory as shares_memory


from scipy import signal

from matplotlib import use, rc_context
use('agg')  # nopep8

from astropy import units
from astropy.io.registry import (get_reader, register_reader)

from gwpy.detector import Channel
from gwpy.time import (Time, LIGOTimeGPS)
from gwpy.timeseries import (TimeSeriesBase, TimeSeriesBaseDict,
                             TimeSeriesBaseList,
                             TimeSeries, TimeSeriesDict, TimeSeriesList,
                             StateVector, StateVectorDict, StateVectorList,
                             StateTimeSeries, StateTimeSeriesDict, Bits)
from gwpy.segments import (Segment, SegmentList,
                           DataQualityFlag, DataQualityDict)
from gwpy.frequencyseries import (FrequencySeries, SpectralVariance)
from gwpy.types import Array2D
from gwpy.spectrogram import Spectrogram
from gwpy.plotter import (TimeSeriesPlot, SegmentPlot)

import mocks
import utils
from mocks import mock
from test_array import TestSeries

SEED = 1
numpy.random.seed(SEED)
GPS_EPOCH = Time(0, format='gps', scale='utc')
ONE_HZ = units.Quantity(1, 'Hz')
ONE_SECOND = units.Quantity(1, 'second')

TEST_GWF_FILE = os.path.join(os.path.split(__file__)[0], 'data',
                             'HLV-GW100916-968654552-1.gwf')
TEST_HDF_FILE = '%s.hdf' % TEST_GWF_FILE[:-4]
TEST_SEGMENT = Segment(968654552, 968654553)


FIND_CHANNEL = 'L1:DCS-CALIB_STRAIN_C01'
FIND_FRAMETYPE = 'L1_HOFT_C01'

LOSC_IFO = 'L1'
LOSC_GW150914 = 1126259462
LOSC_GW150914_SEGMENT = Segment(LOSC_GW150914-2, LOSC_GW150914+2)
LOSC_GW150914_DQ_BITS = [
    'data present',
    'passes cbc CAT1 test',
    'passes cbc CAT2 test',
    'passes cbc CAT3 test',
    'passes burst CAT1 test',
    'passes burst CAT2 test',
    'passes burst CAT3 test',
]

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -----------------------------------------------------------------------------
#
# gwpy.timeseries.core
#
# -----------------------------------------------------------------------------

# -- TimeSeriesBase -----------------------------------------------------------

class TestTimeSeriesBase(TestSeries):
    TEST_CLASS = TimeSeriesBase

    def test_new(self):
        """Test `gwpy.timeseries.TimeSeriesBase` constructor
        """
        array = super(TestTimeSeriesBase, self).test_new()

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
            assert isinstance(plot, TimeSeriesPlot)
            line = plot.gca().lines[0]
            utils.assert_array_equal(line.get_xdata(), array.xindex.value)
            utils.assert_array_equal(line.get_ydata(), array.value)
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)
            return plot  # allow subclasses to extend tests

    @utils.skip_missing_dependency('nds2')
    def test_from_nds2_buffer(self):
        nds_buffer = mocks.nds2_buffer(
            'X1:TEST', self.data, 1000000000, self.data.shape[0], 'm')
        a = self.TEST_CLASS.from_nds2_buffer(nds_buffer)
        assert isinstance(a, self.TEST_CLASS)
        utils.assert_array_equal(a.value, self.data)
        assert a.unit == units.m
        assert a.t0 == 1000000000 * units.s
        assert a.dt == units.s / self.data.shape[0]
        assert a.name == 'X1:TEST'
        assert a.channel == Channel('X1:TEST', sample_rate=self.data.shape[0],
                                    unit='m', type='raw', dtype='float32')
        b = self.TEST_CLASS.from_nds2_buffer(nds_buffer, sample_rate=128)
        assert b.dt == 1/128. * units.s

    @utils.skip_missing_dependency('lal')
    def test_to_from_lal(self, array):
        import lal

        # check that to + from returns the same array
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts)
        utils.assert_quantity_sub_equal(array, a2, exclude=['name', 'channel'])
        assert a2.name is ''

        # test copy=False
        a2 = type(array).from_lal(lalts, copy=False)
        assert shares_memory(a2.value, lalts.data.data)

        # test units
        array.override_unit('undef')
        with pytest.warns(UserWarning):
            lalts = array.to_lal()
        assert lalts.sampleUnits == lal.DimensionlessUnit
        a2 = self.TEST_CLASS.from_lal(lalts)
        assert a2.unit is units.dimensionless_unscaled

    @utils.skip_missing_dependency('pycbc')
    def test_to_from_pycbc(self, array):
        from pycbc.types import TimeSeries as PyCBCTimeSeries

        # test default conversion
        pycbcts = array.to_pycbc()
        assert isinstance(pycbcts, PyCBCTimeSeries)
        nptest.assert_array_equal(array.value, pycbcts.data)
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
            new[key] = type(a)([1, 2, 3, 4, 5], x0=a.xspan[1], dx=a.dx,
                               dtype=a.dtype)
        # check error
        with pytest.raises(ValueError):
            instance.append(new)
        # check padding works (don't validate too much, that is tested
        # elsewhere)
        b = instance.copy()
        b.append(new, pad=0)

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
        a = instance.crop(10, 20)
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

    def test_plot(self, instance):
        with rc_context(rc={'text.usetex': False}):
            plot = instance.plot()
            assert isinstance(plot, TimeSeriesPlot)
            for line, key in zip(plot.gca().lines, instance):
                utils.assert_array_equal(line.get_xdata(),
                                         instance[key].xindex.value)
                utils.assert_array_equal(line.get_ydata(), instance[key].value)
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)
            return plot  # allow subclasses to extend tests


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
        assert a == b

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


# -----------------------------------------------------------------------------
#
# gwpy.timeseries.timeseries
#
# -----------------------------------------------------------------------------

# -- TimeSeries ---------------------------------------------------------------


class TestTimeSeries(TestTimeSeriesBase):
    TEST_CLASS = TimeSeries

    # -- fixtures -------------------------------

    @pytest.fixture(scope='class')
    def losc(self):
        try:
            return self.TEST_CLASS.fetch_open_data(
                LOSC_IFO, *LOSC_GW150914_SEGMENT, format='txt.gz')
        except URLError as e:
            pytest.skip(str(e))

    @pytest.fixture(scope='class')
    def losc_16384(self):
        try:
            return self.TEST_CLASS.fetch_open_data(
                LOSC_IFO, *LOSC_GW150914_SEGMENT, sample_rate=16384,
                format='txt.gz')
        except URLError as e:
            pytest.skip(str(e))

    # -- test class functionality ---------------

    def test_ligotimegps(self):
        # test that LIGOTimeGPS works
        array = self.create(t0=LIGOTimeGPS(0))
        assert array.t0.value == 0
        array.t0 = LIGOTimeGPS(10)
        assert array.t0.value == 10
        array.x0 = LIGOTimeGPS(1000000000)
        assert array.t0.value == 1000000000

        # check epoch access
        array.epoch = LIGOTimeGPS(10)
        assert array.t0.value == 10

    def test_epoch(self):
        array = self.create()
        assert array.epoch.gps == array.x0.value

    # -- test I/O -------------------------------

    @pytest.mark.parametrize('format', ['txt', 'csv'])
    def test_read_write_ascii(self, array, format):
        utils.test_read_write(
            array, format,
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={'exclude': ['name', 'channel', 'unit']})

    @pytest.mark.parametrize('api', [
        None,
        pytest.param(
            'lalframe',
            marks=utils.skip_missing_dependency('lalframe')),
        pytest.param(
            'framecpp',
            marks=utils.skip_missing_dependency('LDAStools.frameCPP')),
    ])
    def test_read_write_gwf(self, api):
        array = self.create(name='TEST')

        # map API to format name
        if api is None:
            fmt = 'gwf'
        else:
            fmt = 'gwf.%s' % api

        # test basic write/read
        try:
            utils.test_read_write(
                array, fmt, extension='gwf', read_args=[array.name],
                assert_equal=utils.assert_quantity_sub_equal,
                assert_kw={'exclude': ['channel']})
        except ImportError as e:
            pytest.skip(str(e))

        # test read keyword arguments
        suffix = '-%d-%d.gwf' % (array.t0.value, array.duration.value)
        with tempfile.NamedTemporaryFile(prefix='GWpy-', suffix=suffix) as f:
            array.write(f.name)

            def read_(**kwargs):
                return type(array).read(f, array.name, format='gwf', **kwargs)

            # test start, end
            start, end = array.span.contract(10)
            t = read_(start=start, end=end)
            utils.assert_quantity_sub_equal(t, array.crop(start, end),
                                            exclude=['channel'])
            assert t.span == (start, end)
            t = read_(start=start)
            utils.assert_quantity_sub_equal(t, array.crop(start=start),
                                            exclude=['channel'])
            t = read_(end=end)
            utils.assert_quantity_sub_equal(t, array.crop(end=end),
                                            exclude=['channel'])

            # test dtype
            t = read_(dtype='float32')
            assert t.dtype is numpy.dtype('float32')
            t = read_(dtype={f.name: 'float64'})
            assert t.dtype is numpy.dtype('float64')

            # check errors
            with pytest.raises((ValueError, RuntimeError)):
                read_(start=array.span[1])
            with pytest.raises((ValueError, RuntimeError)):
                read_(end=array.span[0]-1)

            # check old format prints a deprecation warning
            if api:
                with pytest.warns(DeprecationWarning):
                    type(array).read(f, array.name, format=api)

            # check reading from cache
            try:
                from glue.lal import Cache
            except ImportError:
                pass
            else:
                a2 = self.create(name='TEST', t0=array.span[1],
                                 dt=array.dx)
                suffix = '-%d-%d.gwf' % (a2.t0.value, a2.duration.value)
                with tempfile.NamedTemporaryFile(prefix='GWpy-',
                                                 suffix=suffix) as f2:
                    a2.write(f2.name)
                    cache = Cache.from_urls([f.name, f2.name], coltype=int)
                    comb = type(array).read(cache, 'TEST', format=fmt, nproc=2)
                    utils.assert_quantity_sub_equal(
                        comb, array.append(a2, inplace=False),
                        exclude=['channel'])

    @utils.skip_missing_dependency('h5py')
    @pytest.mark.parametrize('ext', ('hdf5', 'h5'))
    def test_read_write_hdf5(self, ext):
        array = self.create()
        array.channel = 'X1:TEST-CHANNEL'

        with tempfile.NamedTemporaryFile(suffix='.%s' % ext) as f:
            # check array with no name fails
            with pytest.raises(ValueError) as exc:
                array.write(f.name, overwrite=True)
            assert str(exc.value).startswith('Cannot determine HDF5 path')
            array.name = 'TEST'

            # write array (with auto-identify)
            array.write(f.name, overwrite=True)

            # check reading gives the same data (with/without auto-identify)
            ts = type(array).read(f.name, format='hdf5')
            utils.assert_quantity_sub_equal(array, ts)
            ts = type(array).read(f.name)
            utils.assert_quantity_sub_equal(array, ts)

            # check that we can't then write the same data again
            with pytest.raises(IOError):
                array.write(f.name)
            with pytest.raises(RuntimeError):
                array.write(f.name, append=True)

            # check reading with start/end works
            start, end = array.span.contract(25)
            t = type(array).read(f, start=start, end=end)
            utils.assert_quantity_sub_equal(t, array.crop(start, end))

    @utils.skip_minimum_version('scipy', '0.13.0')
    def test_read_write_wav(self):
        array = self.create(dtype='float32')
        utils.test_read_write(
            array, 'wav', read_kw={'mmap': True}, write_kw={'scale': 1},
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={'exclude': ['unit', 'name', 'channel', 'x0']})

    # -- test remote data access ----------------

    @pytest.mark.parametrize('format', [
        None,
        'txt.gz',
        pytest.param('hdf5', marks=utils.skip_missing_dependency('h5py')),
    ])
    def test_fetch_open_data(self, losc, format):
        try:
            ts = self.TEST_CLASS.fetch_open_data(
                LOSC_IFO, *LOSC_GW150914_SEGMENT, format=format)
        except URLError as e:
            pytest.skip(str(e))
        utils.assert_quantity_sub_equal(ts, losc, exclude=['name', 'unit'])

        # try again with 16384 Hz data
        ts = self.TEST_CLASS.fetch_open_data(
            LOSC_IFO, *LOSC_GW150914_SEGMENT, format=format, sample_rate=16384)
        assert ts.sample_rate == 16384 * units.Hz

        # make sure errors happen
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS.fetch_open_data(LOSC_IFO, 0, 1, format=format)
        assert str(exc.value) == (
            "Cannot find a LOSC dataset for %s covering [0, 1)" % LOSC_IFO)

    @utils.skip_missing_dependency('nds2')
    def test_fetch(self):
        ts = self.create(name='X1:TEST', t0=1000000000, unit='m')
        nds_buffer = mocks.nds2_buffer_from_timeseries(ts)
        nds_connection = mocks.nds2_connection(buffers=[nds_buffer])
        with mock.patch('nds2.connection') as mock_connection, \
                mock.patch('nds2.buffer', nds_buffer):
            mock_connection.return_value = nds_connection
            # use verbose=True to hit more lines
            ts2 = self.TEST_CLASS.fetch('X1:TEST', *ts.span, verbose=True)
            # check open connection works
            ts2 = self.TEST_CLASS.fetch('X1:TEST', *ts.span, verbose=True,
                                        connection=nds_connection)
        utils.assert_quantity_sub_equal(ts, ts2, exclude=['channel'])

    @utils.skip_missing_dependency('glue.datafind')
    @utils.skip_missing_dependency('LDAStools.frameCPP')
    @pytest.mark.skipif('LIGO_DATAFIND_SERVER' not in os.environ,
                        reason='No LIGO datafind server configured '
                               'on this host')
    def test_find(self, losc_16384):
        ts = self.TEST_CLASS.find(FIND_CHANNEL, *LOSC_GW150914_SEGMENT,
                                  frametype=FIND_FRAMETYPE)
        utils.assert_quantity_sub_equal(ts, losc_16384,
                                        exclude=['name', 'channel', 'unit'])

        # test observatory
        ts2 = self.TEST_CLASS.find(FIND_CHANNEL, *LOSC_GW150914_SEGMENT,
                                   frametype=FIND_FRAMETYPE,
                                   observatory=FIND_CHANNEL[0])
        utils.assert_quantity_sub_equal(ts, ts2)
        with pytest.raises(RuntimeError):
            self.TEST_CLASS.find(FIND_CHANNEL, *LOSC_GW150914_SEGMENT,
                                 frametype=FIND_FRAMETYPE, observatory='X')

    @utils.skip_missing_dependency('glue.datafind')
    @utils.skip_missing_dependency('LDAStools.frameCPP')
    @pytest.mark.skipif('LIGO_DATAFIND_SERVER' not in os.environ,
                        reason='No LIGO datafind server configured '
                               'on this host')
    def test_find_best_frametype(self):
        from gwpy.io import datafind
        # test a few (channel, frametype) pairs
        for channel, target in [
                ('H1:GDS-CALIB_STRAIN',
                    ['H1_HOFT_C00', 'H1_ER_C00_L1']),
                ('L1:IMC-ODC_CHANNEL_OUT_DQ',
                    ['L1_R']),
                ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend',
                    ['H1_T']),
                ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend',
                    ['H1_M'])]:
            ft = datafind.find_best_frametype(
                channel, 1143504017, 1143504017+100)
            assert ft in target

        # test that this works end-to-end as part of a TimeSeries.find
        ts = self.TEST_CLASS.find(FIND_CHANNEL, *LOSC_GW150914_SEGMENT)

    def test_get(self, losc_16384):
        # get using datafind (maybe)
        try:
            ts = self.TEST_CLASS.get(FIND_CHANNEL, *LOSC_GW150914_SEGMENT,
                                     frametype_match='C01\Z')
        except (ImportError, RuntimeError) as e:
            pytest.skip(str(e))
        except IOError as exc:
            if 'reading from stdin' in str(exc):
                pytest.skip(str(exc))
            raise
        utils.assert_quantity_sub_equal(ts, losc_16384,
                                        exclude=['name', 'channel', 'unit'])

        # get using NDS2 (if datafind could have been used to start with)
        try:
            dfs = os.environ.pop('LIGO_DATAFIND_SERVER')
        except KeyError:
            dfs = None
        else:
            ts2 = self.TEST_CLASS.get(FIND_CHANNEL, *LOSC_GW150914_SEGMENT)
            utils.assert_quantity_sub_equal(ts, ts2,
                                            exclude=['channel', 'unit'])
        finally:
            if dfs is not None:
                os.environ['LIGO_DATAFIND_SERVER'] = dfs

    # -- signal processing methods --------------

    def test_fft(self, losc):
        fs = losc.fft()
        assert isinstance(fs, FrequencySeries)
        assert fs.size == losc.size // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == 1 / losc.duration
        assert fs.channel is losc.channel
        nptest.assert_almost_equal(
            fs.value.max(), 9.793003238789471e-20+3.5377863373683966e-21j)

        # test with nfft arg
        fs = losc.fft(nfft=256)
        assert fs.size == 129
        assert fs.dx == losc.sample_rate / 256

    def test_average_fft(self, losc):
        # test all defaults
        fs = losc.average_fft()
        utils.assert_quantity_sub_equal(fs, losc.detrend().fft())

        # test fftlength
        fs = losc.average_fft(fftlength=0.5)
        assert fs.size == 0.5 * losc.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hertz

        fs = losc.average_fft(fftlength=0.4, overlap=0.2)

    def test_psd(self, losc):
        # test all defaults
        fs = losc.psd()
        assert isinstance(fs, FrequencySeries)
        assert fs.size == losc.size // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == 1 / losc.duration
        assert fs.channel is losc.channel
        assert fs.unit == losc.unit ** 2 / units.Hz

        # test fftlength
        fs = losc.psd(fftlength=0.5)
        assert fs.size == 0.5 * losc.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hz

        # test overlap
        fs = losc.psd(fftlength=0.4, overlap=0.2)

        # test default overlap
        fs2 = losc.psd(fftlength=.4)
        utils.assert_quantity_sub_equal(fs, fs2)

        # test methods
        losc.psd(fftlength=0.4, overlap=0.2, method='welch')
        losc.psd(fftlength=0.4, method='bartlett')
        try:
            losc.psd(fftlength=0.4, overlap=0.2, method='lal-welch')
            losc.psd(fftlength=0.4, method='lal-bartlett')
            losc.psd(fftlength=0.4, overlap=0.2, method='median-mean')
            losc.psd(fftlength=0.4, overlap=0.2, method='median')
        except ImportError as e:
            pass
        else:
            # test LAL method with window specification
            losc.psd(fftlength=0.4, overlap=0.2, method='median-mean',
                     window='hann')

            # test LAL method with non-canonical window specification
            losc.psd(fftlength=0.4, overlap=0.2, method='median-mean',
                     window='hanning')

            # test check for at least two averages (defaults to single FFT)
            with pytest.raises(ValueError) as e:
                losc.psd(method='median-mean')

    def test_asd(self, losc):
        fs = losc.asd()
        utils.assert_quantity_sub_equal(fs, losc.psd() ** (1/2.))

    @utils.skip_minimum_version('scipy', '0.16')
    def test_csd(self, losc):
        # test all defaults
        fs = losc.csd(losc)
        utils.assert_quantity_sub_equal(fs, losc.psd(), exclude=['name'])

        # test fftlength
        fs = losc.csd(losc, fftlength=0.5)
        assert fs.size == 0.5 * losc.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hertz

        # test overlap
        losc.csd(losc, fftlength=0.4, overlap=0.2)

    def test_spectrogram(self, losc):
        # test defaults
        sg = losc.spectrogram(1)  # defaults to 50% overlap for 'hann' window
        assert isinstance(sg, Spectrogram)
        assert sg.shape == (4, losc.sample_rate.value // 2 + 1)
        assert sg.f0 == 0 * units.Hz
        assert sg.df == 1 * units.Hz
        assert sg.channel is losc.channel
        assert sg.unit == losc.unit ** 2 / units.Hz
        assert sg.epoch == losc.epoch
        assert sg.span == losc.span

        # check the same result as PSD
        psd = losc[:int(losc.sample_rate.value)].psd()
        # FIXME: epoch should not be excluded here (probably)
        utils.assert_quantity_sub_equal(sg[0], psd, exclude=['epoch'])

        # test fftlength
        sg = losc.spectrogram(1, fftlength=0.5)
        assert sg.shape == (4, 0.5 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second
        # test overlap
        sg = losc.spectrogram(0.5, fftlength=0.25, overlap=0.125)
        assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        assert sg.dt == 0.5 * units.second
        # test multiprocessing
        sg2 = losc.spectrogram(0.5, fftlength=0.25, overlap=0.125, nproc=2)
        utils.assert_quantity_sub_equal(sg, sg2)
        # test methods
        sg = losc.spectrogram(0.5, fftlength=0.25, method='welch')
        assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        assert sg.dt == 0.5 * units.second
        sg = losc.spectrogram(0.5, fftlength=0.25, method='bartlett')
        assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        assert sg.dt == 0.5 * units.second
        try:
            sg = losc.spectrogram(0.5, fftlength=0.25, method='lal-welch')
        except ImportError:
            pass
        else:
            assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
            assert sg.df == 4 * units.Hertz
            assert sg.dt == 0.5 * units.second
            sg = losc.spectrogram(0.5, fftlength=0.25, method='median-mean')
            assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
            assert sg.df == 4 * units.Hertz
            assert sg.dt == 0.5 * units.second
            sg = losc.spectrogram(0.5, fftlength=0.25, method='median')
            assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
            assert sg.df == 4 * units.Hertz
            assert sg.dt == 0.5 * units.second

        # check that `cross` keyword gets deprecated properly
        # TODO: removed before 1.0 release
        with pytest.warns(DeprecationWarning) as wng:
            try:
                out = losc.spectrogram(0.5, fftlength=.25, cross=losc)
            except AttributeError:
                return  # scipy is too old
        assert '`cross` keyword argument has been deprecated' in \
            wng[0].message.args[0]
        utils.assert_quantity_sub_equal(
            out, losc.csd_spectrogram(losc, 0.5, fftlength=.25))

    def test_spectrogram2(self, losc):
        # test defaults
        sg = losc.spectrogram2(1)
        utils.assert_quantity_sub_equal(sg, losc.spectrogram(1))

        # test fftlength
        sg = losc.spectrogram2(0.5)
        assert sg.shape == (8, 0.5 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 0.5 * units.second
        # test overlap
        sg = losc.spectrogram2(fftlength=0.25, overlap=0.24)
        assert sg.shape == (399, 0.25 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        # note: bizarre stride length because 4096/100 gets rounded
        assert sg.dt == 0.010009765625 * units.second

    def test_spectral_variance(self, losc):
        variance = losc.spectral_variance(.5)
        assert isinstance(variance, SpectralVariance)
        print(variance)
        assert variance.x0 == 0 * units.Hz
        assert variance.dx == 2 * units.Hz
        assert variance.max() == 8

    def test_rayleigh_spectrum(self, losc):
        # assert single FFT creates Rayleigh of 0
        ray = losc.rayleigh_spectrum()
        assert isinstance(ray, FrequencySeries)
        assert ray.unit is units.Unit('')
        assert ray.name == 'Rayleigh spectrum of %s' % losc.name
        assert ray.epoch == losc.epoch
        assert ray.channel is losc.channel
        assert ray.f0 == 0 * units.Hz
        assert ray.df == 1 / losc.duration
        assert ray.sum().value == 0

        # actually test properly
        ray = losc.rayleigh_spectrum(.5)  # no overlap
        assert ray.df == 2 * units.Hz
        nptest.assert_almost_equal(ray.max().value, 2.1239253590490157)
        assert ray.frequencies[ray.argmax()] == 1322 * units.Hz

        ray = losc.rayleigh_spectrum(.5, .25)  # 50 % overlap
        nptest.assert_almost_equal(ray.max().value, 1.8814775174483833)
        assert ray.frequencies[ray.argmax()] == 136 * units.Hz

    @utils.skip_minimum_version('scipy', '0.16')
    def test_csd_spectrogram(self, losc):
        # test defaults
        sg = losc.csd_spectrogram(losc, 1)
        assert isinstance(sg, Spectrogram)
        assert sg.shape == (4, losc.sample_rate.value // 2 + 1)
        assert sg.f0 == 0 * units.Hz
        assert sg.df == 1 * units.Hz
        assert sg.channel is losc.channel
        assert sg.unit == losc.unit ** 2 / units.Hertz
        assert sg.epoch == losc.epoch
        assert sg.span == losc.span

        # check the same result as CSD
        losc1 = losc[:int(losc.sample_rate.value)]
        csd = losc1.csd(losc1)
        utils.assert_quantity_sub_equal(sg[0], csd, exclude=['name', 'epoch'])

        # test fftlength
        sg = losc.csd_spectrogram(losc, 1, fftlength=0.5)
        assert sg.shape == (4, 0.5 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second

        # test overlap
        sg = losc.csd_spectrogram(losc, 0.5, fftlength=0.25, overlap=0.125)
        assert sg.shape == (8, 0.25 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        assert sg.dt == 0.5 * units.second

        # test multiprocessing
        sg2 = losc.csd_spectrogram(losc, 0.5, fftlength=0.25,
                                   overlap=0.125, nproc=2)
        utils.assert_quantity_sub_equal(sg, sg2)

    def test_resample(self, losc):
        """Test :meth:`gwpy.timeseries.TimeSeries.resample`
        """
        # test IIR decimation
        l2 = losc.resample(1024, ftype='iir')
        # FIXME: this test needs to be more robust
        assert l2.sample_rate == 1024 * units.Hz

    def test_rms(self, losc):
        rms = losc.rms(1.)
        assert rms.sample_rate == 1 * units.Hz

    def test_whiten(self):
        # create noise with a glitch in it at 1000 Hz
        noise = self.TEST_CLASS(
            numpy.random.normal(loc=1, size=16384 * 10), sample_rate=16384,
            epoch=-5).zpk([], [0], 1)
        glitchtime = 0.5
        glitch = signal.gausspulse(noise.times.value + glitchtime,
                                   bw=100) * 1e-4
        data = noise + glitch

        # whiten and test that the max amplitude is recovered at the glitch
        tmax = data.times[data.argmax()]
        assert not numpy.isclose(tmax.value, -glitchtime)

        whitened = data.whiten(2, 1)

        assert noise.size == whitened.size
        nptest.assert_almost_equal(whitened.mean().value, 0.0, decimal=4)

        tmax = whitened.times[whitened.argmax()]
        nptest.assert_almost_equal(tmax.value, -glitchtime)

    def test_detrend(self, losc):
        assert not numpy.isclose(losc.value.mean(), 0.0, atol=1e-21)
        detrended = losc.detrend()
        assert numpy.isclose(detrended.value.mean(), 0.0)

    def test_notch(self, losc):
        # test notch runs end-to-end
        notched = losc.notch(60)

        # test breaks when you try and 'fir' notch
        with pytest.raises(NotImplementedError):
            losc.notch(10, type='fir')

    def test_q_transform(self, losc):
        # test simple q-transform
        qspecgram = losc.q_transform(method='welch', fftlength=2)
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (4000, 2403)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 146.61970478954652)

        # test whitening args
        asd = losc.asd(2, 1)
        qsg2 = losc.q_transform(method='welch', whiten=asd)
        utils.assert_quantity_sub_equal(qspecgram, qsg2)

        asd = losc.asd(.5, .25)
        qsg2 = losc.q_transform(method='welch', whiten=asd)
        qsg3 = losc.q_transform(method='welch', fftlength=.5, overlap=.25)
        utils.assert_quantity_sub_equal(qsg2, qsg3)

        # make sure frequency too high presents warning
        with pytest.warns(UserWarning):
            qspecgram = losc.q_transform(method='welch', frange=(0, 10000))
            nptest.assert_almost_equal(qspecgram.yspan[1], 1291.5316316157107)

        # test other normalisations work (or don't)
        q2 = losc.q_transform(method='welch', norm='median')
        utils.assert_quantity_sub_equal(qspecgram, q2)
        losc.q_transform(method='welch', norm='mean')
        losc.q_transform(method='welch', norm=False)
        with pytest.raises(ValueError):
            losc.q_transform(method='welch', norm='blah')

    def test_boolean_statetimeseries(self, array):
        comp = array >= 2 * array.unit
        assert isinstance(comp, StateTimeSeries)
        assert comp.unit is units.Unit('')
        assert comp.name == '%s >= 2.0' % (array.name)


# -- TimeSeriesDict -----------------------------------------------------------

class TestTimeSeriesDict(TestTimeSeriesBaseDict):
    channels = ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN']
    TEST_CLASS = TimeSeriesDict
    ENTRY_CLASS = TimeSeries

    @utils.skip_missing_dependency('LDAStools.frameCPP')
    def test_read_write_gwf(self, instance):
        with tempfile.NamedTemporaryFile(suffix='.gwf') as f:
            instance.write(f.name)
            new = self.TEST_CLASS.read(f.name, instance.keys())
            for key in new:
                utils.assert_quantity_sub_equal(new[key], instance[key],
                                                exclude=['channel'])

    @utils.skip_missing_dependency('h5py')
    def test_read_write_hdf5(self, instance):
        with tempfile.NamedTemporaryFile(suffix='.hdf5') as f:
            instance.write(f.name, overwrite=True)
            new = self.TEST_CLASS.read(f.name, instance.keys())
            for key in new:
                utils.assert_quantity_sub_equal(new[key], instance[key])
            # check auto-detection of names
            new = self.TEST_CLASS.read(f.name)
            for key in new:
                utils.assert_quantity_sub_equal(new[key], instance[key])


# -- TimeSeriesList -----------------------------------------------------------

class TestTimeSeriesList(TestTimeSeriesBaseList):
    TEST_CLASS = TimeSeriesList
    ENTRY_CLASS = TimeSeries


# -----------------------------------------------------------------------------
#
# gwpy.timeseries.statevector
#
# -----------------------------------------------------------------------------

# -- StateTimeSeries ----------------------------------------------------------

class TestStateTimeSeries(TestTimeSeriesBase):
    TEST_CLASS = StateTimeSeries

    @classmethod
    def setup_class(cls):
        cls.data = numpy.asarray([0, 1, 1, 1, 0, 0, 0, 1, 0, 0,
                                  1, 1, 1, 0, 1, 0, 1, 1, 1, 1],
                                 dtype=bool)

    def test_new(self):
        sts = self.create()
        assert isinstance(sts, self.TEST_CLASS)
        assert sts.dtype is numpy.dtype('bool')

    def test_getitem(self, array):
        assert isinstance(array[0], numpy.bool_)

    def test_unit(self, array):
        assert array.unit is units.dimensionless_unscaled

        # check that we can't delete the unit
        with pytest.raises(AttributeError):
            del array.unit

        # check that we can't set the unit
        with pytest.raises(TypeError):
            self.create(unit='test')

    def test_math(self, array):
        # test that math operations give back booleans
        a2 = array ** 2
        assert a2.dtype is numpy.dtype(bool)
        utils.assert_array_equal(array.value, a2.value)

    def test_override_unit(self):
        return NotImplemented

    def test_is_compatible(self):
        return NotImplemented

    def test_to_from_pycbc(self):
        return NotImplemented

    def test_to_from_lal(self):
        return NotImplemented

    def test_from_nds2_buffer(self):
        return NotImplemented


# -- StateTimeSeriesDict ------------------------------------------------------

class TestStateTimeSeriesDict(TestTimeSeriesBaseDict):
    TEST_CLASS = StateTimeSeriesDict
    ENTRY_CLASS = StateTimeSeries
    DTYPE = 'bool'

    def test_resample(self):
        return NotImplemented


# -- StateVector---------------------------------------------------------------

class TestStateVector(TestTimeSeriesBase):
    TEST_CLASS = StateVector
    DTYPE = 'uint32'

    @classmethod
    def setup_class(cls):
        numpy.random.seed(0)
        cls.data = numpy.random.randint(
            2**4+1, size=100).astype(cls.DTYPE, copy=False)

    def test_bits(self, array):
        assert isinstance(array.bits, Bits)
        assert array.bits == ['Bit %d' % i for i in range(32)]

        bits = ['Bit %d' % i for i in range(4)]

        sv = self.create(bits=bits)
        assert isinstance(sv.bits, Bits)
        assert sv.bits.channel is sv.channel
        assert sv.bits.epoch == sv.epoch
        assert sv.bits == bits

        del sv.bits
        del sv.bits
        assert isinstance(sv.bits, Bits)
        assert sv.bits == ['Bit %d' % i for i in range(32)]

        sv = self.create(dtype='uint16')
        assert sv.bits == ['Bit %d' % i for i in range(16)]

    def test_boolean(self, array):
        b = array.boolean
        assert isinstance(b, Array2D)
        assert b.shape == (array.size, len(array.bits))
        # array[0] == 12, check boolean equivalent
        utils.assert_array_equal(b[0], [int(12) >> j & 1 for j in range(32)])

    def test_get_bit_series(self, array):
        # test default
        bs = array.get_bit_series()
        assert isinstance(bs, StateTimeSeriesDict)
        assert list(bs.keys()) == array.bits
        # check that correct number of samples match (simple test)
        assert bs['Bit 2'].sum() == 43

        # check that bits in gives bits out
        bs = array.get_bit_series(['Bit 0', 'Bit 3'])
        assert list(bs.keys()) == ['Bit 0', 'Bit 3']
        assert [v.sum() for v in bs.values()] == [50, 41]

        # check that invalid bits throws exception
        with pytest.raises(ValueError) as exc:
            array.get_bit_series(['blah'])
        assert str(exc.value) == "Bit 'blah' not found in StateVector"

    @utils.skip_missing_dependency('lal')
    def test_plot(self, array):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot()
            assert isinstance(plot, TimeSeriesPlot)
            # make sure there were no lines drawn
            assert plot.gca().lines == []
            # assert one collection for each of known and active segmentlists
            assert len(plot.gca().collections) == len(array.bits) * 2
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)
            plot.close()

            # test timeseries plotting as normal
            plot = array.plot(format='timeseries')
            assert isinstance(plot, TimeSeriesPlot)
            line = plot.gca().lines[0]
            utils.assert_array_equal(line.get_xdata(), array.xindex.value)
            utils.assert_array_equal(line.get_ydata(), array.value)
            plot.close()

    def test_resample(self, array):
        # check downsampling by factor of 2
        a2 = array.resample(array.sample_rate / 2.)
        assert a2.sample_rate == array.sample_rate / 2.
        assert a2.bits is array.bits
        utils.assert_array_equal(a2.value[:10],
                                 [12, 0, 3, 0, 4, 0, 6, 5, 8, 0])

        # check upsampling raises NotImplementedError
        with pytest.raises(NotImplementedError):
            array.resample(array.sample_rate * 2.)

        # check resampling by non-integer factor raises error
        with pytest.raises(ValueError):
            array.resample(array.sample_rate * .75)
        with pytest.raises(ValueError):
            array.resample(array.sample_rate * 1.5)


# -- StateVectorDict ----------------------------------------------------------

class TestStateVectorDict(TestTimeSeriesBaseDict):
    TEST_CLASS = StateVectorDict
    ENTRY_CLASS = StateVector
    DTYPE = 'uint32'


# -- StateVectorList ----------------------------------------------------------

class TestStateVectorList(TestTimeSeriesBaseList):
    TEST_CLASS = StateVectorList
    ENTRY_CLASS = StateVector
    DTYPE = 'uint32'
