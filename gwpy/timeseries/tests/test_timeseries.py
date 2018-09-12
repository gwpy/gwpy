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

import importlib
import os.path
from itertools import (chain, product)
from ssl import SSLError

import six
from six.moves.urllib.error import URLError

import pytest

import numpy
from numpy import testing as nptest

from scipy import signal

from astropy import units

from ...frequencyseries import (FrequencySeries, SpectralVariance)
from ...segments import Segment
from ...signal import filter_design
from ...spectrogram import Spectrogram
from ...tests import (mocks, utils)
from ...tests.mocks import mock
from ...time import LIGOTimeGPS
from .. import (TimeSeries, TimeSeriesDict, TimeSeriesList, StateTimeSeries)
from .test_core import (TestTimeSeriesBase as _TestTimeSeriesBase,
                        TestTimeSeriesBaseDict as _TestTimeSeriesBaseDict,
                        TestTimeSeriesBaseList as _TestTimeSeriesBaseList)

FIND_CHANNEL = 'L1:DCS-CALIB_STRAIN_C02'
FIND_FRAMETYPE = 'L1_HOFT_C02'

LOSC_IFO = 'L1'
LOSC_GW150914 = 1126259462
LOSC_GW150914_SEGMENT = Segment(LOSC_GW150914-2, LOSC_GW150914+2)
LOSC_GW150914_DQ_NAME = {
    'hdf5': 'Data quality',
    'gwf': 'L1:LOSC-DQMASK',
}
LOSC_GW150914_DQ_BITS = {
    'hdf5': [
        'data present',
        'passes cbc CAT1 test',
        'passes cbc CAT2 test',
        'passes cbc CAT3 test',
        'passes burst CAT1 test',
        'passes burst CAT2 test',
        'passes burst CAT3 test',
    ],
    'gwf': [
        'DATA',
        'CBC_CAT1',
        'CBC_CAT2',
        'CBC_CAT3',
        'BURST_CAT1',
        'BURST_CAT2',
        'BURST_CAT3',
    ],
}

LOSC_FETCH_ERROR = (URLError, SSLError)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class TestTimeSeries(_TestTimeSeriesBase):
    TEST_CLASS = TimeSeries

    # -- fixtures -------------------------------

    @pytest.fixture(scope='class')
    def losc(self):
        try:
            return self.TEST_CLASS.fetch_open_data(
                LOSC_IFO, *LOSC_GW150914_SEGMENT)
        except LOSC_FETCH_ERROR as e:
            pytest.skip(str(e))

    @pytest.fixture(scope='class')
    def losc_16384(self):
        try:
            return self.TEST_CLASS.fetch_open_data(
                LOSC_IFO, *LOSC_GW150914_SEGMENT, sample_rate=16384)
        except LOSC_FETCH_ERROR as e:
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
        with utils.TemporaryFilename(prefix='GWpy-', suffix=suffix) as tmp:
            array.write(tmp)

            def read_(**kwargs):
                return type(array).read(tmp, array.name, format=fmt,
                                        **kwargs)

            # test reading unicode (python2)
            if six.PY2:
                type(array).read(six.u(tmp), array.name, format=fmt)

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

            # test dtype - DEPRECATED
            with pytest.warns(DeprecationWarning):
                t = read_(dtype='float32')
            assert t.dtype is numpy.dtype('float32')
            with pytest.warns(DeprecationWarning):
                t = read_(dtype={array.name: 'float64'})
            assert t.dtype is numpy.dtype('float64')

            # check errors
            with pytest.raises((ValueError, RuntimeError)):
                read_(start=array.span[1])
            with pytest.raises((ValueError, RuntimeError)):
                read_(end=array.span[0]-1)

            # check reading from multiple files
            a2 = self.create(name='TEST', t0=array.span[1], dt=array.dx)
            suffix = '-%d-%d.gwf' % (a2.t0.value, a2.duration.value)
            with utils.TemporaryFilename(prefix='GWpy-',
                                         suffix=suffix) as tmp2:
                a2.write(tmp2)
                cache = [tmp, tmp2]
                comb = type(array).read(cache, 'TEST', format=fmt, nproc=2)
                utils.assert_quantity_sub_equal(
                    comb, array.append(a2, inplace=False),
                    exclude=['channel'])

    @pytest.mark.parametrize('ext', ('hdf5', 'h5'))
    @pytest.mark.parametrize('channel', [
        None,
        'test',
        'X1:TEST-CHANNEL',
    ])
    def test_read_write_hdf5(self, ext, channel):
        array = self.create()
        array.channel = channel

        with utils.TemporaryFilename(suffix='.%s' % ext) as tmp:
            # check array with no name fails
            with pytest.raises(ValueError) as exc:
                array.write(tmp, overwrite=True)
            assert str(exc.value).startswith('Cannot determine HDF5 path')
            array.name = 'TEST'

            # write array (with auto-identify)
            array.write(tmp, overwrite=True)

            # check reading gives the same data (with/without auto-identify)
            ts = type(array).read(tmp, format='hdf5')
            utils.assert_quantity_sub_equal(array, ts)
            ts = type(array).read(tmp)
            utils.assert_quantity_sub_equal(array, ts)

            # check that we can't then write the same data again
            with pytest.raises(IOError):
                array.write(tmp)
            with pytest.raises(RuntimeError):
                array.write(tmp, append=True)

            # check reading with start/end works
            start, end = array.span.contract(25)
            t = type(array).read(tmp, start=start, end=end)
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
        'hdf5',
        pytest.param(  # only frameCPP actually reads units properly
            'gwf', marks=utils.skip_missing_dependency('LDAStools.frameCPP')),
    ])
    def test_fetch_open_data(self, losc, format):
        try:
            ts = self.TEST_CLASS.fetch_open_data(
                LOSC_IFO, *LOSC_GW150914_SEGMENT, format=format, verbose=True)
        except LOSC_FETCH_ERROR as e:
            pytest.skip(str(e))
        utils.assert_quantity_sub_equal(ts, losc,
                                        exclude=['name', 'unit', 'channel'])

        # try again with 16384 Hz data
        ts = self.TEST_CLASS.fetch_open_data(
            LOSC_IFO, *LOSC_GW150914_SEGMENT, format=format, sample_rate=16384)
        assert ts.sample_rate == 16384 * units.Hz

        # make sure errors happen
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS.fetch_open_data(LOSC_IFO, 0, 1, format=format)
        assert str(exc.value) == (
            "Cannot find a LOSC dataset for %s covering [0, 1)" % LOSC_IFO)

        # check errors with multiple tags
        try:
            with pytest.raises(ValueError) as exc:
                self.TEST_CLASS.fetch_open_data(
                    LOSC_IFO, 1187008880, 1187008884)
            assert str(exc.value).lower().startswith('multiple losc url tags')
            self.TEST_CLASS.fetch_open_data(LOSC_IFO, 1187008880, 1187008884,
                                            tag='CLN')
        except LOSC_FETCH_ERROR:
            pass

    @utils.skip_missing_dependency('nds2')
    def test_fetch(self):
        ts = self.create(name='L1:TEST', t0=1000000000, unit='m')
        nds_buffer = mocks.nds2_buffer_from_timeseries(ts)
        nds_connection = mocks.nds2_connection(buffers=[nds_buffer])
        with mock.patch('nds2.connection') as mock_connection, \
                mock.patch('nds2.buffer', nds_buffer):
            mock_connection.return_value = nds_connection
            # use verbose=True to hit more lines
            ts2 = self.TEST_CLASS.fetch('L1:TEST', *ts.span, verbose=True)
            # check open connection works
            ts2 = self.TEST_CLASS.fetch('L1:TEST', *ts.span, verbose=True,
                                        connection=nds_connection)
        utils.assert_quantity_sub_equal(ts, ts2, exclude=['channel'])

    @utils.skip_missing_dependency('nds2')
    def test_fetch_empty_iterate_error(self):
        # test that the correct error is raised if nds2.connection.iterate
        # yields no buffers (and no errors)

        # mock connection with no data
        nds_connection = mocks.nds2_connection()

        def find_channels(name, *args, **kwargs):
            return [mocks.nds2_channel(name, 128, '')]

        nds_connection.find_channels = find_channels

        # run fetch and assert error
        with mock.patch('nds2.connection') as mock_connection:
            mock_connection.return_value = nds_connection
            with pytest.raises(RuntimeError) as exc:
                self.TEST_CLASS.fetch('L1:TEST', 0, 1, host='nds.gwpy')
            assert 'no data received' in str(exc)

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
    @pytest.mark.parametrize('channel, expected', [
        ('H1:GDS-CALIB_STRAIN', ['H1_HOFT_C00', 'H1_ER_C00_L1']),
        ('L1:IMC-ODC_CHANNEL_OUT_DQ', ['L1_R']),
        ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend', ['H1_T']),
        ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend', ['H1_M'])
    ])
    def test_find_best_frametype(self, channel, expected):
        from gwpy.io import datafind
        try:
            ft = datafind.find_best_frametype(
                channel, 1143504017, 1143504017+100)
        except ValueError as exc:  # ignore
            if str(exc).lower().startswith('cannot locate'):
                pytest.skip(str(exc))
            raise
        assert ft in expected

    @utils.skip_missing_dependency('glue.datafind')
    @utils.skip_missing_dependency('LDAStools.frameCPP')
    @pytest.mark.skipif('LIGO_DATAFIND_SERVER' not in os.environ,
                        reason='No LIGO datafind server configured '
                               'on this host')
    def test_find_best_frametype_in_find(self, losc_16384):
        ts = self.TEST_CLASS.find(FIND_CHANNEL, *LOSC_GW150914_SEGMENT)
        utils.assert_quantity_sub_equal(ts, losc_16384,
                                        exclude=['name', 'channel', 'unit'])

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

    @pytest.mark.parametrize('method', ('welch', 'bartlett'))
    def test_psd_basic(self, losc, method):
        # check that basic methods always post a warning telling the user
        # to be more specific
        with pytest.warns(UserWarning):
            fs = losc.psd(1, method=method, window=None)

        # and check that the basic parameters are sane
        assert fs.size == losc.sample_rate.value // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == 1 * units.Hz
        assert fs.name == losc.name
        assert fs.channel is losc.channel
        assert fs.unit == losc.unit ** 2 / units.Hz

    def test_psd_default_overlap(self, losc):
        utils.assert_quantity_sub_equal(
            losc.psd(.5, window='hann'),
            losc.psd(.5, .25, window='hann'),
        )

    @utils.skip_missing_dependency('lal')
    def test_psd_lal_median_mean(self, losc):
        # check that warnings and errors get raised in the right place
        # for a median-mean PSD with the wrong data size or parameters

        # single segment should raise error
        with pytest.raises(ValueError):
            losc.psd(abs(losc.span), method='lal_median_mean')

        # odd number of segments should warn
        with pytest.warns(UserWarning):
            losc.psd(1, .5, method='lal_median_mean')

    @pytest.mark.parametrize('library, method', chain(
        product(['scipy'], ['welch', 'bartlett']),
        product(['pycbc.psd'], ['welch', 'bartlett', 'median', 'median_mean']),
        product(['lal'], ['welch', 'bartlett', 'median', 'median_mean']),
    ))
    @pytest.mark.parametrize(
        'window', (None, 'hann', ('kaiser', 24), 'array'),
    )
    def test_psd(self, losc, library, method, window):
        try:
            importlib.import_module(library)
        except ImportError as exc:
            pytest.skip(str(exc))

        fftlength = .5
        overlap = .25

        # remove final .25 seconds to stop median-mean complaining
        # (means an even number of overlapping FFT segments)
        if method == 'median_mean':
            losc = losc.crop(end=losc.span[1]-overlap)

        # get actual method name
        library = library.split('.', 1)[0]
        method = '{}_{}'.format(library, method)

        def _psd(fftlength, overlap=None, **kwargs):
            # create window of the correct length
            if window == 'array':
                nfft = (losc.size if fftlength is None else
                        int(fftlength * losc.sample_rate.value))
                _window = signal.get_window('hamming', nfft)
            else:
                _window = window

            # generate PSD
            return losc.psd(fftlength=fftlength, overlap=overlap,
                            method=method, window=_window)

        try:
            fs = _psd(.5, .25)
        except TypeError as exc:
            # catch pycbc window as array error
            # FIXME: remove after PyCBC 1.10 is released
            if str(exc).startswith('unhashable type'):
                pytest.skip(str(exc))
            raise

        # and check that the basic parameters are sane
        assert fs.size == fftlength * losc.sample_rate.value // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == units.Hz / fftlength
        assert fs.name == losc.name
        assert fs.channel is losc.channel
        assert fs.unit == losc.unit ** 2 / units.Hz

    def test_asd(self, losc):
        fs = losc.asd(1)
        utils.assert_quantity_sub_equal(fs, losc.psd(1) ** (1/2.))

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

    @staticmethod
    def _window_helper(series, fftlength, window='hamming'):
        nfft = int(series.sample_rate.value * fftlength)
        return signal.get_window(window, nfft)

    @pytest.mark.parametrize('method', [
        'scipy-welch', 'scipy-bartlett',
        'lal-welch', 'lal-bartlett', 'lal-median',
        'pycbc-welch', 'pycbc-bartlett', 'pycbc-median',
    ])
    @pytest.mark.parametrize(
        'window', (None, 'hann', ('kaiser', 24), 'array'),
    )
    def test_spectrogram(self, losc, method, window):
        # generate window for 'array'
        win = self._window_helper(losc, 1) if window == 'array' else window

        # generate spectrogram
        try:
            sg = losc.spectrogram(1, method=method, window=win)
        except ImportError as exc:
            if method.startswith(('lal', 'pycbc')):
                pytest.skip(str(exc))
            raise

        # validate
        assert isinstance(sg, Spectrogram)
        assert sg.shape == (abs(losc.span),
                            losc.sample_rate.value // 2 + 1)
        assert sg.f0 == 0 * units.Hz
        assert sg.df == 1 * units.Hz
        assert sg.channel is losc.channel
        assert sg.unit == losc.unit ** 2 / units.Hz
        assert sg.epoch == losc.epoch
        assert sg.span == losc.span

        # check the first time-bin is the same result as .psd()
        n = int(losc.sample_rate.value)
        if window == 'hann' and not method.endswith('bartlett'):
            n *= 1.5  # default is 50% overlap
        psd = losc[:int(n)].psd(fftlength=1, method=method, window=win)
        # FIXME: epoch should not be excluded here (probably)
        print(psd)
        print(sg[0])
        utils.assert_quantity_sub_equal(sg[0], psd, exclude=['epoch'],
                                        almost_equal=True)

        # test fftlength
        win2 = self._window_helper(losc, .5) if window == 'array' else window
        sg = losc.spectrogram(1, fftlength=0.5, window=win2)
        assert sg.shape == (abs(losc.span),
                            0.5 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second

        # test auto-overlap
        if window == 'hann':
            sg2 = losc.spectrogram(1, fftlength=0.5, overlap=.25,
                                   window='hann')
            utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

        # test multiprocessing
        sg2 = losc.spectrogram(1, fftlength=0.5, nproc=2, window=win)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    @pytest.mark.parametrize('library', [
        pytest.param('lal', marks=utils.skip_missing_dependency('lal')),
        pytest.param('pycbc',
                     marks=utils.skip_missing_dependency('pycbc.psd')),
    ])
    def test_spectrogram_median_mean(self, losc, library):
        method = '{0}-median-mean'.format(library)
        # median-mean will fail on pycbc, and warn on LAL, if not given
        # the correct data for an even number of FFTs

        if library == 'lal':
            with pytest.warns(UserWarning):
                sg = losc.spectrogram(1.5, fftlength=.5, overlap=0,
                                      method=method)
        else:
            sg = losc.spectrogram(1.5, fftlength=.5, overlap=0, method=method)

        # but should still work
        assert sg.dt == 1.5 * units.second
        assert sg.df == 2 * units.Hertz

    def test_spectrogram2(self, losc):
        # test defaults
        sg = losc.spectrogram2(1, overlap=0)
        utils.assert_quantity_sub_equal(
            sg, losc.spectrogram(1, fftlength=1, overlap=0,
                                 method='scipy-welch', window='hann'))

        # test fftlength
        sg = losc.spectrogram2(0.5)
        assert sg.shape == (16, 0.5 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 0.25 * units.second
        # test overlap
        sg = losc.spectrogram2(fftlength=0.25, overlap=0.24)
        assert sg.shape == (399, 0.25 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        # note: bizarre stride length because 4096/100 gets rounded
        assert sg.dt == 0.010009765625 * units.second

    @utils.skip_minimum_version('scipy', '0.16')
    def test_fftgram(self, losc):
        fgram = losc.fftgram(1)
        fs = int(losc.sample_rate.value)
        f, t, sxx = signal.spectrogram(losc, fs,
                                         window='hann',
                                         nperseg=fs,
                                         mode='complex')
        utils.assert_array_equal(losc.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram)
        fgram = losc.fftgram(1, overlap=0.5)
        f, t, sxx = signal.spectrogram(losc, fs,
                                         window='hann',
                                         nperseg=fs,
                                         noverlap=fs//2,
                                         mode='complex')
        utils.assert_array_equal(losc.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram)

    def test_spectral_variance(self, losc):
        variance = losc.spectral_variance(.5)
        assert isinstance(variance, SpectralVariance)
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

    def test_demodulate(self):
        # create a timeseries that is simply one loud sinusoidal oscillation
        # at a particular frequency, then demodulate at that frequency and
        # recover the amplitude and phase
        amp, phase, f = 1., numpy.pi/4, 30
        duration, sample_rate, stride = 600, 4096, 60
        t = numpy.linspace(0, duration, duration*sample_rate)
        data = TimeSeries(amp * numpy.cos(2*numpy.pi*f*t + phase),
                          unit='', times=t)

        # test with exp=True
        demod = data.demodulate(f, stride=stride, exp=True)
        assert demod.unit == data.unit
        assert demod.size == duration // stride
        utils.assert_allclose(numpy.abs(demod.value), amp, rtol=1e-5)
        utils.assert_allclose(numpy.angle(demod.value), phase, rtol=1e-5)

        # test with exp=False, deg=True
        mag, ph = data.demodulate(f, stride=stride)
        assert mag.unit == data.unit
        assert mag.size == ph.size
        assert ph.unit == 'deg'
        utils.assert_allclose(mag.value, amp, rtol=1e-5)
        utils.assert_allclose(ph.value, numpy.rad2deg(phase), rtol=1e-5)

        # test with exp=False, deg=False
        mag, ph = data.demodulate(f, stride=stride, deg=False)
        assert ph.unit == 'rad'
        utils.assert_allclose(ph.value, phase, rtol=1e-5)

    def test_taper(self):
        # create a flat timeseries, then taper it
        t = numpy.linspace(0, 1, 2048)
        data = TimeSeries(numpy.cos(10*numpy.pi*t), times=t, unit='')
        tapered = data.taper()

        # check that the tapered timeseries goes to zero at its ends,
        # and that the operation does not change the original data
        assert tapered[0].value == 0
        assert tapered[-1].value == 0
        assert tapered.unit == data.unit
        assert tapered.size == data.size
        utils.assert_allclose(data.value, numpy.cos(10*numpy.pi*t))

    def test_inject(self):
        # create a timeseries out of an array of zeros
        duration, sample_rate = 1, 4096
        data = TimeSeries(numpy.zeros(duration*sample_rate), t0=0,
                          sample_rate=sample_rate, unit='')

        # create a second timeseries to inject into the first
        w_times = data.times.value[:2048]
        waveform = TimeSeries(numpy.cos(2*numpy.pi*30*w_times), times=w_times)

        # test that we recover this waveform when we add it to data,
        # and that the operation does not change the original data
        new_data = data.inject(waveform)
        assert new_data.unit == data.unit
        assert new_data.size == data.size
        ind, = new_data.value.nonzero()
        assert len(ind) == waveform.size
        utils.assert_allclose(new_data.value[ind], waveform.value)
        utils.assert_allclose(data.value, numpy.zeros(duration*sample_rate))

    def test_whiten(self):
        # create noise with a glitch in it at 1000 Hz
        noise = self.TEST_CLASS(
            numpy.random.normal(loc=1, size=16384 * 10), sample_rate=16384,
            epoch=-5).zpk([], [0], 1)
        glitchtime = 0.5
        glitch = signal.gausspulse(noise.times.value - glitchtime,
                                   bw=100) * 1e-4
        data = noise + glitch

        # whiten and test that the max amplitude is recovered at the glitch
        tmax = data.times[data.argmax()]
        assert not numpy.isclose(tmax.value, glitchtime)

        whitened = data.whiten(2, 1)

        assert whitened.size == noise.size
        nptest.assert_almost_equal(whitened.mean().value, 0.0, decimal=4)
        nptest.assert_almost_equal(whitened.std().value, 1.0, decimal=4)

        tmax = whitened.times[whitened.argmax()]
        nptest.assert_almost_equal(tmax.value, glitchtime)

    def test_detrend(self, losc):
        assert not numpy.isclose(losc.value.mean(), 0.0, atol=1e-21)
        detrended = losc.detrend()
        assert numpy.isclose(detrended.value.mean(), 0.0)

    def test_filter(self, losc):
        zpk = [], [], 1
        fts = losc.filter(zpk, analog=True)
        utils.assert_quantity_sub_equal(losc, fts)

        # check SOS filters can be used directly
        zpk = filter_design.highpass(50, sample_rate=losc.sample_rate)
        try:
            sos = signal.zpk2sos(*zpk)
        except AttributeError:  # scipy < 0.16
            pass
        else:
            utils.assert_quantity_almost_equal(losc.filter(zpk),
                                               losc.filter(sos))

    def test_zpk(self, losc):
        zpk = [10, 10], [1, 1], 100
        utils.assert_quantity_sub_equal(
            losc.zpk(*zpk), losc.filter(*zpk, analog=True))

    def test_notch(self, losc):
        # test notch runs end-to-end
        losc.notch(60)

        # test breaks when you try and 'fir' notch
        with pytest.raises(NotImplementedError):
            losc.notch(10, type='fir')

    def test_q_transform(self, losc):
        # test simple q-transform
        qspecgram = losc.q_transform(method='scipy-welch', fftlength=2)
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (4000, 2403)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 156.0411964254892)

        # test whitening args
        asd = losc.asd(2, 1, method='scipy-welch')
        qsg2 = losc.q_transform(method='scipy-welch', whiten=asd)
        utils.assert_quantity_sub_equal(qspecgram, qsg2, almost_equal=True)

        asd = losc.asd(.5, .25, method='scipy-welch')
        qsg2 = losc.q_transform(method='scipy-welch', whiten=asd)
        qsg3 = losc.q_transform(method='scipy-welch',
                                fftlength=.5, overlap=.25)
        utils.assert_quantity_sub_equal(qsg2, qsg3, almost_equal=True)

        # make sure frequency too high presents warning
        with pytest.warns(UserWarning):
            qspecgram = losc.q_transform(method='scipy-welch',
                                         frange=(0, 10000))
            nptest.assert_almost_equal(qspecgram.yspan[1], 1291.5316316157107)

        # test other normalisations work (or don't)
        q2 = losc.q_transform(method='scipy-welch', norm='median')
        utils.assert_quantity_sub_equal(qspecgram, q2, almost_equal=True)
        losc.q_transform(method='scipy-welch', norm='mean')
        losc.q_transform(method='scipy-welch', norm=False)
        with pytest.raises(ValueError):
            losc.q_transform(method='scipy-welch', norm='blah')

    def test_boolean_statetimeseries(self, array):
        comp = array >= 2 * array.unit
        assert isinstance(comp, StateTimeSeries)
        assert comp.unit is units.Unit('')
        assert comp.name == '%s >= 2.0' % (array.name)
        assert (array == array).name == '{0} == {0}'.format(array.name)

    def test_coherence(self):
        try:
            tsh = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
            tsl = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
        except LOSC_FETCH_ERROR as exc:
            pytest.skip(str(exc))
        coh = tsh.coherence(tsl, fftlength=1.0)
        assert coh.df == 1 * units.Hz
        assert coh.frequencies[coh.argmax()] == 60 * units.Hz

    def test_coherence_spectrogram(self):
        try:
            tsh = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
            tsl = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
        except LOSC_FETCH_ERROR as exc:
            pytest.skip(str(exc))
        cohsg = tsh.coherence_spectrogram(tsl, 4, fftlength=1.0)
        assert cohsg.t0 == tsh.t0
        assert cohsg.dt == 4 * units.second
        assert cohsg.df == 1 * units.Hz
        tmax, fmax = numpy.unravel_index(cohsg.argmax(), cohsg.shape)
        assert cohsg.frequencies[fmax] == 60 * units.Hz


# -- TimeSeriesDict -----------------------------------------------------------

class TestTimeSeriesDict(_TestTimeSeriesBaseDict):
    channels = ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN']
    TEST_CLASS = TimeSeriesDict
    ENTRY_CLASS = TimeSeries

    @utils.skip_missing_dependency('LDAStools.frameCPP')
    def test_read_write_gwf(self, instance):
        with utils.TemporaryFilename(suffix='.gwf') as tmp:
            instance.write(tmp)
            new = self.TEST_CLASS.read(tmp, instance.keys())
            for key in new:
                utils.assert_quantity_sub_equal(new[key], instance[key],
                                                exclude=['channel'])

    def test_read_write_hdf5(self, instance):
        with utils.TemporaryFilename(suffix='.hdf5') as tmp:
            instance.write(tmp, overwrite=True)
            new = self.TEST_CLASS.read(tmp, instance.keys())
            for key in new:
                utils.assert_quantity_sub_equal(new[key], instance[key])
            # check auto-detection of names
            new = self.TEST_CLASS.read(tmp)
            for key in new:
                utils.assert_quantity_sub_equal(new[key], instance[key])


# -- TimeSeriesList -----------------------------------------------------------

class TestTimeSeriesList(_TestTimeSeriesBaseList):
    TEST_CLASS = TimeSeriesList
    ENTRY_CLASS = TimeSeries
