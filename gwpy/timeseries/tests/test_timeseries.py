# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

import os.path
from itertools import (chain, product)
from ssl import SSLError

import six
from six.moves.urllib.error import URLError

import pytest

import numpy
from numpy import testing as nptest

from scipy import (signal, __version__ as scipy_version)

from astropy import units

from ...frequencyseries import (FrequencySeries, SpectralVariance)
from ...segments import Segment
from ...signal import filter_design
from ...table import EventTable
from ...spectrogram import Spectrogram
from ...testing import (mocks, utils)
from ...testing.compat import mock
from ...time import LIGOTimeGPS
from ...utils.misc import null_context
from .. import (TimeSeries, TimeSeriesDict, TimeSeriesList, StateTimeSeries)
from ..io.gwf import APIS as GWF_APIS
from .test_core import (TestTimeSeriesBase as _TestTimeSeriesBase,
                        TestTimeSeriesBaseDict as _TestTimeSeriesBaseDict,
                        TestTimeSeriesBaseList as _TestTimeSeriesBaseList)

SKIP_FRAMECPP = utils.skip_missing_dependency('LDAStools.frameCPP')
SKIP_LAL = utils.skip_missing_dependency('lal')
SKIP_LALFRAME = utils.skip_missing_dependency('lalframe')
SKIP_PYCBC_PSD = utils.skip_missing_dependency('pycbc.psd')

if scipy_version < '1.2.0':
    SCIPY_METHODS = ('welch', 'bartlett')
else:
    SCIPY_METHODS = ('welch', 'bartlett', 'median')

FIND_CHANNEL = 'L1:DCS-CALIB_STRAIN_C02'
FIND_FRAMETYPE = 'L1_HOFT_C02'

LOSC_IFO = 'L1'
LOSC_GW150914 = 1126259462
LOSC_GW150914_SEGMENT = Segment(LOSC_GW150914-2, LOSC_GW150914+2)
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
        pytest.param('lalframe', marks=SKIP_LALFRAME),
        pytest.param('framecpp', marks=SKIP_FRAMECPP),
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
            with pytest.deprecated_call():
                t = read_(dtype='float32')
            assert t.dtype is numpy.dtype('float32')
            with pytest.deprecated_call():
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

    @pytest.mark.parametrize('api', [
        pytest.param('framecpp', marks=SKIP_FRAMECPP),
    ])
    def test_read_write_gwf_error(self, api, losc):
        with utils.TemporaryFilename(suffix=".gwf") as tmp:
            losc.write(tmp, format="gwf.{}".format(api))
            with pytest.raises(ValueError) as exc:
                self.TEST_CLASS.read(tmp, "another channel",
                                     format="gwf.{}".format(api))
            assert str(exc.value) == (
                "no Fr{Adc,Proc,Sim}Data structures with the "
                "name another channel"
            )

            with pytest.raises(ValueError) as exc:
                self.TEST_CLASS.read(tmp, losc.name,
                                     start=losc.span[0]-1, end=losc.span[0],
                                     format="gwf.{}".format(api))
            assert str(exc.value).startswith(
                "Failed to read {0!r} from {1!r}".format(losc.name, tmp)
            )

    @SKIP_LALFRAME
    def test_read_gwf_scaled_lalframe(self):
        with pytest.warns(None) as record:
            data = self.TEST_CLASS.read(
                utils.TEST_GWF_FILE,
                "L1:LDAS-STRAIN",
                format="gwf.lalframe",
            )
        assert not record.list  # no warning
        with pytest.warns(UserWarning):
            data2 = self.TEST_CLASS.read(
                utils.TEST_GWF_FILE,
                "L1:LDAS-STRAIN",
                format="gwf.lalframe",
                scaled=True,
            )
        utils.assert_quantity_sub_equal(data, data2)

    @SKIP_FRAMECPP
    @SKIP_LALFRAME
    @pytest.mark.parametrize("ctype", ("adc", "proc", "sim", None))
    @pytest.mark.parametrize("format_", GWF_APIS)
    def test_write_gwf_type(self, losc, format_, ctype):
        from ...io.gwf import get_channel_type

        # on debian, python=3, python-ldas-tools-framecpp < 2.6.9,
        # the simdata test causes a segfault
        import platform
        import sys
        if (
            format_ == "framecpp" and
            ctype == "sim" and
            sys.version_info[0] >= 3 and
            "debian" in platform.platform()
        ):
            pytest.xfail(
                "reading Sim data with "
                "python-ldas-tools-framecpp < 2.6.9 is broken"
            )

        gwfformat = "gwf.{}".format(format_)
        expected_ctype = ctype if ctype else "proc"

        with utils.TemporaryFilename(suffix=".gwf") as tmp:
            losc.write(tmp, type=ctype, format=gwfformat)
            assert get_channel_type(losc.name, tmp) == expected_ctype
            try:
                new = type(losc).read(tmp, losc.name, format=gwfformat)
            except OverflowError:
                # python-ldas-tools-framecpp < 2.6.9
                if format_ == "framecpp" and ctype == "sim":
                    pytest.xfail(
                        "reading Sim data with "
                        "python-ldas-tools-framecpp < 2.6.9 is broken"
                    )
                raise
        # epoch seems to mismatch at O(1e-12), which is unfortunate
        utils.assert_quantity_sub_equal(
            losc,
            new,
            exclude=("channel", "x0"),
        )

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
            with pytest.raises((RuntimeError, OSError)):
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

    def test_read_pad(self):
        a = self.TEST_CLASS.read(
            utils.TEST_HDF5_FILE,
            "H1:LDAS-STRAIN",
        )
        b = self.TEST_CLASS.read(
            utils.TEST_HDF5_FILE,
            "H1:LDAS-STRAIN",
            pad=0.,
            start=a.span[0]-1,
            end=a.span[1]+1,
        )
        utils.assert_quantity_sub_equal(
            a.pad(
                (int(a.sample_rate.value), int(a.sample_rate.value)),
                mode="constant",
                constant_values=(0,),
            ),
            b,
        )

    @utils.skip_missing_dependency('nds2')
    def test_from_nds2_buffer_dynamic_scaled(self):
        # build fake buffer for LIGO channel
        nds_buffer = mocks.nds2_buffer(
            'H1:TEST',
            self.data,
            1000000000,
            self.data.shape[0],
            'm',
            name='test',
            slope=2,
            offset=1,
        )

        # check scaling defaults to off
        utils.assert_array_equal(
            self.TEST_CLASS.from_nds2_buffer(nds_buffer).value,
            nds_buffer.data,
        )
        utils.assert_array_equal(
            self.TEST_CLASS.from_nds2_buffer(nds_buffer, scaled=False).value,
            nds_buffer.data,
        )
        utils.assert_array_equal(
            self.TEST_CLASS.from_nds2_buffer(nds_buffer, scaled=True).value,
            nds_buffer.data * 2 + 1,
        )

    # -- test remote data access ----------------

    @utils.skip_minimum_version("gwosc", "0.4.0")
    @pytest.mark.parametrize('format', [
        'hdf5',
        pytest.param('gwf', marks=SKIP_FRAMECPP),
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

    @utils.skip_missing_dependency('nds2')
    @pytest.mark.parametrize('protocol', (1, 2))
    def test_fetch(self, protocol):
        ts = self.create(name='L1:TEST', t0=1000000000, unit='m')
        nds_buffer = mocks.nds2_buffer_from_timeseries(ts)
        nds_connection = mocks.nds2_connection(buffers=[nds_buffer],
                                               protocol=protocol)
        with mock.patch('nds2.connection') as mock_connection, \
                mock.patch('nds2.buffer', nds_buffer):
            mock_connection.return_value = nds_connection

            # use verbose=True to hit more lines
            ts2 = self.TEST_CLASS.fetch('L1:TEST', *ts.span, verbose=True)
            utils.assert_quantity_sub_equal(ts, ts2, exclude=['channel'])

            # check open connection works
            ts2 = self.TEST_CLASS.fetch('L1:TEST', *ts.span, verbose=True,
                                        connection=nds_connection)
            utils.assert_quantity_sub_equal(ts, ts2, exclude=['channel'])

            # check padding works (with warning for nds2-server connections)
            ctx = pytest.warns(UserWarning) if protocol > 1 else null_context()
            with ctx:
                ts2 = self.TEST_CLASS.fetch('L1:TEST', *ts.span.protract(10),
                                            pad=-100., host='anything')
            assert ts2.span == ts.span.protract(10)
            assert ts2[0] == -100. * ts.unit
            assert ts2[10] == ts[0]
            assert ts2[-11] == ts[-1]
            assert ts2[-1] == -100. * ts.unit

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
            assert 'no data received' in str(exc.value)

    @SKIP_FRAMECPP
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

    @SKIP_FRAMECPP
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

    @SKIP_FRAMECPP
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
                                     frametype_match=r'C01\Z')
        except (ImportError, RuntimeError) as e:
            pytest.skip(str(e))
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

    def test_psd_default_overlap(self, losc):
        utils.assert_quantity_sub_equal(
            losc.psd(.5, window='hann'),
            losc.psd(.5, .25, window='hann'),
        )

    @SKIP_LAL
    def test_psd_lal_median_mean(self, losc):
        # check that warnings and errors get raised in the right place
        # for a median-mean PSD with the wrong data size or parameters

        # single segment should raise error
        with pytest.raises(ValueError), pytest.deprecated_call():
            losc.psd(abs(losc.span), method='lal_median_mean')

        # odd number of segments should warn
        # pytest hides the second DeprecationWarning that should have been
        # triggered here, for some reason
        with pytest.warns(UserWarning):
            losc.psd(1, .5, method='lal_median_mean')

    @pytest.mark.parametrize('method', SCIPY_METHODS)
    def test_psd(self, noisy_sinusoid, method):
        fftlength = .5
        overlap = .25
        fs = noisy_sinusoid.psd(fftlength=fftlength, overlap=overlap)
        assert fs.unit == noisy_sinusoid.unit ** 2 / "Hz"
        assert fs.max() == fs.value_at(500)
        assert fs.size == fftlength * noisy_sinusoid.sample_rate.value // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == units.Hz / fftlength
        assert fs.name == noisy_sinusoid.name
        assert fs.channel is noisy_sinusoid.channel

    @pytest.mark.parametrize('library, method', chain(
        product(['pycbc.psd'], ['welch', 'bartlett', 'median', 'median_mean']),
        product(['lal'], ['welch', 'bartlett', 'median', 'median_mean']),
    ))
    def test_psd_deprecated(self, noisy_sinusoid, library, method):
        """Test deprecated average methods for TimeSeries.psd
        """
        pytest.importorskip(library)

        fftlength = .5
        overlap = .25

        # remove final .25 seconds to stop median-mean complaining
        # (means an even number of overlapping FFT segments)
        if method == "median_mean":
            end = noisy_sinusoid.span[1]
            noisy_sinusoid = noisy_sinusoid.crop(end=end-overlap)

        # get actual method name
        library = library.split('.', 1)[0]

        with pytest.deprecated_call():
            psd = noisy_sinusoid.psd(fftlength=fftlength, overlap=overlap,
                                     method="{0}-{1}".format(library, method))

        assert isinstance(psd, FrequencySeries)
        assert psd.unit == noisy_sinusoid.unit ** 2 / "Hz"
        assert psd.max() == psd.value_at(500)

    def test_asd(self, losc):
        fs = losc.asd(1)
        utils.assert_quantity_sub_equal(fs, losc.psd(1) ** (1/2.))

    @utils.skip_minimum_version('scipy', '0.16')
    def test_csd(self, noisy_sinusoid, corrupt_noisy_sinusoid):
        # test that csd(self) is the same as psd()
        fs = noisy_sinusoid.csd(noisy_sinusoid)
        utils.assert_quantity_sub_equal(
            fs,
            noisy_sinusoid.psd(),
            exclude=['name'],
        )

        # test fftlength
        fs = noisy_sinusoid.csd(corrupt_noisy_sinusoid, fftlength=0.5)
        assert fs.size == 0.5 * noisy_sinusoid.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hertz
        utils.assert_quantity_sub_equal(
            fs,
            noisy_sinusoid.csd(corrupt_noisy_sinusoid, fftlength=0.5,
                               overlap=0.25),
        )

    @staticmethod
    def _window_helper(series, fftlength, window='hamming'):
        nfft = int(series.sample_rate.value * fftlength)
        return signal.get_window(window, nfft)

    @pytest.mark.parametrize('method', [
        'scipy-welch',
        'scipy-bartlett',
        pytest.param('lal-welch', marks=SKIP_LAL),
        pytest.param('lal-bartlett', marks=SKIP_LAL),
        pytest.param('lal-median', marks=SKIP_LAL),
        pytest.param('pycbc-welch', marks=SKIP_PYCBC_PSD),
        pytest.param('pycbc-bartlett', marks=SKIP_PYCBC_PSD),
        pytest.param('pycbc-median', marks=SKIP_PYCBC_PSD),
    ])
    @pytest.mark.parametrize(
        'window', (None, 'hann', ('kaiser', 24), 'array'),
    )
    def test_spectrogram(self, losc, method, window):
        # generate window for 'array'
        win = self._window_helper(losc, 1) if window == 'array' else window

        if method.startswith(("lal", "pycbc")):
            ctx = pytest.deprecated_call
        else:
            ctx = null_context

        # generate spectrogram
        with ctx():
            sg = losc.spectrogram(1, method=method, window=win)

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
        with ctx():
            psd = losc[:int(n)].psd(fftlength=1, method=method, window=win)
        # FIXME: epoch should not be excluded here (probably)
        utils.assert_quantity_sub_equal(sg[0], psd, exclude=['epoch'],
                                        almost_equal=True)

    def test_spectrogram_fftlength(self, losc):
        sg = losc.spectrogram(1, fftlength=0.5)
        assert sg.shape == (abs(losc.span),
                            0.5 * losc.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second

    def test_spectrogram_overlap(self, losc):
        sg = losc.spectrogram(1, fftlength=0.5, window="hann")
        sg2 = losc.spectrogram(1, fftlength=0.5, window="hann", overlap=.25)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    def test_spectrogram_multiprocessing(self, losc):
        sg = losc.spectrogram(1, fftlength=0.5)
        sg2 = losc.spectrogram(1, fftlength=0.5, nproc=2)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    @pytest.mark.parametrize('library', [
        pytest.param('lal', marks=SKIP_LAL),
        pytest.param('pycbc', marks=SKIP_PYCBC_PSD),
    ])
    def test_spectrogram_median_mean(self, losc, library):
        method = '{0}-median-mean'.format(library)

        # median-mean warn on LAL if not given the correct data for an
        # even number of FFTs.
        # pytest only asserts a single warning, and UserWarning will take
        # precedence apparently, so check that for lal
        if library == 'lal':
            warn_ctx = pytest.warns(UserWarning)
        else:
            warn_ctx = pytest.deprecated_call()

        with warn_ctx:
            sg = losc.spectrogram(1.5, fftlength=.5, overlap=0, method=method)

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
        f, t, sxx = signal.spectrogram(
            losc, fs,
            window='hann',
            nperseg=fs,
            mode='complex',
        )
        utils.assert_array_equal(losc.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram)

        fgram = losc.fftgram(1, overlap=0.5)
        f, t, sxx = signal.spectrogram(
            losc, fs,
            window='hann',
            nperseg=fs,
            noverlap=fs//2,
            mode='complex',
        )
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

    def test_resample_noop(self):
        data = self.TEST_CLASS([1, 2, 3, 4, 5])
        with pytest.warns(UserWarning):
            new = data.resample(data.sample_rate)
            assert data is new

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

    def test_heterodyne(self):
        # create a timeseries that is simply one loud sinusoidal oscillation,
        # with a frequency and frequency derivative, then heterodyne using the
        # phase evolution to recover the amplitude and phase
        amp, phase, f, fdot = 1., numpy.pi/4, 30, 1e-4
        duration, sample_rate, stride = 600, 4096, 60
        t = numpy.linspace(0, duration, duration*sample_rate)
        phases = 2*numpy.pi*(f*t + 0.5*fdot*t**2)
        data = TimeSeries(amp * numpy.cos(phases + phase),
                          unit='', times=t)

        # test exceptions
        with pytest.raises(TypeError):
            data.heterodyne(1.0)

        with pytest.raises(ValueError):
            data.heterodyne(phases[0:len(phases) // 2])

        # test with default settings
        het = data.heterodyne(phases, stride=stride)
        assert het.unit == data.unit
        assert het.size == duration // stride
        utils.assert_allclose(numpy.abs(het.value), 0.5*amp, rtol=1e-4)
        utils.assert_allclose(numpy.angle(het.value), phase, rtol=2e-4)

        # test with singlesided=True
        het = data.heterodyne(
            phases, stride=stride, singlesided=True
        )
        assert het.unit == data.unit
        assert het.size == duration // stride
        utils.assert_allclose(numpy.abs(het.value), amp, rtol=1e-4)
        utils.assert_allclose(numpy.angle(het.value), phase, rtol=2e-4)

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

        # run the same tests for a user-specified taper duration
        dtapered = data.taper(duration=0.1)
        assert dtapered[0].value == 0
        assert dtapered[-1].value == 0
        assert dtapered.unit == data.unit
        assert dtapered.size == data.size
        utils.assert_allclose(data.value, numpy.cos(10*numpy.pi*t))

        # run the same tests for a user-specified number of samples to taper
        stapered = data.taper(nsamples=10)
        assert stapered[0].value == 0
        assert stapered[-1].value == 0
        assert stapered.unit == data.unit
        assert stapered.size == data.size
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

    @utils.skip_minimum_version("scipy", "1.1.0")
    def test_gate(self):
        # generate Gaussian noise with std = 0.5
        noise = self.TEST_CLASS(numpy.random.normal(scale=0.5, size=16384*64),
                                sample_rate=16384, epoch=-32)
        # generate a glitch with amplitude 20 at 1000 Hz
        glitchtime = 0.0
        glitch = signal.gausspulse(noise.times.value - glitchtime,
                                   bw=100) * 20
        data = noise + glitch

        # check that the glitch is at glitchtime as expected
        tmax = data.times.value[data.argmax()]
        nptest.assert_almost_equal(tmax, glitchtime)

        # gating method will be called with whiten = False to decouple
        # whitening method from gating method
        tzero = 1.0
        tpad = 1.0
        threshold = 10.0
        gated = data.gate(tzero=tzero, tpad=tpad, threshold=threshold,
                          whiten=False)

        # check that the maximum value is not within the region set to zero
        tleft = glitchtime - tzero
        tright = glitchtime + tzero
        assert not tleft < gated.times.value[gated.argmax()] < tright

        # check that there are no remaining values above the threshold
        assert gated.max() < threshold

    def test_whiten(self):
        # create noise with a glitch in it at 1000 Hz
        noise = self.TEST_CLASS(
            numpy.random.normal(loc=1, scale=.5, size=16384 * 64),
            sample_rate=16384, epoch=-32).zpk([], [0], 1)
        glitchtime = 0.5
        glitch = signal.gausspulse(noise.times.value - glitchtime,
                                   bw=100) * 1e-4
        data = noise + glitch

        # when the input is stationary Gaussian noise, the output should have
        # zero mean and unit variance
        whitened = noise.whiten(detrend='linear')
        assert whitened.size == noise.size
        nptest.assert_almost_equal(whitened.mean().value, 0.0, decimal=2)
        nptest.assert_almost_equal(whitened.std().value, 1.0, decimal=2)

        # when a loud signal is present, the max amplitude should be recovered
        # at the time of that signal
        tmax = data.times[data.argmax()]
        assert not numpy.isclose(tmax.value, glitchtime)

        whitened = data.whiten(detrend='linear')
        tmax = whitened.times[whitened.argmax()]
        nptest.assert_almost_equal(tmax.value, glitchtime)

    def test_convolve(self):
        data = self.TEST_CLASS(
            signal.hann(1024), sample_rate=512, epoch=-1
        )
        filt = numpy.array([1, 0])

        # check that the 'valid' data are unchanged by this filter
        convolved = data.convolve(filt)
        assert convolved.size == data.size
        utils.assert_allclose(convolved.value[1:-1], data.value[1:-1])

    def test_correlate(self):
        # create noise and a glitch template at 1000 Hz
        noise = self.TEST_CLASS(
            numpy.random.normal(size=16384 * 64), sample_rate=16384, epoch=-32
            ).zpk([], [1], 1)
        glitchtime = -16.5
        glitch = self.TEST_CLASS(
            signal.gausspulse(numpy.arange(-1, 1, 1./16384), bw=100),
            sample_rate=16384, epoch=glitchtime-1)

        # check that, without a signal present, we only see background
        snr = noise.correlate(glitch, whiten=True)
        tmax = snr.times[snr.argmax()]
        assert snr.size == noise.size
        assert not numpy.isclose(tmax.value, glitchtime)
        nptest.assert_almost_equal(snr.mean().value, 0.0, decimal=1)
        nptest.assert_almost_equal(snr.std().value, 1.0, decimal=1)

        # inject and recover the glitch
        data = noise.inject(glitch * 1e-4)
        snr = data.correlate(glitch, whiten=True)
        tmax = snr.times[snr.argmax()]
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

    def test_q_gram(self, losc):
        # test simple q-transform
        qgram = losc.q_gram()
        assert isinstance(qgram, EventTable)
        assert qgram.meta['q'] == 45.25483399593904
        assert qgram['energy'].min() >= 5.5**2 / 2
        nptest.assert_almost_equal(qgram['energy'].max(), 10559.25, decimal=2)

    def test_q_transform(self, losc):
        # test simple q-transform
        qspecgram = losc.q_transform(method='scipy-welch', fftlength=2)
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (1000, 2403)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 155.93567, decimal=5)

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
            nptest.assert_almost_equal(
                qspecgram.yspan[1], 1291.5316, decimal=4)

        # test other normalisations work (or don't)
        q2 = losc.q_transform(method='scipy-welch', norm='median')
        utils.assert_quantity_sub_equal(qspecgram, q2, almost_equal=True)
        losc.q_transform(method='scipy-welch', norm='mean')
        losc.q_transform(method='scipy-welch', norm=False)
        with pytest.raises(ValueError):
            losc.q_transform(method='scipy-welch', norm='blah')

    def test_q_transform_logf(self, losc):
        # test q-transform with log frequency spacing
        qspecgram = losc.q_transform(method='scipy-welch', fftlength=2,
                                     logf=True)
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (1000, 500)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 155.93774, decimal=5)

    def test_q_transform_nan(self):
        data = TimeSeries(numpy.empty(256*10) * numpy.nan, sample_rate=256)
        with pytest.raises(ValueError) as exc:
            data.q_transform()
        assert str(exc.value) == 'Input signal contains non-numerical values'

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

    @SKIP_FRAMECPP
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
