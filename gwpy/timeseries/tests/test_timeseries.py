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

import os.path
import warnings
from contextlib import nullcontext
from itertools import (chain, product)
from unittest import mock

import pytest

import numpy
from numpy import testing as nptest

from scipy import signal

from astropy import units

from ...frequencyseries import (FrequencySeries, SpectralVariance)
from ...segments import (Segment, SegmentList, DataQualityFlag)
from ...signal import filter_design
from ...signal.window import planck
from ...spectrogram import Spectrogram
from ...table import EventTable
from ...testing import (mocks, utils)
from ...testing.errors import (
    pytest_skip_cvmfs_read_error,
    pytest_skip_network_error,
)
from ...types import Index
from ...time import LIGOTimeGPS
from .. import (TimeSeries, TimeSeriesDict, TimeSeriesList, StateTimeSeries)
from ..io.gwf import get_default_gwf_api
from .test_core import (TestTimeSeriesBase as _TestTimeSeriesBase,
                        TestTimeSeriesBaseDict as _TestTimeSeriesBaseDict,
                        TestTimeSeriesBaseList as _TestTimeSeriesBaseList)

try:
    get_default_gwf_api()
except ImportError:
    HAVE_GWF_API = False
else:
    HAVE_GWF_API = True

GWF_APIS = [
    pytest.param(
        None,
        marks=pytest.mark.skipif(not HAVE_GWF_API, reason="no GWF API"),
    ),
    pytest.param('lalframe', marks=pytest.mark.requires("lalframe")),
    pytest.param('framecpp', marks=pytest.mark.requires("LDAStools.frameCPP")),
    pytest.param('framel', marks=pytest.mark.requires("framel")),
]


LIVETIME = DataQualityFlag(
    name='X1:TEST-FLAG:1',
    active=SegmentList([
        Segment(0, 32),
        Segment(34, 34.5),
    ]),
    known=SegmentList([Segment(0, 64)]),
    isgood=True,
)

GWOSC_DATAFIND_SERVER = "datafind.gw-openscience.org"
GWOSC_GW150914_IFO = "L1"
GWOSC_GW150914_CHANNEL = "L1:GWOSC-16KHZ_R1_STRAIN"
NDS2_GW150914_CHANNEL = "L1:DCS-CALIB_STRAIN_C02"
GWOSC_GW150914_FRAMETYPE = "L1_LOSC_16_V1"
GWOSC_GW150914 = 1126259462
GWOSC_GW150914_SEGMENT = Segment(GWOSC_GW150914-2, GWOSC_GW150914+2)
GWOSC_GW150914_DQ_BITS = {
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

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def _gwosc_cvmfs(func):
    """Decorate ``func`` with all necessary CVMFS-related decorators
    """
    for dec in (
        pytest.mark.cvmfs,
        pytest.mark.requires("LDAStools.frameCPP"),
        pytest.mark.skipif(
            not os.path.isdir('/cvmfs/gwosc.osgstorage.org/'),
            reason="GWOSC CVMFS repository not available",
        ),
        pytest_skip_cvmfs_read_error,
    ):
        func = dec(func)
    return func


class TestTimeSeries(_TestTimeSeriesBase):
    TEST_CLASS = TimeSeries

    # -- fixtures -------------------------------

    @pytest.fixture(scope='class')
    @pytest_skip_network_error
    def gw150914(self):
        return self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            *GWOSC_GW150914_SEGMENT,
        )

    @pytest.fixture(scope='class')
    @pytest_skip_network_error
    def gw150914_16384(self):
        return self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            *GWOSC_GW150914_SEGMENT,
            sample_rate=16384,
        )

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

    def test_read_ascii_header(self, tmpdir):
        """Check that ASCII files with headers are read without extra options

        [regression: https://github.com/gwpy/gwpy/issues/1473]
        """
        txt = tmpdir / "text.txt"
        txt.write_text(
            "# time (s)\tdata (strain)\n0\t1\n1\t2\n2\t3",
            encoding="utf-8",
        )
        data = self.TEST_CLASS.read(txt, format="txt")
        utils.assert_array_equal(data.times, Index((0, 1, 2), unit="s"))
        utils.assert_array_equal(data.value, (1, 2, 3))

    @pytest.mark.parametrize('api', GWF_APIS)
    def test_read_write_gwf(self, tmp_path, api):
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
        except ImportError as e:  # pragma: no-cover
            pytest.skip(str(e))

        # test read keyword arguments
        tmp = tmp_path / "test.gwf"
        array.write(tmp, format=fmt)

        def read_(**kwargs):
            return type(array).read(tmp, array.name, format=fmt,
                                    **kwargs)

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

    @pytest.mark.parametrize('api', GWF_APIS)
    def test_read_write_gwf_gps_errors(self, tmp_path, api):
        fmt = "gwf" if api is None else "gwf." + api
        array = self.create(name='TEST')
        tmp = tmp_path / "test.gwf"
        array.write(tmp, format=fmt)

        # check that reading past the end of the array fails
        with pytest.raises((ValueError, RuntimeError)):
            self.TEST_CLASS.read(
                tmp,
                array.name,
                format=fmt,
                start=array.span[1],
            )

        # check that reading before the start of the array also fails
        with pytest.raises((ValueError, RuntimeError)):
            self.TEST_CLASS.read(
                tmp,
                array.name,
                format=fmt,
                end=array.span[0]-1,
            )

    @pytest.mark.parametrize('api', GWF_APIS)
    @pytest.mark.parametrize('nproc', (1, 2))
    def test_read_write_gwf_multiple(self, tmp_path, api, nproc):
        """Check that each GWF API can read a series of files, either in
        a single process, or in multiple processes

        Regression: https://github.com/gwpy/gwpy/issues/1486
        """
        fmt = "gwf" if api is None else "gwf." + api
        a1 = self.create(name='TEST')
        a2 = self.create(name='TEST', t0=a1.span[1], dt=a1.dx)

        tmp1 = tmp_path / "test1.gwf"
        tmp2 = tmp_path / "test3.gwf"
        a1.write(tmp1, format=fmt)
        a2.write(tmp2, format=fmt)
        cache = [tmp1, tmp2]

        comb = self.TEST_CLASS.read(
            cache,
            'TEST',
            start=a1.span[0],
            end=a2.span[1],
            format=fmt,
            nproc=nproc,
        )
        utils.assert_quantity_sub_equal(
            comb,
            a1.append(a2, inplace=False),
            exclude=['channel'],
        )

    @pytest.mark.parametrize('api', [
        pytest.param(
            'framecpp',
            marks=pytest.mark.requires("LDAStools.frameCPP"),
        ),
    ])
    def test_read_write_gwf_error(self, tmp_path, api, gw150914):
        tmp = tmp_path / "test.gwf"
        gw150914.write(tmp, format="gwf.{}".format(api))
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS.read(tmp, "another channel",
                                 format="gwf.{}".format(api))
        assert str(exc.value) == (
            "no Fr{Adc,Proc,Sim}Data structures with the "
            "name another channel"
        )

        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS.read(
                tmp,
                gw150914.name,
                start=gw150914.span[0]-1,
                end=gw150914.span[0],
                format="gwf.{}".format(api),
            )
        assert str(exc.value).startswith(
            "Failed to read {0!r} from {1!r}".format(gw150914.name, str(tmp))
        )

    @pytest.mark.requires("lalframe")
    def test_read_gwf_scaled_lalframe(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            data = self.TEST_CLASS.read(
                utils.TEST_GWF_FILE,
                "L1:LDAS-STRAIN",
                format="gwf.lalframe",
            )

        with pytest.warns(UserWarning):
            data2 = self.TEST_CLASS.read(
                utils.TEST_GWF_FILE,
                "L1:LDAS-STRAIN",
                format="gwf.lalframe",
                scaled=True,
            )
        utils.assert_quantity_sub_equal(data, data2)

    @pytest.mark.requires("LDAStools.frameCPP")
    @pytest.mark.parametrize("ctype", ("adc", "proc", "sim", None))
    @pytest.mark.parametrize("api", GWF_APIS)
    def test_write_gwf_type(self, gw150914, tmp_path, api, ctype):
        from ...io.gwf import get_channel_type

        # on debian, python=3, python-ldas-tools-framecpp < 2.6.9,
        # the simdata test causes a segfault
        import platform
        import sys
        if (
            api == "framecpp"
            and ctype == "sim"
            and sys.version_info[0] >= 3
            and "debian" in platform.platform()
        ):
            pytest.xfail(
                "reading Sim data with "
                "python-ldas-tools-framecpp < 2.6.9 is broken"
            )

        fmt = "gwf" if api is None else "gwf." + api
        expected_ctype = ctype if ctype else "proc"

        tmp = tmp_path / "test.gwf"
        gw150914.write(tmp, type=ctype, format=fmt)
        assert get_channel_type(gw150914.name, tmp) == expected_ctype
        try:
            new = type(gw150914).read(tmp, gw150914.name, format=fmt)
        except OverflowError:
            # python-ldas-tools-framecpp < 2.6.9
            if api == "framecpp" and ctype == "sim":
                pytest.xfail(
                    "reading Sim data with "
                    "python-ldas-tools-framecpp < 2.6.9 is broken"
                )
            raise

        # epoch seems to mismatch at O(1e-12), which is unfortunate
        utils.assert_quantity_sub_equal(
            gw150914,
            new,
            exclude=("channel", "x0"),
        )

    @pytest.mark.parametrize("api", GWF_APIS)
    def test_write_gwf_channel_name(self, tmp_path, api):
        """Test that writing GWF when `channel` is set but `name` is not
        uses the `channel` name
        """
        array = self.create(channel="data")
        assert not array.name
        tmp = tmp_path / "test.gwf"
        fmt = "gwf" if api is None else "gwf." + api
        array.write(tmp, format=fmt)
        array2 = type(array).read(tmp, str(array.channel), format="gwf")
        assert array2.name == str(array.channel)
        utils.assert_quantity_sub_equal(
            array,
            array2,
            exclude=("name", "channel"),
        )

    @pytest.mark.parametrize('ext', ('hdf5', 'h5'))
    @pytest.mark.parametrize('channel', [
        None,
        'test',
        'X1:TEST-CHANNEL',
    ])
    def test_read_write_hdf5(self, tmp_path, ext, channel):
        array = self.create()
        array.channel = channel

        tmp = tmp_path / "test.{}".format(ext)
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
        with pytest.raises((IOError, OSError, RuntimeError, ValueError)):
            array.write(tmp, append=True)

        # check reading with start/end works
        start, end = array.span.contract(25)
        t = type(array).read(tmp, start=start, end=end)
        utils.assert_quantity_sub_equal(t, array.crop(start, end))

    def test_read_write_wav(self):
        array = self.create(dtype='float32')
        utils.test_read_write(
            array, 'wav', read_kw={'mmap': True}, write_kw={'scale': 1},
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={'exclude': ['unit', 'name', 'channel', 'x0']})

    @pytest.mark.parametrize("pre, post", [
        pytest.param(None, None, id="none"),
        pytest.param(0, 0, id="zero"),
        pytest.param(None, 1, id="right"),
        pytest.param(1, None, id="left"),
        pytest.param(1, 1, id="both"),
    ])
    def test_read_pad(self, pre, post):
        a = self.TEST_CLASS.read(
            utils.TEST_HDF5_FILE,
            "H1:LDAS-STRAIN",
        )
        start = None if pre is None else a.span[0] - pre
        end = None if post is None else a.span[1] + post
        b = self.TEST_CLASS.read(
            utils.TEST_HDF5_FILE,
            "H1:LDAS-STRAIN",
            pad=0.,
            start=start,
            end=end,
        )
        pres = 0 if not pre else int(pre * a.sample_rate.value)
        posts = 0 if not post else int(post * a.sample_rate.value)
        utils.assert_quantity_sub_equal(
            a.pad(
                (pres, posts),
                mode="constant",
                constant_values=(0,),
            ),
            b,
        )

    def test_read_pad_raise(self):
        """Check that `TimeSeries.read` with `gap='raise'` actually
        raises appropriately.

        [regression: https://github.com/gwpy/gwpy/issues/1211]
        """
        from gwpy.io.cache import file_segment
        span = file_segment(utils.TEST_HDF5_FILE)
        with pytest.raises(ValueError):
            self.TEST_CLASS.read(
                utils.TEST_HDF5_FILE,
                "H1:LDAS-STRAIN",
                pad=0.,
                start=span[0],
                end=span[1]+1.,
                gap="raise",
            )

    @pytest.mark.requires("nds2")
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

    @pytest.mark.parametrize('format', [
        'hdf5',
        pytest.param('gwf', marks=pytest.mark.requires("LDAStools.frameCPP")),
    ])
    @pytest_skip_network_error
    def test_fetch_open_data(self, gw150914, format):
        ts = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            *GWOSC_GW150914_SEGMENT,
            format=format,
            verbose=True,
        )
        utils.assert_quantity_sub_equal(ts, gw150914,
                                        exclude=['name', 'unit', 'channel'])

        # try again with 16384 Hz data
        ts = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            *GWOSC_GW150914_SEGMENT,
            format=format,
            sample_rate=16384,
        )
        assert ts.sample_rate == 16384 * units.Hz

    @pytest_skip_network_error
    def test_fetch_open_data_error(self):
        """Test that TimeSeries.fetch_open_data raises errors it receives
        from the `gwosc` module.
        """
        with pytest.raises(ValueError):
            self.TEST_CLASS.fetch_open_data(
                GWOSC_GW150914_IFO,
                0,
                1,
            )

    @pytest.mark.requires("nds2")
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
            ctx = pytest.warns(UserWarning) if protocol > 1 else nullcontext()
            with ctx:
                ts2 = self.TEST_CLASS.fetch('L1:TEST', *ts.span.protract(10),
                                            pad=-100., host='anything')
            assert ts2.span == ts.span.protract(10)
            assert ts2[0] == -100. * ts.unit
            assert ts2[10] == ts[0]
            assert ts2[-11] == ts[-1]
            assert ts2[-1] == -100. * ts.unit

    @pytest.mark.requires("nds2")
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

    @_gwosc_cvmfs
    @mock.patch.dict(
        "os.environ",
        {"GWDATAFIND_SERVER": GWOSC_DATAFIND_SERVER},
    )
    def test_find(self, gw150914_16384):
        ts = self.TEST_CLASS.find(
            GWOSC_GW150914_CHANNEL,
            *GWOSC_GW150914_SEGMENT,
            frametype=GWOSC_GW150914_FRAMETYPE,
        )
        utils.assert_quantity_sub_equal(ts, gw150914_16384,
                                        exclude=['name', 'channel', 'unit'])

        # test observatory
        ts2 = self.TEST_CLASS.find(
            GWOSC_GW150914_CHANNEL,
            *GWOSC_GW150914_SEGMENT,
            frametype=GWOSC_GW150914_FRAMETYPE,
            observatory=GWOSC_GW150914_IFO[0],
        )
        utils.assert_quantity_sub_equal(ts, ts2)
        with pytest.raises(RuntimeError):
            self.TEST_CLASS.find(
                GWOSC_GW150914_CHANNEL,
                *GWOSC_GW150914_SEGMENT,
                frametype=GWOSC_GW150914_FRAMETYPE,
                observatory='X',
            )

    @_gwosc_cvmfs
    @mock.patch.dict(
        "os.environ",
        {"GWDATAFIND_SERVER": GWOSC_DATAFIND_SERVER},
    )
    def test_find_best_frametype_in_find(self, gw150914_16384):
        ts = self.TEST_CLASS.find(
            GWOSC_GW150914_CHANNEL,
            *GWOSC_GW150914_SEGMENT,
        )
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=['name', 'channel', 'unit'],
        )

    @_gwosc_cvmfs
    @mock.patch.dict(
        # force 'import nds2' to fail so that we are actually testing
        # the gwdatafind API or nothing
        "sys.modules",
        {"nds2": None},
    )
    @mock.patch.dict(
        "os.environ",
        {"GWDATAFIND_SERVER": GWOSC_DATAFIND_SERVER},
    )
    def test_get_datafind(self, gw150914_16384):
        try:
            ts = self.TEST_CLASS.get(
                GWOSC_GW150914_CHANNEL,
                *GWOSC_GW150914_SEGMENT,
                frametype_match=r'V1\Z',
            )
        except (ImportError, RuntimeError) as e:  # pragma: no-cover
            pytest.skip(str(e))
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=['name', 'channel', 'unit'],
        )

    @pytest.mark.requires("nds2")
    @utils.skip_kerberos_credential
    @mock.patch.dict(os.environ)
    def test_get_nds2(self, gw150914_16384):
        # get using NDS2 (if datafind could have been used to start with)
        os.environ.pop('GWDATAFIND_SERVER', None)
        ts = self.TEST_CLASS.get(
            NDS2_GW150914_CHANNEL,
            *GWOSC_GW150914_SEGMENT,
        )
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=['name', 'channel', 'unit'],
        )

    # -- signal processing methods --------------

    def test_fft(self, gw150914):
        fs = gw150914.fft()
        assert isinstance(fs, FrequencySeries)
        assert fs.size == gw150914.size // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == 1 / gw150914.duration
        assert fs.channel is gw150914.channel
        nptest.assert_almost_equal(
            fs.value.max(), 9.793003238789471e-20+3.5377863373683966e-21j)

        # test with nfft arg
        fs = gw150914.fft(nfft=256)
        assert fs.size == 129
        assert fs.dx == gw150914.sample_rate / 256

    def test_average_fft(self, gw150914):
        # test all defaults
        fs = gw150914.average_fft()
        utils.assert_quantity_sub_equal(fs, gw150914.detrend().fft())

        # test fftlength
        fs = gw150914.average_fft(fftlength=0.5)
        assert fs.size == 0.5 * gw150914.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hertz

        fs = gw150914.average_fft(fftlength=0.4, overlap=0.2)

    def test_psd_default_overlap(self, gw150914):
        utils.assert_quantity_sub_equal(
            gw150914.psd(.5, method="median", window="hann"),
            gw150914.psd(.5, .25, method="median", window="hann"),
        )

    @pytest.mark.requires("lal")
    def test_psd_lal_median_mean(self, gw150914):
        # check that warnings and errors get raised in the right place
        # for a median-mean PSD with the wrong data size or parameters

        # single segment should raise error
        with pytest.raises(ValueError), pytest.deprecated_call():
            gw150914.psd(abs(gw150914.span), method='lal_median_mean')

        # odd number of segments should warn
        # pytest hides the second DeprecationWarning that should have been
        # triggered here, for some reason
        with pytest.warns(UserWarning):
            gw150914.psd(1, .5, method='lal_median_mean')

    @pytest.mark.parametrize('method', ('welch', 'bartlett', 'median'))
    def test_psd(self, noisy_sinusoid, method):
        fftlength = .5
        overlap = .25
        fs = noisy_sinusoid.psd(
            fftlength=fftlength,
            overlap=overlap,
            method=method,
        )
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

    def test_asd(self, gw150914):
        kw = {
            "method": "median",
        }
        utils.assert_quantity_sub_equal(
            gw150914.asd(1, **kw),
            gw150914.psd(1, **kw) ** (1/2.),
        )

    def test_csd(self, noisy_sinusoid, corrupt_noisy_sinusoid):
        # test that csd(self) is the same as psd()
        fs = noisy_sinusoid.csd(noisy_sinusoid)
        utils.assert_quantity_sub_equal(
            fs,
            noisy_sinusoid.psd(method="welch"),
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
        'scipy-median',
        pytest.param("lal-welch", marks=pytest.mark.requires("lal")),
        pytest.param("lal-bartlett", marks=pytest.mark.requires("lal")),
        pytest.param("lal-median", marks=pytest.mark.requires("lal")),
        pytest.param("pycbc-welch", marks=pytest.mark.requires("pycbc.psd")),
        pytest.param(
            "pycbc-bartlett",
            marks=pytest.mark.requires("pycbc.psd"),
        ),
        pytest.param("pycbc-median", marks=pytest.mark.requires("pycbc.psd")),
    ])
    @pytest.mark.parametrize(
        'window', (None, 'hann', ('kaiser', 24), 'array'),
    )
    def test_spectrogram(self, gw150914, method, window):
        # generate window for 'array'
        win = self._window_helper(gw150914, 1) if window == 'array' else window

        if method.startswith(("lal", "pycbc")):
            ctx = pytest.deprecated_call
        else:
            ctx = nullcontext

        # generate spectrogram
        with ctx():
            sg = gw150914.spectrogram(1, method=method, window=win)

        # validate
        assert isinstance(sg, Spectrogram)
        assert sg.shape == (abs(gw150914.span),
                            gw150914.sample_rate.value // 2 + 1)
        assert sg.f0 == 0 * units.Hz
        assert sg.df == 1 * units.Hz
        assert sg.channel is gw150914.channel
        assert sg.unit == gw150914.unit ** 2 / units.Hz
        assert sg.epoch == gw150914.epoch
        assert sg.span == gw150914.span

        # check the first time-bin is the same result as .psd()
        n = int(gw150914.sample_rate.value)
        if window == 'hann' and not method.endswith('bartlett'):
            n *= 1.5  # default is 50% overlap
        with ctx():
            psd = gw150914[:int(n)].psd(fftlength=1, method=method, window=win)
        # FIXME: epoch should not be excluded here (probably)
        utils.assert_quantity_sub_equal(sg[0], psd, exclude=['epoch'],
                                        almost_equal=True)

    def test_spectrogram_fftlength(self, gw150914):
        sg = gw150914.spectrogram(1, fftlength=0.5, method="median")
        assert sg.shape == (abs(gw150914.span),
                            0.5 * gw150914.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second

    def test_spectrogram_overlap(self, gw150914):
        kw = {
            "fftlength": 0.5,
            "window": "hann",
            "method": "median",
        }
        sg = gw150914.spectrogram(1, **kw)
        sg2 = gw150914.spectrogram(1, overlap=.25, **kw)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    def test_spectrogram_multiprocessing(self, gw150914):
        kw = {
            "fftlength": 0.5,
            "window": "hann",
            "method": "median",
        }
        sg = gw150914.spectrogram(1, **kw)
        sg2 = gw150914.spectrogram(1, nproc=2, **kw)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    @pytest.mark.parametrize('library', [
        pytest.param('lal', marks=pytest.mark.requires("lal")),
        pytest.param('pycbc', marks=pytest.mark.requires("pycbc.psd")),
    ])
    def test_spectrogram_median_mean(self, gw150914, library):
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
            sg = gw150914.spectrogram(
                1.5,
                fftlength=.5,
                overlap=0,
                method=method,
            )

        assert sg.dt == 1.5 * units.second
        assert sg.df == 2 * units.Hertz

    def test_spectrogram2(self, gw150914):
        # test defaults
        sg = gw150914.spectrogram2(1, overlap=0)
        utils.assert_quantity_sub_equal(
            sg,
            gw150914.spectrogram(
                1,
                fftlength=1,
                overlap=0,
                method='scipy-welch',
                window='hann',
            ),
        )

        # test fftlength
        sg = gw150914.spectrogram2(0.5)
        assert sg.shape == (16, 0.5 * gw150914.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 0.25 * units.second
        # test overlap
        sg = gw150914.spectrogram2(fftlength=0.25, overlap=0.24)
        assert sg.shape == (399, 0.25 * gw150914.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        # note: bizarre stride length because 4096/100 gets rounded
        assert sg.dt == 0.010009765625 * units.second

    def test_fftgram(self, gw150914):
        fgram = gw150914.fftgram(1)
        fs = int(gw150914.sample_rate.value)
        f, t, sxx = signal.spectrogram(
            gw150914, fs,
            window='hann',
            nperseg=fs,
            mode='complex',
        )
        utils.assert_array_equal(gw150914.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram)

        fgram = gw150914.fftgram(1, overlap=0.5)
        f, t, sxx = signal.spectrogram(
            gw150914, fs,
            window='hann',
            nperseg=fs,
            noverlap=fs//2,
            mode='complex',
        )
        utils.assert_array_equal(gw150914.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram)

    def test_spectral_variance(self, gw150914):
        variance = gw150914.spectral_variance(.5, method="median")
        assert isinstance(variance, SpectralVariance)
        assert variance.x0 == 0 * units.Hz
        assert variance.dx == 2 * units.Hz
        assert variance.max() == 8

    def test_rayleigh_spectrum(self, gw150914):
        # assert single FFT creates Rayleigh of 0
        ray = gw150914.rayleigh_spectrum()
        assert isinstance(ray, FrequencySeries)
        assert ray.unit is units.Unit('')
        assert ray.name == 'Rayleigh spectrum of %s' % gw150914.name
        assert ray.epoch == gw150914.epoch
        assert ray.channel is gw150914.channel
        assert ray.f0 == 0 * units.Hz
        assert ray.df == 1 / gw150914.duration
        assert ray.sum().value == 0

        # actually test properly
        ray = gw150914.rayleigh_spectrum(.5)  # no overlap
        assert ray.df == 2 * units.Hz
        nptest.assert_almost_equal(ray.max().value, 2.1239253590490157)
        assert ray.frequencies[ray.argmax()] == 1322 * units.Hz

        ray = gw150914.rayleigh_spectrum(.5, .25)  # 50 % overlap
        nptest.assert_almost_equal(ray.max().value, 1.8814775174483833)
        assert ray.frequencies[ray.argmax()] == 136 * units.Hz

    def test_csd_spectrogram(self, gw150914):
        # test defaults
        sg = gw150914.csd_spectrogram(gw150914, 1)
        assert isinstance(sg, Spectrogram)
        assert sg.shape == (4, gw150914.sample_rate.value // 2 + 1)
        assert sg.f0 == 0 * units.Hz
        assert sg.df == 1 * units.Hz
        assert sg.channel is gw150914.channel
        assert sg.unit == gw150914.unit ** 2 / units.Hertz
        assert sg.epoch == gw150914.epoch
        assert sg.span == gw150914.span

        # check the same result as CSD
        cropped = gw150914[:int(gw150914.sample_rate.value)]
        csd = cropped.csd(cropped)
        utils.assert_quantity_sub_equal(sg[0], csd, exclude=['name', 'epoch'])

        # test fftlength
        sg = gw150914.csd_spectrogram(gw150914, 1, fftlength=0.5)
        assert sg.shape == (4, 0.5 * gw150914.sample_rate.value // 2 + 1)
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second

        # test overlap
        sg = gw150914.csd_spectrogram(
            gw150914,
            0.5,
            fftlength=0.25,
            overlap=0.125,
        )
        assert sg.shape == (8, 0.25 * gw150914.sample_rate.value // 2 + 1)
        assert sg.df == 4 * units.Hertz
        assert sg.dt == 0.5 * units.second

        # test multiprocessing
        sg2 = gw150914.csd_spectrogram(
            gw150914,
            0.5,
            fftlength=0.25,
            overlap=0.125,
            nproc=2,
        )
        utils.assert_quantity_sub_equal(sg, sg2)

    def test_resample(self, gw150914):
        """Test :meth:`gwpy.timeseries.TimeSeries.resample`
        """
        # test IIR decimation
        l2 = gw150914.resample(1024, ftype='iir')
        # FIXME: this test needs to be more robust
        assert l2.sample_rate == 1024 * units.Hz

    def test_resample_noop(self):
        data = self.TEST_CLASS([1, 2, 3, 4, 5])
        with pytest.warns(UserWarning):
            new = data.resample(data.sample_rate)
            assert data is new

    def test_rms(self, gw150914):
        rms = gw150914.rms(1.)
        assert rms.sample_rate == 1 * units.Hz

    @mock.patch('gwpy.segments.DataQualityFlag.query',
                return_value=LIVETIME)
    def test_mask(self, dqflag):
        # craft a timeseries of ones that can be easily tested against
        # a few interesting corner cases
        data = TimeSeries(numpy.ones(8192), sample_rate=128)
        masked = data.mask(flag='X1:TEST-FLAG:1')

        # create objects to test against
        window = planck(128, nleft=64, nright=64)
        times = (data.t0 + numpy.arange(data.size) * data.dt).value
        (live, ) = numpy.nonzero([t in LIVETIME.active for t in times])
        (dead, ) = numpy.nonzero([t not in LIVETIME.active for t in times])

        # verify the mask is correct
        assert data.is_compatible(masked)
        assert live.size + dead.size == data.size
        assert numpy.all(numpy.isfinite(masked.value[live]))
        assert numpy.all(numpy.isnan(masked.value[dead]))
        utils.assert_allclose(masked.value[:4032], numpy.ones(4032))
        utils.assert_allclose(masked.value[4032:4096], window[-64:])
        utils.assert_allclose(masked.value[4352:4416],
                              window[:64] * window[-64:])

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
        whitened = noise.whiten(detrend='linear', method="median")
        assert whitened.size == noise.size
        nptest.assert_almost_equal(whitened.mean().value, 0.0, decimal=2)
        nptest.assert_almost_equal(whitened.std().value, 1.0, decimal=2)

        # when a loud signal is present, the max amplitude should be recovered
        # at the time of that signal
        tmax = data.times[data.argmax()]
        assert not numpy.isclose(tmax.value, glitchtime)

        whitened = data.whiten(detrend='linear', method="median")
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
            numpy.random.normal(size=16384 * 64),
            sample_rate=16384,
            epoch=-32,
        ).zpk([], [1], 1)
        glitchtime = -16.5
        glitch = self.TEST_CLASS(
            signal.gausspulse(numpy.arange(-1, 1, 1./16384), bw=100),
            sample_rate=16384,
            epoch=glitchtime-1,
        )

        # check that, without a signal present, we only see background
        snr = noise.correlate(glitch, whiten=True, method="median")
        tmax = snr.times[snr.argmax()]
        assert snr.size == noise.size
        assert not numpy.isclose(tmax.value, glitchtime)
        nptest.assert_almost_equal(snr.mean().value, 0.0, decimal=1)
        nptest.assert_almost_equal(snr.std().value, 1.0, decimal=1)

        # inject and recover the glitch
        data = noise.inject(glitch * 1e-4)
        snr = data.correlate(glitch, whiten=True, method="median")
        tmax = snr.times[snr.argmax()]
        nptest.assert_almost_equal(tmax.value, glitchtime)

    def test_detrend(self, gw150914):
        assert not numpy.isclose(gw150914.value.mean(), 0.0, atol=1e-21)
        detrended = gw150914.detrend()
        assert numpy.isclose(detrended.value.mean(), 0.0)

    def test_filter(self, gw150914):
        zpk = [], [], 1
        fts = gw150914.filter(zpk, analog=True)
        utils.assert_quantity_sub_equal(gw150914, fts)

        # check SOS filters can be used directly
        zpk = filter_design.highpass(50, sample_rate=gw150914.sample_rate)
        sos = signal.zpk2sos(*zpk)
        utils.assert_quantity_almost_equal(
            gw150914.filter(zpk),
            gw150914.filter(sos),
        )

    def test_zpk(self, gw150914):
        zpk = [10, 10], [1, 1], 100
        utils.assert_quantity_sub_equal(
            gw150914.zpk(*zpk), gw150914.filter(*zpk, analog=True))

    def test_notch(self, gw150914):
        # test notch runs end-to-end
        gw150914.notch(60)

        # test breaks when you try and 'fir' notch
        with pytest.raises(NotImplementedError):
            gw150914.notch(10, type='fir')

    def test_q_gram(self, gw150914):
        # test simple q-transform
        qgram = gw150914.q_gram()
        assert isinstance(qgram, EventTable)
        assert qgram.meta['q'] == 45.25483399593904
        assert qgram['energy'].min() >= 5.5**2 / 2
        nptest.assert_almost_equal(qgram['energy'].max(), 10559.25, decimal=2)

    def test_q_transform(self, gw150914):
        # test simple q-transform
        qspecgram = gw150914.q_transform(method='scipy-welch', fftlength=2)
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (1000, 2403)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 155.93567, decimal=5)

        # test whitening args
        asd = gw150914.asd(2, 1, method='scipy-welch')
        qsg2 = gw150914.q_transform(method='scipy-welch', whiten=asd)
        utils.assert_quantity_sub_equal(qspecgram, qsg2, almost_equal=True)

        asd = gw150914.asd(.5, .25, method='scipy-welch')
        qsg2 = gw150914.q_transform(method='scipy-welch', whiten=asd)
        qsg3 = gw150914.q_transform(
            method='scipy-welch',
            fftlength=.5,
            overlap=.25,
        )
        utils.assert_quantity_sub_equal(qsg2, qsg3, almost_equal=True)

        # make sure frequency too high presents warning
        with pytest.warns(UserWarning):
            qspecgram = gw150914.q_transform(
                method='scipy-welch',
                frange=(0, 10000),
            )
            nptest.assert_almost_equal(
                qspecgram.yspan[1],
                1291.5316,
                decimal=4,
            )

        # test other normalisations work (or don't)
        q2 = gw150914.q_transform(method='scipy-welch', norm='median')
        utils.assert_quantity_sub_equal(qspecgram, q2, almost_equal=True)
        gw150914.q_transform(method='scipy-welch', norm='mean')
        gw150914.q_transform(method='scipy-welch', norm=False)
        with pytest.raises(ValueError):
            gw150914.q_transform(method='scipy-welch', norm='blah')

    def test_q_transform_logf(self, gw150914):
        # test q-transform with log frequency spacing
        qspecgram = gw150914.q_transform(
            method='scipy-welch',
            fftlength=2,
            logf=True,
        )
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (1000, 500)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 155.93774, decimal=5)

    def test_q_transform_nan(self):
        data = TimeSeries(numpy.empty(256*10) * numpy.nan, sample_rate=256)
        with pytest.raises(ValueError) as exc:
            data.q_transform(method="median")
        assert str(exc.value) == 'Input signal contains non-numerical values'

    def test_boolean_statetimeseries(self, array):
        comp = array >= 2 * array.unit
        assert isinstance(comp, StateTimeSeries)
        assert comp.unit is units.Unit('')
        assert comp.name == '%s >= 2.0' % (array.name)
        assert (array == array).name == '{0} == {0}'.format(array.name)

    @pytest_skip_network_error
    def test_transfer_function(self):
        tsh = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
        tsl = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
        tf = tsh.transfer_function(tsl, fftlength=1.0, overlap=0.5)
        assert tf.df == 1 * units.Hz
        assert tf.frequencies[abs(tf).argmax()] == 516 * units.Hz

    @pytest_skip_network_error
    def test_coherence(self):
        tsh = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
        tsl = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
        coh = tsh.coherence(tsl, fftlength=1.0)
        assert coh.df == 1 * units.Hz
        assert coh.frequencies[coh.argmax()] == 60 * units.Hz

    @pytest_skip_network_error
    def test_coherence_spectrogram(self):
        tsh = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
        tsl = TimeSeries.fetch_open_data('L1', 1126259446, 1126259478)
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

    @pytest.mark.requires("framel")
    def test_read_write_gwf(self, instance, tmp_path):
        tmp = tmp_path / "test.gwf"
        instance.write(tmp)
        new = self.TEST_CLASS.read(tmp, instance.keys())
        for key in new:
            utils.assert_quantity_sub_equal(new[key], instance[key],
                                            exclude=['channel'])

    def test_read_write_hdf5(self, instance, tmp_path):
        tmp = tmp_path / "test.h5"
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
