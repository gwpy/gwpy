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

import logging
import sys
from contextlib import nullcontext
from itertools import (
    chain,
    product,
)
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    TypeVar,
)
from unittest import mock

import numpy
import pytest
from astropy import units
from numpy import testing as nptest
from requests.exceptions import HTTPError
from scipy import signal
from scipy.signal.windows import tukey

from ...frequencyseries import FrequencySeries, SpectralVariance
from ...io.gwf import get_backend as get_gwf_backend
from ...segments import DataQualityFlag, Segment, SegmentList
from ...signal import filter_design
from ...spectrogram import Spectrogram
from ...table import EventTable
from ...testing import mocks, utils
from ...testing.errors import (
    pytest_skip_flaky_network,
    pytest_skip_network_error,
)
from ...time import LIGOTimeGPS
from ...types import Index
from .. import StateTimeSeries, TimeSeries, TimeSeriesDict, TimeSeriesList
from ..io import gwf as tsio_gwf
from .test_core import (
    TestTimeSeriesBase as _TestTimeSeriesBase,
    TestTimeSeriesBaseDict as _TestTimeSeriesBaseDict,
    TestTimeSeriesBaseList as _TestTimeSeriesBaseList,
)

if TYPE_CHECKING:
    from typing import ClassVar

TimeSeriesType = TypeVar("TimeSeriesType", bound=TimeSeries)

try:
    get_gwf_backend(
        "gwpy.timeseries.io.gwf",
        backends=tsio_gwf.BACKENDS,
    )
except ImportError:
    HAVE_GWF_BACKEND = False
else:
    HAVE_GWF_BACKEND = True

RNG = numpy.random.default_rng(0)

# remote URL for test .gwf file
TEST_HDF5_URL = (
    "https://gitlab.com/gwpy/gwpy/-/raw/v3.0.10/"
    + Path(utils.TEST_HDF5_FILE).relative_to(
        utils.TEST_DATA_PATH.parent.parent.parent,
    ).as_posix()
)

SKIP_GWF_BACKEND = pytest.mark.skipif(
    not HAVE_GWF_BACKEND,
    reason="no GWF backend",
)
GWF_BACKENDS = [
    pytest.param(None, marks=SKIP_GWF_BACKEND),
    pytest.param("lalframe", marks=pytest.mark.requires("lalframe")),
    pytest.param("framecpp", marks=pytest.mark.requires("LDAStools.frameCPP")),
    pytest.param("framel", marks=pytest.mark.requires("framel")),
]

LIVETIME = DataQualityFlag(
    name="X1:TEST-FLAG:1",
    active=SegmentList([
        Segment(0, 32),
        Segment(34, 34.5),
    ]),
    known=SegmentList([Segment(0, 64)]),
    isgood=True,
)

GWOSC_DATAFIND_SERVER = "datafind.gwosc.org"

# Description of GW150914 for tests
# This is used in various tests
GWOSC_GW150914_IFO = "L1"
GWOSC_GW150914_CHANNEL = "L1:GWOSC-16KHZ_R1_STRAIN"
NDS2_GW150914_CHANNEL = "L1:DCS-CALIB_STRAIN_C02"
GWOSC_GW150914_FRAMETYPE = "L1_LOSC_16_V1"
GWOSC_GW150914 = 1126259462
GWOSC_GW150914_SEGMENT: Segment[float] = Segment(GWOSC_GW150914 - 2, GWOSC_GW150914 + 2)
GWOSC_GW150914_SEGMENT_32: Segment[float] = Segment(
    GWOSC_GW150914 - 16,
    GWOSC_GW150914 + 16,
)
GWOSC_GW150914_DQ_NAME = {
    "hdf5": "L1:DQmask",
    "gwf": "L1:GWOSC-4KHZ_R1_DQMASK",
}
GWOSC_GW150914_DQ_BITS = {
    "hdf5": [
        "Passes DATA test",
        "Passes CBC_CAT1 test",
        "Passes CBC_CAT2 test",
        "Passes CBC_CAT3 test",
        "Passes BURST_CAT1 test",
        "Passes BURST_CAT2 test",
        "Passes BURST_CAT3 test",
    ],
    "gwf": [
        "DATA",
        "CBC_CAT1",
        "CBC_CAT2",
        "CBC_CAT3",
        "BURST_CAT1",
        "BURST_CAT2",
        "BURST_CAT3",
    ],
}
GWOSC_GW150914_INJ_NAME = {
    "hdf5": "L1:Injmask",
    "gwf": "L1:GWOSC-4KHZ_R1_INJMASK",
}
GWOSC_GW150914_INJ_BITS = {
    "hdf5": [
        "Passes NO_CBC_HW_INJ test",
        "Passes NO_BURST_HW_INJ test",
        "Passes NO_DETCHAR_HW_INJ test",
        "Passes NO_CW_HW_INJ test",
        "Passes NO_STOCH_HW_INJ test",
    ],
    "gwf": [
        "NO_CBC_HW_INJ",
        "NO_BURST_HW_INJ",
        "NO_DETCHAR_HW_INJ",
        "NO_CW_HW_INJ",
        "NO_STOCH_HW_INJ",
    ],
}

# Description of GW190814
# This is used in various tests in particular to test GWOSC new file format
# (but not for NDS and DataFind)
GWOSC_GW190814_IFO = "L1"
GWOSC_GW190814_CHANNEL = "L1:GWOSC-16KHZ_R1_STRAIN"
GWOSC_GW190814 = 1249852257
GWOSC_GW190814_SEGMENT: Segment[float] = Segment(GWOSC_GW190814 - 2, GWOSC_GW190814 + 2)
GWOSC_GW190814_SEGMENT_32: Segment[float] = Segment(
    GWOSC_GW190814 - 16,
    GWOSC_GW190814 + 16,
)
GWOSC_GW190814_DQ_NAME =  {
    "hdf5": "L1:GWOSC-4KHZ_R1_DQMASK",
    "gwf": "L1:GWOSC-4KHZ_R1_DQMASK",
}
GWOSC_GW190814_DQ_BITS = GWOSC_GW150914_DQ_BITS
GWOSC_GW190814_INJ_NAME = {
    "hdf5": "L1:GWOSC-4KHZ_R1_INJMASK",
    "gwf": "L1:GWOSC-4KHZ_R1_INJMASK",
}
GWOSC_GW190814_INJ_BITS = GWOSC_GW150914_INJ_BITS


__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _gwosc_pelican(func):
    """Decorate ``func`` with all necessary GWOSC/Pelican-related decorators."""
    for dec in (
        pytest.mark.requires("lalframe"),
        pytest.mark.requires("requests_pelican"),
        pytest_skip_flaky_network,
        mock.patch.dict(
            "os.environ",
            {"GWDATAFIND_SERVER": GWOSC_DATAFIND_SERVER},
        ),
    ):
        func = dec(func)
    return func


class TestTimeSeries(_TestTimeSeriesBase[TimeSeriesType]):
    """Test `TimeSeries`."""

    TEST_CLASS: type[TimeSeriesType] = TimeSeries

    # -- fixtures --------------------

    @pytest.fixture(scope="class")
    @pytest_skip_network_error
    def gw150914(self) -> TimeSeriesType:
        """TimeSeries containing GW150914 data from GWOSC at 4096 Hz."""
        return self.TEST_CLASS.get(
            GWOSC_GW150914_IFO,
            *GWOSC_GW150914_SEGMENT,
            sample_rate=4096,
        )

    @pytest.fixture(scope="class")
    @pytest_skip_network_error
    def gw150914_16384(self) -> TimeSeriesType:
        """TimeSeries containing GW150914 data from GWOSC at 16384 Hz."""
        return self.TEST_CLASS.get(
            GWOSC_GW150914_IFO,
            GWOSC_GW150914_SEGMENT.start,
            GWOSC_GW150914_SEGMENT.end,
            sample_rate=16384,
        )

    @pytest.fixture(scope="class")
    @pytest_skip_network_error
    def gw150914_h1_32(self) -> TimeSeriesType:
        """TimeSeries containing 32-seconds of H1 GW150914 data from GWOSC."""
        return self.TEST_CLASS.get(
            "H1",
            GWOSC_GW150914_SEGMENT_32.start,
            GWOSC_GW150914_SEGMENT_32.end,
            source="gwosc",
            sample_rate=4096,
        )

    @pytest.fixture(scope="class")
    @pytest_skip_network_error
    def gw150914_l1_32(self) -> TimeSeriesType:
        """TimeSeries containing 32-seconds of L1 GW150914 data from GWOSC."""
        return self.TEST_CLASS.get(
            "L1",
            GWOSC_GW150914_SEGMENT_32.start,
            GWOSC_GW150914_SEGMENT_32.end,
            source="gwosc",
            sample_rate=4096,
        )

    # -- test class functionality ----

    def test_ligotimegps(self):
        """Test that TimeSeries.t0 and x0 can be set with LIGOTimeGPS."""
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

    def test_epoch(self):  # type: ignore[override]
        """Test `TimeSeries.epoch`."""
        array = self.create()
        assert array.epoch.gps == array.x0.value  # type: ignore[union-attr]

    # -- test I/O --------------------

    def test_read_cache(self, tmp_path):
        """Test that `TimeSeries.read` handles caches well."""
        # create an array and write it to a file
        array = self.create(t0=0, dt=10/self.data.size, name="TEST1")
        file1 = tmp_path / "X-data-0-10.h5"
        array.write(file1)

        # create a second array, but don't write it
        file2 = tmp_path / "X-data-10-10.h5"

        # create a cache that talks about both files
        cache = [file1, file2]

        # check that when reading _and_ asking for times covered by the
        # first file, the cache is appropriately filtered
        data = self.TEST_CLASS.read(
            cache,
            "TEST1",
            format="hdf5",
            start=0,
            end=10,
        )
        utils.assert_quantity_sub_equal(array, data)

    def test_read_start_end_args(self, tmp_path):
        """Test that `TimeSeries.read` handles start/end as positionals."""
        # Create an array and write it to a file
        array = self.create(t0=0, dt=10/self.data.size, name="TEST1")
        h5file = tmp_path / "X-data-0-10.h5"
        array.write(h5file)

        # Try and read (half) the file passing start/end as positionals
        data = self.TEST_CLASS.read(
            h5file,
            "TEST1",
            0,
            5,
            format="hdf5",
        )
        half = array.size // 2
        utils.assert_quantity_sub_equal(array[:half], data)

    @pytest.mark.parametrize("fmt", ["txt", "csv"])
    def test_read_write_ascii(self, array, fmt):
        """Test reading and writing ASCII files."""
        utils.test_read_write(
            array,
            fmt,
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={"exclude": ["name", "channel", "unit"]},
        )

    def test_read_ascii_header(self, tmpdir):
        """Check that ASCII files with headers are read without extra options.

        [regression: https://gitlab.com/gwpy/gwpy/-/issues/1473]
        """
        txt = tmpdir / "text.txt"
        txt.write_text(
            "# time (s)\tdata (strain)\n0\t1\n1\t2\n2\t3",
            encoding="utf-8",
        )
        data = self.TEST_CLASS.read(txt, format="txt")
        utils.assert_array_equal(data.times, Index((0, 1, 2), unit="s"))
        utils.assert_array_equal(data.value, (1, 2, 3))

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_read_write_gwf(self, tmp_path, backend):
        """Test reading and writing GWF files."""
        array = self.create(name="TEST")

        # test basic write/read
        try:
            utils.test_read_write(
                array,
                "gwf",
                extension="gwf",
                read_args=[array.name],
                read_kw={"backend": backend},
                write_kw={"backend": backend},
                assert_equal=utils.assert_quantity_sub_equal,
                assert_kw={"exclude": ["channel"]},
            )
        except ImportError as e:  # pragma: no-cover
            pytest.skip(str(e))

        # test read keyword arguments
        tmp = tmp_path / "test.gwf"
        array.write(tmp, format="gwf", backend=backend)

        def read_(**kwargs):
            return type(array).read(
                tmp,
                array.name,
                format="gwf",
                backend=backend,
                **kwargs,
            )

        # test start and end
        start, end = array.span.contract(10)
        t = read_(start=start, end=end)
        utils.assert_quantity_sub_equal(
            t, array.crop(start, end),
            exclude=["channel"],
        )
        assert t.span == (start, end)

        # test start only
        t = read_(start=start)
        utils.assert_quantity_sub_equal(
            t,
            array.crop(start=start),
            exclude=["channel"],
        )

        # test end only
        t = read_(end=end)
        utils.assert_quantity_sub_equal(
            t,
            array.crop(end=end),
            exclude=["channel"],
        )

    @SKIP_GWF_BACKEND
    def test_write_gwf_overwrite(self, tmp_path):
        """Test that writing GWF with overwrite works."""
        tmp = tmp_path / "test.gwf"

        # Create a file (that mustn't already exist)
        array = self.create(name="TEST")
        array.write(tmp, format="gwf", overwrite=False)

        # check that we can't write again without overwrite
        with pytest.raises(
            OSError,
            match="File exists",
        ):
            array.write(tmp, format="gwf", overwrite=False)

        # check that overwrite=True is default and works
        array.write(tmp, format="gwf")

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_read_gwf_end_error(self, backend):
        """Test that reading past the end of available data fails."""
        with pytest.raises(
            ValueError,
            match=r"(cannot read data|failed to read)",
        ):
            self.TEST_CLASS.read(
                utils.TEST_GWF_FILE,
                "L1:LDAS-STRAIN",
                format="gwf",
                backend=backend,
                start=utils.TEST_GWF_SPAN[1],
            )

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_read_gwf_negative_duration_error(self, backend):
        """Test that reading a negative duration fails."""
        with pytest.raises(
            ValueError,
            match=r"(cannot read data|failed to read)",
        ):
            self.TEST_CLASS.read(
                utils.TEST_GWF_FILE,
                "L1:LDAS-STRAIN",
                format="gwf",
                backend=backend,
                end=utils.TEST_GWF_SPAN[0]-1,
            )

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    @pytest.mark.parametrize("parallel", [1, 2])
    def test_read_write_gwf_multiple(self, tmp_path, backend, parallel):
        """Check that each GWF backend can read a series of files.

        Regression: https://gitlab.com/gwpy/gwpy/-/issues/1486
        """
        a1 = self.create(name="TEST")
        a2 = self.create(name="TEST", t0=a1.span[1], dt=a1.dx)

        tmp1 = tmp_path / "test1.gwf"
        tmp2 = tmp_path / "test2.gwf"
        a1.write(tmp1, format="gwf", backend=backend)
        a2.write(tmp2, format="gwf", backend=backend)
        cache = [tmp1, tmp2]

        comb = self.TEST_CLASS.read(
            cache,
            "TEST",
            start=a1.span[0],
            end=a2.span[1],
            format="gwf",
            backend=backend,
            parallel=parallel,
        )
        utils.assert_quantity_sub_equal(
            comb,
            a1.append(a2, inplace=False),
            exclude=["channel"],
        )

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_read_write_gwf_name_error(self, tmp_path, backend, gw150914):
        """Test that reading GWF handles missing/wrong channel names."""
        tmp = tmp_path / "test.gwf"
        gw150914.write(tmp, format="gwf", backend=backend)

        # wrong channel (framel on windows gives a slightly different error)
        with pytest.raises(
            ValueError,
            match=r"({})".format("|".join((  # noqa: FLY002
                "^channel 'another channel' not found",
                "vector not found: another channel",
            ))),
        ):
            self.TEST_CLASS.read(
                tmp,
                "another channel",
                format="gwf",
                backend=backend,
            )

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_read_write_gwf_interval_error(self, tmp_path, backend, gw150914):
        """Test that reading GWF handles out-of-bounds times."""
        tmp = tmp_path / "test.gwf"
        gw150914.write(tmp, format="gwf", backend=backend)

        # wrong times (error message is different for each backend)
        with pytest.raises(
            ValueError,
            match=r"(cannot read data|failed to read)",
        ):
            self.TEST_CLASS.read(
                tmp,
                gw150914.name,
                start=gw150914.span[0]-1,
                end=gw150914.span[0],
                format="gwf",
                backend=backend,
            )

    @pytest.mark.requires("lalframe")  # for get_channel_type
    @pytest.mark.parametrize("ctype", ["adc", "proc", "sim", None])
    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_write_gwf_type(self, gw150914, tmp_path, backend, ctype):
        """Test writing GWF for each FrData type."""
        from ...io.gwf import get_channel_type

        expected_ctype = ctype or "proc"

        # write the file
        tmp = tmp_path / "test.gwf"
        gw150914.write(tmp, type=ctype, format="gwf", backend=backend)

        # assert that the type is correct
        assert get_channel_type(gw150914.name, tmp) == expected_ctype

        # read the file and check that the data match
        new = type(gw150914).read(
            tmp,
            gw150914.name,
            format="gwf",
            backend=backend,
        )

        # epoch seems to mismatch at O(1e-12), which is unfortunate
        utils.assert_quantity_sub_equal(
            gw150914,
            new,
            exclude=("channel", "x0"),
        )

    @pytest.mark.parametrize("backend", GWF_BACKENDS)
    def test_write_gwf_channel_name(self, tmp_path, backend):
        """Test that writing GWF uses `channel` when `name` is missing."""
        # create data
        array = self.create(channel="data")
        assert not array.name

        # write data
        tmp = tmp_path / "test.gwf"
        array.write(tmp, format="gwf", backend=backend)

        # read it back check
        array2 = type(array).read(tmp, str(array.channel), format="gwf")
        assert array2.name == str(array.channel)
        utils.assert_quantity_sub_equal(
            array,
            array2,
            exclude=("name", "channel"),
        )

    @pytest.mark.parametrize("ext", ["hdf5", "h5"])
    @pytest.mark.parametrize("channel", [
        None,
        "test",
        "X1:TEST-CHANNEL",
    ])
    def test_read_write_hdf5(self, tmp_path, ext, channel):
        """Test reading and writing HDF5 files."""
        array = self.create()
        array.channel = channel

        tmp = tmp_path / f"test.{ext}"
        # check array with no name fails
        with pytest.raises(
            ValueError,
            match=r"^Cannot determine HDF5 path",
        ):
            array.write(tmp, overwrite=True)
        array.name = "TEST"

        # write array (with auto-identify)
        array.write(tmp, overwrite=True)

        # check reading gives the same data (with/without auto-identify)
        ts = type(array).read(tmp, format="hdf5")
        utils.assert_quantity_sub_equal(array, ts)
        ts = type(array).read(tmp)
        utils.assert_quantity_sub_equal(array, ts)

        # check that we can't then write the same data again
        with pytest.raises(
            OSError,
            match="File exists",
        ):
            array.write(tmp)
        with pytest.raises((IOError, OSError, RuntimeError, ValueError)):
            array.write(tmp, append=True)

        # check reading with start/end works
        start, end = array.span.contract(25)
        t = type(array).read(tmp, start=start, end=end)
        utils.assert_quantity_sub_equal(t, array.crop(start, end))

    def test_read_write_wav(self):
        """Test reading and writing WAV files."""
        array = self.create(dtype="float32")
        utils.test_read_write(
            array, "wav", read_kw={"mmap": True}, write_kw={"scale": 1},
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={"exclude": ["unit", "name", "channel", "x0"]})

    @pytest.mark.parametrize(("pre", "post"), [
        pytest.param(None, None, id="none"),
        pytest.param(0, 0, id="zero"),
        pytest.param(None, 1, id="right"),
        pytest.param(1, None, id="left"),
        pytest.param(1, 1, id="both"),
    ])
    def test_read_pad(self, pre, post):
        """Check that `TimeSeries.read` with `pad` works as expected."""
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
        """Check that `TimeSeries.read` with `gap='raise'` raises appropriately.

        [regression: https://gitlab.com/gwpy/gwpy/-/issues/1211]
        """
        from gwpy.io.cache import file_segment
        span = file_segment(utils.TEST_HDF5_FILE)
        with pytest.raises(
            ValueError,
            match="does not cover requested interval",
        ):
            self.TEST_CLASS.read(
                utils.TEST_HDF5_FILE,
                "H1:LDAS-STRAIN",
                pad=0.,
                start=span[0],
                end=span[1]+1.,
                gap="raise",
            )

    @pytest_skip_flaky_network
    def test_read_remote(self):
        """Test that reading directly from a remote URI works."""
        local = self.TEST_CLASS.read(
            utils.TEST_HDF5_FILE,
            "H1:LDAS-STRAIN",
        )
        remote = self.TEST_CLASS.read(
            TEST_HDF5_URL,
            "H1:LDAS-STRAIN",
            cache=False,
        )
        utils.assert_quantity_sub_equal(local, remote)

    @pytest.mark.requires("nds2")
    def test_from_nds2_buffer_dynamic_scaled(self):
        """Test `TimeSeries.from_nds2_buffer` with scaled data."""
        # build fake buffer for LIGO channel
        nds_buffer = mocks.nds2_buffer(
            "H1:TEST",
            self.data,
            1000000000,
            self.data.shape[0],
            "m",
            name="test",
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

    # -- test remote data access -----

    @pytest.mark.parametrize("fmt", [
        "hdf5",
        pytest.param("gwf", marks=pytest.mark.requires("lalframe")),
    ])
    @pytest_skip_flaky_network
    def test_fetch_open_data(self, gw150914, fmt):
        """Test `TimeSeries.fetch_open_data`."""
        ts = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            GWOSC_GW150914_SEGMENT.start,
            GWOSC_GW150914_SEGMENT.end,
            format=fmt,
            timeout=60,
        )
        utils.assert_quantity_sub_equal(
            ts,
            gw150914,
            exclude=["name", "unit", "channel"],
        )

        # try again with 16384 Hz data
        ts = self.TEST_CLASS.fetch_open_data(
            GWOSC_GW150914_IFO,
            GWOSC_GW150914_SEGMENT.start,
            GWOSC_GW150914_SEGMENT.end,
            format=fmt,
            sample_rate=16384,
            timeout=60,
        )
        assert ts.sample_rate == 16384 * units.Hz

    @pytest_skip_flaky_network
    def test_fetch_open_data_error(self):
        """Test that TimeSeries.fetch_open_data propagates errors."""
        with pytest.RaisesGroup(
            pytest.RaisesExc(ValueError, match="Cannot find a GWOSC dataset"),
            match="failed to get data",
        ):
            self.TEST_CLASS.fetch_open_data(
                GWOSC_GW150914_IFO,
                0,
                1,
            )

    @pytest.mark.requires("nds2")
    @pytest.mark.parametrize("protocol", [
        pytest.param(1, id="nds1"),
        pytest.param(2, id="nds2"),
    ])
    def test_fetch(self, nds2_connection, protocol):
        """Test `TimeSeries.fetch`."""
        # set protocol
        nds2_connection.get_protocol.return_value = protocol

        # get expected result
        expected = self.TEST_CLASS.from_nds2_buffer(nds2_connection._buffers[0])

        # execute fetch()
        ts = self.TEST_CLASS.fetch(
            "X1:test",
            1000000000,
            1000000004,
        )
        utils.assert_quantity_sub_equal(ts, expected)

    @pytest.mark.requires("nds2")
    def test_fetch_connection(self, nds2_connection):
        """Test `TimeSeries.fetch(..., connection=<>)`."""
        expected = self.TEST_CLASS.from_nds2_buffer(nds2_connection._buffers[0])
        ts = self.TEST_CLASS.fetch(
            "X1:test",
            1000000000,
            1000000004,
            connection=nds2_connection,
        )
        utils.assert_quantity_sub_equal(ts, expected)

    @pytest.mark.requires("nds2")
    @pytest.mark.parametrize("protocol", [
        pytest.param(1, id="nds1"),
        pytest.param(2, id="nds2"),
    ])
    def test_fetch_pad(self, caplog, nds2_connection, protocol):
        """Test `TimeSeries.fetch(..., pad=...)`."""
        # set protocol
        nds2_connection.get_protocol.return_value = protocol

        # get expected result
        expected = self.TEST_CLASS.from_nds2_buffer(nds2_connection._buffers[0])

        with caplog.at_level(logging.DEBUG, logger="gwpy.timeseries.io.nds2"):
            # check padding works
            ts = self.TEST_CLASS.fetch(
                "X1:test",
                *expected.span.protract(2),
                pad=-100.,
                connection=nds2_connection,
            )

        assert ts.span == expected.span.protract(2)
        nptest.assert_array_equal(
            ts.value,
            numpy.concatenate((
                numpy.ones(int(2 * ts.sample_rate.value)) * -100.,
                expected.value,
                numpy.ones(int(2 * ts.sample_rate.value)) * -100.,
            )),
        )

        # check that the logger emitted warnings about the padding
        if protocol == 2 and sys.version_info >= (3, 12):
            for msg in (
                "[nds.test.gwpy] Availability check complete, "
                "found 1 viable segments of data with 66.67% coverage",
                "[nds.test.gwpy] Gaps will be padded with -100.0",
            ):
                assert (
                    "gwpy.timeseries.io.nds2",
                    logging.DEBUG,
                    msg,
                ) in caplog.record_tuples

    @pytest.mark.requires("nds2")
    def test_fetch_empty_iterate_error(self, nds2_connection):
        """Test `TimeSeries.fetch()` handling of no data."""
        # patch find_channels() to return the channel, even though
        # iterate() won't return any data
        def find_channels(name, *args, **kwargs):  # noqa: ARG001
            return [mocks.nds2_channel(name, 128, "")]

        nds2_connection.find_channels = find_channels
        nds2_connection._buffers = []

        # check that we get the right error
        with pytest.RaisesGroup(
            pytest.RaisesExc(RuntimeError, match="no data received from nds.test.gwpy"),
            match="failed to get data",
        ):
            self.TEST_CLASS.fetch("X1:test-missing", 0, 1, host="nds.gwpy")

    @_gwosc_pelican
    @pytest.mark.parametrize("kwargs", [
        pytest.param({"verbose": True}, id="default"),
        pytest.param({"observatory": GWOSC_GW150914_IFO[0]}, id="observatory"),
    ])
    def test_find(self, gw150914_16384, kwargs):
        """Test that `TimeSeries.find()` can actually find data."""
        ts = self.TEST_CLASS.find(
            GWOSC_GW150914_CHANNEL,
            *GWOSC_GW150914_SEGMENT,
            frametype=GWOSC_GW150914_FRAMETYPE,
            urltype="osdf",
            host="datafind.gwosc.org",
            **kwargs,
        )
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=["name", "channel", "unit"],
        )

    @_gwosc_pelican
    def test_find_datafind_httperror(self):
        """Test that HTTPErrors are presented in `find()`."""
        # Use an invalid observatory to trigger a 400 error from datafind
        with pytest.raises(ExceptionGroup, match="failed to get data") as excinfo:
            self.TEST_CLASS.find(
                GWOSC_GW150914_CHANNEL,
                *GWOSC_GW150914_SEGMENT,
                frametype=GWOSC_GW150914_FRAMETYPE,
                observatory="X",
                host="datafind.gwosc.org",
            )
        # Check that the 400 error is part of the exception group
        assert excinfo.group_contains(HTTPError, match="400 Client Error")

    @mock.patch.dict(
        "os.environ",
        {"GWDATAFIND_SERVER": GWOSC_DATAFIND_SERVER},
    )
    def test_find_datafind_runtimeerror(self):
        """Test that empty datafind caches result in RuntimeErrors in `find()`."""
        # Use a time where no data is available to trigger a RuntimeError
        with pytest.raises(ExceptionGroup, match="failed to get data") as excinfo:
            self.TEST_CLASS.find(
                GWOSC_GW150914_CHANNEL,
                *GWOSC_GW150914_SEGMENT.shift(-1e8),
                frametype=GWOSC_GW150914_FRAMETYPE,
                host="datafind.gwosc.org",
            )
        # Check that the RuntimeError is part of the exception group
        assert excinfo.group_contains(RuntimeError)

    def test_find_observatory_error(self):
        """Test that `find()` raises ValueError on inconsistent observatory."""
        with pytest.raises(ExceptionGroup, match="failed to get data") as excinfo:
            self.TEST_CLASS.find(
                "Test",
                0,
                1,
                frametype="X1_TEST",
            )
        assert excinfo.group_contains(
            ValueError,
            match="Cannot parse list of IFOs from channel names",
        )

    @_gwosc_pelican
    def test_find_best_frametype_in_find(self, gw150914_16384):
        """Test that `TimeSeries.find()` best frametype selection works."""
        ts = self.TEST_CLASS.find(
            GWOSC_GW150914_CHANNEL,
            GWOSC_GW150914_SEGMENT.start,
            GWOSC_GW150914_SEGMENT.end,
            urltype="osdf",
            host="datafind.gwosc.org",
        )
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=["name", "channel", "unit"],
        )

    @_gwosc_pelican
    def test_get_datafind(self, gw150914_16384):
        """Test that `TimeSeries.get(..., source='datafind')` works."""
        ts = self.TEST_CLASS.get(
            GWOSC_GW150914_CHANNEL,
            GWOSC_GW150914_SEGMENT.start,
            GWOSC_GW150914_SEGMENT.end,
            source="gwdatafind",
            host="datafind.gwosc.org",
            frametype_match=r"V1\Z",
            urltype="osdf",
        )
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=["name", "channel", "unit"],
        )

    @pytest_skip_flaky_network
    def test_get_gwosc_kwargs(self, gw150914):
        """Test that `TimeSeries.get(..., frametype="X")` doesn't break GWOSC.

        GWDataFind should be tried first as a source, but if the frametype doesn't
        match anything, it should fall back to the GWOSC API without falling over.
        """
        try:
            ts = self.TEST_CLASS.get(
                GWOSC_GW150914_IFO,
                GWOSC_GW150914_SEGMENT.start,
                GWOSC_GW150914_SEGMENT.end,
                frametype="WONT_MATCH",
                sample_rate=4096,
            )
        except* ImportError as e:  # pragma: no-cover
            pytest.skip(str(e))
        utils.assert_quantity_sub_equal(
            ts,
            gw150914,
            exclude=["name", "channel", "unit"],
        )

    @pytest_skip_flaky_network
    @pytest.mark.requires("nds2")
    @utils.skip_kerberos_credential
    def test_get_nds2(self, gw150914_16384):
        """Test that `TimeSeries.get(..., source="nds2")` works."""
        # get using NDS2 (if datafind could have been used to start with)
        with mock.patch("gwpy.timeseries.TimeSeries.fetch") as mock_fetch:
            ts = self.TEST_CLASS.get(
                NDS2_GW150914_CHANNEL,
                *GWOSC_GW150914_SEGMENT,
                source="nds2",
            )
        mock_fetch.assert_not_called()
        utils.assert_quantity_sub_equal(
            ts,
            gw150914_16384,
            exclude=["name", "channel", "unit"],
        )

    # -- signal processing methods ---

    def test_fft(self, gw150914):
        """Test `TimeSeries.fft`."""
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

    @pytest.mark.parametrize("data", [
        [1., 0., -1., 0.],
        [1., 2., 3., 2., 1., 0.],
        numpy.arange(10),
        RNG.random(size=100),
    ])
    def test_fft_ifft(self, data):
        """Test that `TimeSeries.fft().ifft()` returns the original data."""
        a = self.TEST_CLASS(data)
        utils.assert_quantity_sub_equal(
            a,
            a.fft().ifft(),
            almost_equal=True,
            rtol=1e-7,
            atol=1e-10,
        )

    def test_average_fft(self, gw150914):
        """Test `TimeSeries.average_fft()`."""
        # test all defaults
        fs = gw150914.average_fft()
        utils.assert_quantity_sub_equal(fs, gw150914.detrend().fft())

        # test fftlength
        fs = gw150914.average_fft(fftlength=0.5)
        assert fs.size == 0.5 * gw150914.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hertz

        fs = gw150914.average_fft(fftlength=0.4, overlap=0.2)

    def test_psd_default_overlap(self, gw150914):
        """Test that `TimeSeries.psd()` default overlap is half fftlength."""
        utils.assert_quantity_sub_equal(
            gw150914.psd(.5, method="median", window="hann"),
            gw150914.psd(.5, .25, method="median", window="hann"),
        )

    @pytest.mark.requires("lal")
    def test_psd_lal_median_mean(self, gw150914):
        """Test `TimeSeries.psd(method='lal_median_mean')`."""
        # check that warnings and errors get raised in the right place
        # for a median-mean PSD with the wrong data size or parameters

        # single segment should raise error
        with (
            pytest.raises(ValueError, match="Cannot calculate median-mean spectrum"),
            pytest.deprecated_call(),
        ):
            gw150914.psd(abs(gw150914.span), method="lal_median_mean")

        # odd number of segments should warn
        with (
            pytest.warns(UserWarning, match="Data array is the wrong size"),
            pytest.deprecated_call(),
        ):
            gw150914.psd(1, .5, method="lal_median_mean")

    @pytest.mark.parametrize("method", ["welch", "bartlett", "median"])
    def test_psd(self, noisy_sinusoid, method):
        """Test `TimeSeries.psd()`."""
        fftlength = .5
        overlap = .25
        fs = noisy_sinusoid.psd(
            fftlength=fftlength,
            overlap=overlap,
            method=method,
        )
        assert fs.unit == noisy_sinusoid.unit ** 2 / units.Hz
        assert fs.max() == fs.value_at(500)
        assert fs.size == fftlength * noisy_sinusoid.sample_rate.value // 2 + 1
        assert fs.f0 == 0 * units.Hz
        assert fs.df == units.Hz / fftlength
        assert fs.name == noisy_sinusoid.name
        assert fs.channel is noisy_sinusoid.channel

    @pytest.mark.parametrize(("library", "method"), chain(
        product(["pycbc.psd"], ["welch", "bartlett", "median", "median_mean"]),
        product(["lal"], ["welch", "bartlett", "median", "median_mean"]),
    ))
    def test_psd_deprecated(self, noisy_sinusoid, library, method):
        """Test deprecated average methods for TimeSeries.psd."""
        pytest.importorskip(library)

        fftlength = .5
        overlap = .25

        # remove final .25 seconds to stop median-mean complaining
        # (means an even number of overlapping FFT segments)
        if method == "median_mean":
            end = noisy_sinusoid.span[1]
            noisy_sinusoid = noisy_sinusoid.crop(end=end-overlap)

        # get actual method name
        library = library.split(".", 1)[0]

        with pytest.deprecated_call():
            psd = noisy_sinusoid.psd(
                fftlength=fftlength,
                overlap=overlap,
                method=f"{library}-{method}",
            )

        assert isinstance(psd, FrequencySeries)
        assert psd.unit == noisy_sinusoid.unit ** 2 / units.Hz
        assert psd.max() == psd.value_at(500)

    def test_asd(self, gw150914):
        """Test that `TimeSeries.asd()` is the sqrt of `psd()`."""
        kw = {
            "method": "median",
        }
        utils.assert_quantity_sub_equal(
            gw150914.asd(1, **kw),
            gw150914.psd(1, **kw) ** (1/2.),
        )

    def test_csd(self, noisy_sinusoid):
        """Test that `TimeSeries.csd(self)` is the same as `psd()`."""
        fs = noisy_sinusoid.csd(noisy_sinusoid, average="mean")
        utils.assert_quantity_sub_equal(
            fs.abs(),
            noisy_sinusoid.psd(method="welch"),
            exclude=["name"],
            almost_equal=True,
        )

    def test_csd_fftlength(self, noisy_sinusoid, corrupt_noisy_sinusoid):
        """Test that `TimeSeries.csd` uses the ``fftlength`` keyword properly."""
        # test fftlength is used
        fs = noisy_sinusoid.csd(corrupt_noisy_sinusoid, fftlength=0.5)
        assert fs.size == 0.5 * noisy_sinusoid.sample_rate.value // 2 + 1
        assert fs.df == 2 * units.Hertz

        # test that the default overlap works (by comparing to explicit)
        utils.assert_quantity_sub_equal(
            fs,
            noisy_sinusoid.csd(
                corrupt_noisy_sinusoid,
                fftlength=0.5,
                overlap=0.25,
            ),
        )

    @staticmethod
    def _window_helper(series, fftlength, window="hamming"):
        nfft = int(series.sample_rate.value * fftlength)
        return signal.get_window(window, nfft)

    @pytest.mark.parametrize("method", [
        "scipy-welch",
        "scipy-bartlett",
        "scipy-median",
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
    @pytest.mark.parametrize("window", [None, "hann", ("kaiser", 24), "array"])
    def test_spectrogram(self, gw150914, method, window):
        """Test `TimeSeries.spectrogram()`."""
        # generate window for 'array'
        win = self._window_helper(gw150914, 1) if window == "array" else window

        if method.startswith(("lal", "pycbc")):
            ctx = pytest.deprecated_call  # ignore: type[assignment]
        else:
            ctx = nullcontext  # ignore: type[assignment]

        # generate spectrogram
        with ctx():
            sg = gw150914.spectrogram(1, method=method, window=win)

        # validate
        assert isinstance(sg, Spectrogram)
        assert sg.shape == (
            abs(gw150914.span),
            gw150914.sample_rate.value // 2 + 1,
        )
        assert sg.f0 == 0 * units.Hz
        assert sg.df == 1 * units.Hz
        assert sg.channel is gw150914.channel
        assert sg.unit == gw150914.unit ** 2 / units.Hz
        assert sg.epoch == gw150914.epoch
        assert sg.span == gw150914.span

        # check the first time-bin is the same result as .psd()
        n = int(gw150914.sample_rate.value)
        if window == "hann" and not method.endswith("bartlett"):
            n = int(n * 1.5)  # default is 50% overlap
        with ctx():
            psd = gw150914[:n].psd(fftlength=1, method=method, window=win)
        # FIXME: epoch should not be excluded here (probably)
        utils.assert_quantity_sub_equal(
            sg[0],
            psd,
            exclude=["epoch"],
            almost_equal=True,
        )

    def test_spectrogram_fftlength(self, gw150914):
        """Test `TimeSeries.spectrogram()` uses the ``fftlength`` keyword properly."""
        sg = gw150914.spectrogram(1, fftlength=0.5, method="median")
        assert sg.shape == (
            abs(gw150914.span),
            0.5 * gw150914.sample_rate.value // 2 + 1,
        )
        assert sg.df == 2 * units.Hertz
        assert sg.dt == 1 * units.second

    def test_spectrogram_overlap(self, gw150914):
        """Test `TimeSeries.spectrogram()` uses the ``overlap`` keyword properly."""
        kw = {
            "fftlength": 0.5,
            "window": "hann",
            "method": "median",
        }
        sg = gw150914.spectrogram(1, **kw)
        sg2 = gw150914.spectrogram(1, overlap=.25, **kw)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    def test_spectrogram_multiprocessing(self, gw150914):
        """Test `TimeSeries.spectrogram()` with multiprocessing."""
        kw = {
            "fftlength": 0.5,
            "window": "hann",
            "method": "median",
        }
        sg = gw150914.spectrogram(1, **kw)
        sg2 = gw150914.spectrogram(1, nproc=2, **kw)
        utils.assert_quantity_sub_equal(sg, sg2, almost_equal=True)

    @pytest.mark.parametrize("library", [
        pytest.param("lal", marks=pytest.mark.requires("lal")),
        pytest.param("pycbc", marks=pytest.mark.requires("pycbc.psd")),
    ])
    def test_spectrogram_median_mean(self, gw150914, library):
        """Test `TimeSeries.spectrogram()` with (deprecated) median-mean."""
        method = f"{library}-median-mean"

        # the LAL implementation of median-mean warns if not given the
        # correct amount of data for an even number of FFTs.
        if library == "lal":
            lal_warn_ctx = pytest.warns(
                UserWarning,
                match="Data array is the wrong size",
            )
        else:
            lal_warn_ctx = nullcontext()  # ignore: type[assignment]

        with pytest.deprecated_call(), lal_warn_ctx:
            sg = gw150914.spectrogram(
                1.5,
                fftlength=.5,
                overlap=0,
                method=method,
            )

        assert sg.dt == 1.5 * units.second
        assert sg.df == 2 * units.Hertz

    def test_spectrogram2(self, gw150914):
        """Test `TimeSeries.spectrogram2()`."""
        # test defaults
        sg = gw150914.spectrogram2(1, overlap=0)
        utils.assert_quantity_sub_equal(
            sg,
            gw150914.spectrogram(
                1,
                fftlength=1,
                overlap=0,
                method="scipy-welch",
                window="hann",
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
        """Test `TimeSeries.fftgram()`."""
        fgram = gw150914.fftgram(1)
        fs = int(gw150914.sample_rate.value)
        f, t, sxx = signal.spectrogram(
            gw150914,
            fs,
            window="hann",
            nperseg=fs,
            mode="complex",
        )
        utils.assert_array_equal(gw150914.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram.value)

    def test_fftgram_overlap(self, gw150914):
        """Test `TimeSeries.fftgram()` with overlap."""
        fgram = gw150914.fftgram(1, overlap=0.5)
        fs = int(gw150914.sample_rate.value)
        f, t, sxx = signal.spectrogram(
            gw150914,
            fs,
            window="hann",
            nperseg=fs,
            noverlap=fs//2,
            mode="complex",
        )
        utils.assert_array_equal(gw150914.t0.value + t, fgram.xindex.value)
        utils.assert_array_equal(f, fgram.yindex.value)
        utils.assert_array_equal(sxx.T, fgram.value)

    def test_spectral_variance(self, gw150914):
        """Test `TimeSeries.spectral_variance()`."""
        variance = gw150914.spectral_variance(.5, method="median")
        assert isinstance(variance, SpectralVariance)
        assert variance.x0 == 0 * units.Hz
        assert variance.dx == 2 * units.Hz
        assert variance.max() == 8

    def test_rayleigh_spectrum(self, gw150914):
        """Test `TimeSeries.rayleigh_spectrum()`."""
        # assert single FFT creates Rayleigh of 0
        ray = gw150914.rayleigh_spectrum()
        assert isinstance(ray, FrequencySeries)
        assert ray.unit is units.Unit("")
        assert ray.name == f"Rayleigh spectrum of {gw150914.name}"
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
        """Test `TimeSeries.csd_spectrogram()`."""
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
        utils.assert_quantity_sub_equal(sg[0], csd, exclude=["name", "epoch"])

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

    @pytest.mark.parametrize("ftype", ["fir", "iir"])
    @pytest.mark.parametrize("rate", [
        1024,
        pytest.param(units.Quantity(1024, "Hz"), id="1024 Hz"),
        pytest.param(units.Quantity(1.5, "kHz"), id="1.5 kHz"),
    ])
    def test_resample(self, gw150914, rate, ftype):
        """Test `gwpy.timeseries.TimeSeries.resample`."""
        # resample
        l2 = gw150914.resample(rate, ftype=ftype)
        assert l2.sample_rate == units.Quantity(rate, units.Hz)

        # compare ASDs
        asd1 = gw150914.asd(fftlength=4, overlap=2).crop(30, 400)
        asd2 = l2.asd(fftlength=4, overlap=2).crop(30, 400)
        max1 = asd1.argmax()
        max2 = asd2.argmax()
        # check that the peak occurs at the same frequency
        assert max1 == max2
        # check that the peak itself is approximately the same
        assert asd1[max1].value == pytest.approx(asd2[max2].value)

    def test_resample_simple_upsample(self, gw150914):
        """Test consistency when upsampling by 2x`."""
        upsamp = gw150914.resample(gw150914.sample_rate.value * 2)
        assert numpy.allclose(gw150914.value, upsamp.value[::2])

    def test_resample_noop(self):
        """Test that resampling to the same rate returns self with a warning."""
        data = self.TEST_CLASS([1, 2, 3, 4, 5])
        with pytest.warns(
            UserWarning,
            match=r"resample\(\) rate matches current sample_rate",
        ):
            new = data.resample(data.sample_rate)
        assert data is new

    def test_resample_rate_units(self):
        """Check that TimeSeries.resample fails on weird units."""
        with pytest.raises(
            ValueError,
            match="invalid resampling rate",
        ):
            self.TEST_CLASS([1, 2, 3, 4, 5]).resample(units.Quantity(1, "m"))

    def test_rms(self, gw150914):
        """Test `TimeSeries.rms()`."""
        rms = gw150914.rms(1.)
        assert rms.sample_rate == 1 * units.Hz

    @mock.patch("gwpy.segments.DataQualityFlag.query", return_value=LIVETIME)
    def test_mask(self, mock_query):
        """Test `TimeSeries.mask()`."""
        # craft a timeseries of ones that can be easily tested against
        # a few interesting corner cases
        data = TimeSeries(numpy.ones(8192), sample_rate=128)
        masked = data.mask(flag="X1:TEST-FLAG:1")

        # create tukey window for comparison
        window = tukey(128, alpha=1.0)
        times = (data.t0 + numpy.arange(data.size) * data.dt).value
        (live, ) = numpy.nonzero([t in LIVETIME.active for t in times])
        (dead, ) = numpy.nonzero([t not in LIVETIME.active for t in times])

        # verify the mask is correct
        assert data.is_compatible(masked)
        assert live.size + dead.size == data.size
        assert numpy.all(numpy.isfinite(masked.value[live]))
        assert numpy.all(numpy.isnan(masked.value[dead]))
        utils.assert_allclose(masked.value[:4032], numpy.ones(4032))
        # First segment [0, 32): only right side tapered, use last 64 samples
        utils.assert_allclose(masked.value[4032:4096], window[-64:])
        # Second segment [34, 34.5): both sides tapered
        utils.assert_allclose(
            masked.value[4352:4416],
            window[:64] * window[-64:],
        )

    def test_demodulate(self):
        """Test `TimeSeries.demodulate()`."""
        # create a timeseries that is simply one loud sinusoidal oscillation
        # at a particular frequency, then demodulate at that frequency and
        # recover the amplitude and phase
        amp, phase, f = 1., numpy.pi/4, 30
        duration, sample_rate, stride = 600, 4096, 60
        t = numpy.linspace(0, duration, duration*sample_rate)
        data = TimeSeries(amp * numpy.cos(2*numpy.pi*f*t + phase),
                          unit="", times=t)

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
        assert ph.unit == "deg"
        utils.assert_allclose(mag.value, amp, rtol=1e-5)
        utils.assert_allclose(ph.value, numpy.rad2deg(phase), rtol=1e-5)

        # test with exp=False, deg=False
        mag, ph = data.demodulate(f, stride=stride, deg=False)
        assert ph.unit == "rad"
        utils.assert_allclose(ph.value, phase, rtol=1e-5)

    def test_heterodyne(self):
        """Test `TimeSeries.heterodyne()`."""
        # create a timeseries that is simply one loud sinusoidal oscillation,
        # with a frequency and frequency derivative, then heterodyne using the
        # phase evolution to recover the amplitude and phase
        amp, phase, f, fdot = 1., numpy.pi/4, 30, 1e-4
        duration, sample_rate, stride = 600, 4096, 60
        t = numpy.linspace(0, duration, duration*sample_rate)
        phases = 2 * numpy.pi * (f * t + 0.5 * fdot * t ** 2)
        data = TimeSeries(
            amp * numpy.cos(phases + phase),
            unit="",
            times=t,
        )

        # test with default settings
        het = data.heterodyne(phases, stride=stride)
        assert het.unit == data.unit
        assert het.size == duration // stride
        utils.assert_allclose(numpy.abs(het.value), 0.5*amp, rtol=1e-4)
        utils.assert_allclose(numpy.angle(het.value), phase, rtol=2e-4)

        # test with singlesided=True
        het = data.heterodyne(
            phases,
            stride=stride,
            singlesided=True,
        )
        assert het.unit == data.unit
        assert het.size == duration // stride
        utils.assert_allclose(numpy.abs(het.value), amp, rtol=1e-4)
        utils.assert_allclose(numpy.angle(het.value), phase, rtol=2e-4)

    def test_heterodyne_scalar_phase_error(self, array):
        """Test `TimeSeries.heterodyne()` scalar phase error."""
        with pytest.raises(
            ValueError,
            match="Phase is not array_like",
        ):
            array.heterodyne(1.0)

    def test_heterodyne_phase_length_error(self, array):
        """Test `TimeSeries.heterodyne()` handling of a bad phase array."""
        with pytest.raises(
            ValueError,
            match="Phase array must be the same length",
        ):
            array.heterodyne(array[0:len(array) // 2])

    def test_taper(self):
        """Test `TimeSeries.taper()`."""
        # create a flat timeseries, then taper it
        t = numpy.linspace(0, 1, 2048)
        data = TimeSeries(numpy.cos(10*numpy.pi*t), times=t, unit="")
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

    def test_gate(self):
        # generate Gaussian noise with std = 0.5
        noise = self.TEST_CLASS(
            numpy.random.normal(scale=0.5, size=16384*64),
            sample_rate=16384,
            epoch=-32,
        )
        # generate a glitch with amplitude 20 at 1000 Hz
        glitchtime = 0.0
        glitch = signal.gausspulse(
            noise.times.value - glitchtime,
            bw=100,
        ) * 20
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

    def test_whiten(self, colored_noise, gausspulse):
        """Test `TimeSeries.whiten()`."""
        # Inject the pulse into the middle of the noise
        glitchtime = (colored_noise.t0 + colored_noise.duration / 2).value
        gausspulse.shift(glitchtime)
        data = colored_noise.inject(gausspulse * 1e-2)

        # When the input is stationary Gaussian noise, the output should have
        # zero mean and unit variance
        whitened = colored_noise.whiten(detrend="linear", method="median")
        assert whitened.size == colored_noise.size
        nptest.assert_almost_equal(whitened.mean().value, 0.0, decimal=1)
        nptest.assert_almost_equal(whitened.std().value, 1.0, decimal=1)
        tmax = data.times[data.argmax()].value
        assert tmax != glitchtime

        # When a loud signal is present,
        # the max amplitude should be recovered at the time of that signal
        whitened = data.whiten(detrend="linear", method="median")
        tmax = whitened.times[whitened.argmax()].value
        assert tmax == pytest.approx(glitchtime)

    def test_convolve(self):
        data = self.TEST_CLASS(
            signal.get_window("hann", 1024),
            sample_rate=512,
            epoch=-1,
        )
        filt = numpy.array([1, 0])

        # check that the 'valid' data are unchanged by this filter
        convolved = data.convolve(filt)
        assert convolved.size == data.size
        utils.assert_allclose(convolved.value[1:-1], data.value[1:-1])

    def test_correlate(self, colored_noise, gausspulse):
        # Inject the pulse into the middle of the noise
        glitchtime = (colored_noise.t0 + colored_noise.duration / 2).value
        gausspulse.shift(glitchtime)
        data = colored_noise.inject(gausspulse * 1e-2)

        # check that, without a signal present, we only see background
        snr = colored_noise.correlate(gausspulse, whiten=True, method="median")
        tmax = snr.times[snr.argmax()]
        assert snr.size == colored_noise.size
        assert not numpy.isclose(tmax.value, glitchtime)
        nptest.assert_almost_equal(snr.mean().value, 0.0, decimal=1)
        nptest.assert_almost_equal(snr.std().value, 1.0, decimal=1)

        # Recover the glitch
        snr = data.correlate(gausspulse, whiten=True, method="median")
        tmax = snr.times[snr.argmax()]
        nptest.assert_almost_equal(tmax.value, glitchtime)

    def test_detrend(self, gw150914):
        assert not numpy.isclose(gw150914.value.mean(), 0.0, atol=1e-21)
        detrended = gw150914.detrend()
        assert numpy.isclose(detrended.value.mean(), 0.0)

    def test_filter(self, gw150914):
        """Test `TimeSeries.filter()`."""
        fts = gw150914.filter(([], [], 1), analog=True)
        utils.assert_quantity_sub_equal(gw150914, fts)

    def test_filter_sos(self, gw150914):
        """Test `TimeSeries.filter()` with SOS filters."""
        zpk = filter_design.highpass(50, sample_rate=gw150914.sample_rate)
        sos = signal.zpk2sos(*zpk)
        utils.assert_quantity_almost_equal(
            gw150914.filter(zpk),
            gw150914.filter(sos),
        )

    def test_zpk(self, gw150914):
        """Test `TimeSeries.zpk()`.

        The zpk method is just a wrapper around filter, so we just
        check that they give the same result.
        """
        zpk = [10, 10], [1, 1], 100
        utils.assert_quantity_sub_equal(
            gw150914.zpk(*zpk, analog=True, unit="Hz"),
            gw150914.filter(zpk, analog=True, unit="Hz"),
        )

    def test_highpass_happy_path(self, gw150914):
        """Check that passband val are approx equal, stopband are not."""
        asd = gw150914.asd()
        hp_asd = gw150914.highpass(100).asd()

        eqinds = numpy.where(hp_asd.frequencies.value > 200)[0]
        eqind0 = eqinds[0]

        # be within 50% for all values after
        # (tolerance increased from 40% due to fix for numerical
        # stability in ZPK filter gain application - see issue #1544)
        # numpy allclose formula:
        # |a-b| <= atol + rtol * |b|

        assert numpy.allclose(
            hp_asd[eqind0:].value,
            asd[eqind0:].value,
            rtol=0.5,
            atol=0,
        )

        # dont be within 40% for all value before
        assert not numpy.allclose(
            hp_asd[:eqind0].value,
            asd[:eqind0].value,
            rtol=0.4,
            atol=0,
        )

    def test_lowpass_happy_path(self, gw150914):
        """Check that passband val are approx equal, stopband are not."""
        asd = gw150914.asd()
        lp_asd = gw150914.lowpass(500).asd()

        eqinds = numpy.where(lp_asd.frequencies.value < 500)[0]
        eqind0 = eqinds[0]

        # be within 40% for all values before
        # numpy allclose formula:
        # |a-b| <= atol + rtol * |b|

        assert not numpy.allclose(
            lp_asd[eqind0:].value,
            asd[eqind0:].value,
            rtol=0.4,
            atol=0,
        )

        # dont be within 40% for all value after
        assert numpy.allclose(
            lp_asd[:eqind0].value,
            asd[:eqind0].value,
            rtol=0.4,
            atol=0,
        )

    def test_notch_happy_path(self, gw150914):
        """Check passband vals are approx equal, stopband are not."""
        nf = 10
        notched = gw150914.notch(nf, filtfilt=True)
        notched_asd = notched.asd()
        asd = gw150914.asd()

        n_eps = 3
        eps = n_eps * notched_asd.df.value

        # indices outside interval around 10 rad/s
        l_inds = numpy.where(notched_asd.frequencies.value < nf - eps)[0]
        r_inds = numpy.where(notched_asd.frequencies.value > nf + eps)[0]
        # index at 10 rad/s
        nf_ind = numpy.argmin(numpy.abs(notched_asd.frequencies.value - nf))
        # indices inside interval around 10 rad/s
        nf_inds = numpy.arange(nf_ind - n_eps, nf_ind + n_eps)

        assert l_inds[-1] <= nf_ind
        assert r_inds[0] >= nf_ind

        # be within 40% for all values outside nbrhood
        assert numpy.allclose(
            notched_asd[l_inds].value,
            asd[l_inds].value,
            rtol=0.4,
            atol=0,
        )
        assert numpy.allclose(
            notched_asd[r_inds].value,
            asd[r_inds].value,
            rtol=0.4,
            atol=0,
        )

        # dont be within 40% for all values inside nbrhood
        assert not numpy.allclose(
            notched_asd[nf_inds].value,
            asd[nf_inds].value,
            rtol=0.4,
            atol=0,
        )

        # biggest difference between filtered and unfiltered
        # should be at closest f to nf=10
        absd = numpy.abs(notched_asd.value - asd.value)
        assert numpy.isclose(absd[nf_ind], numpy.max(absd))

    def test_bandpass_happy_path(self, gw150914):
        """Check that passband val are approx equal, stopband are not."""
        asd = gw150914.asd()
        bp_asd = gw150914.bandpass(100, 1000).asd()

        eqinds = numpy.where(
            numpy.logical_and(
                bp_asd.frequencies.value > 100,
                bp_asd.frequencies.value < 1000,
            ),
        )[0]

        eqind0 = eqinds[0]
        eqindn = eqinds[-1]

        # be within 50% for all values in passband
        # (tolerance increased from 40% due to fix for numerical
        # stability in ZPK filter gain application - see issue #1544)
        # numpy allclose formula:
        # |a-b| <= atol + rtol * |b|

        assert numpy.allclose(
            bp_asd[eqind0:eqindn].value,
            asd[eqind0:eqindn].value,
            rtol=0.5,
            atol=0,
        )

        # dont be within 40% for all value before
        assert not numpy.allclose(
            bp_asd[:eqind0].value,
            asd[:eqind0].value,
            rtol=0.4,
            atol=0,
        )

        # or after
        assert not numpy.allclose(
            bp_asd[eqindn:].value,
            asd[eqindn:].value,
            rtol=0.4,
            atol=0,
        )

    def test_notch(self, gw150914):
        """Test `TimeSeries.notch()`."""
        # test notch runs end-to-end
        gw150914.notch(60)

        # test breaks when you try and 'fir' notch
        with pytest.raises(NotImplementedError):
            gw150914.notch(10, type="fir")

    def test_q_gram(self, gw150914):
        """Test `TimeSeries.q_gram()`."""
        qgram = gw150914.q_gram()
        assert isinstance(qgram, EventTable)
        assert qgram.meta["q"] == 45.25483399593904
        assert qgram["energy"].min() >= 5.5**2 / 2
        nptest.assert_almost_equal(qgram["energy"].max(), 10559.25, decimal=2)

    def test_q_transform(self, gw150914):
        """Test `TimeSeries.q_transform()`."""
        qspecgram = gw150914.q_transform(method="scipy-welch", fftlength=2)
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (1000, 2403)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 155.93567, decimal=5)
        nptest.assert_almost_equal(
            qspecgram.value.mean(),
            1.936469,
            decimal=5,
        )

        # test whitening args
        asd = gw150914.asd(2, 1, method="scipy-welch")
        qsg2 = gw150914.q_transform(method="scipy-welch", whiten=asd)
        utils.assert_quantity_sub_equal(qspecgram, qsg2, almost_equal=True)

        asd = gw150914.asd(.5, .25, method="scipy-welch")
        qsg2 = gw150914.q_transform(method="scipy-welch", whiten=asd)
        qsg3 = gw150914.q_transform(
            method="scipy-welch",
            fftlength=.5,
            overlap=.25,
        )
        utils.assert_quantity_sub_equal(qsg2, qsg3, almost_equal=True)

        # make sure frequency too high presents warning
        with pytest.warns(
            UserWarning,
            match="upper frequency of 10000.0 Hz is too high",
        ):
            qspecgram = gw150914.q_transform(
                method="scipy-welch",
                frange=(0, 10000),
            )
        nptest.assert_almost_equal(
            qspecgram.yspan[1],
            1291.5316,
            decimal=4,
        )

        # test other normalisations work (or don't)
        q2 = gw150914.q_transform(method="scipy-welch", norm="median")
        utils.assert_quantity_sub_equal(qspecgram, q2, almost_equal=True)
        gw150914.q_transform(method="scipy-welch", norm="mean")
        gw150914.q_transform(method="scipy-welch", norm=False)
        with pytest.raises(
            ValueError,
            match=r"^invalid normalisation 'blah'$",
        ):
            gw150914.q_transform(method="scipy-welch", norm="blah")

    def test_q_transform_logf(self, gw150914):
        """Test `TimeSeries.q_transform()` with log frequency spacing."""
        qspecgram = gw150914.q_transform(
            method="scipy-welch",
            fftlength=2,
            logf=True,
        )
        assert isinstance(qspecgram, Spectrogram)
        assert qspecgram.shape == (1000, 500)
        assert qspecgram.q == 5.65685424949238
        nptest.assert_almost_equal(qspecgram.value.max(), 155.93774, decimal=5)

    def test_q_transform_nan(self):
        """Test that `TimeSeries.q_transform()` raises for NaN input."""
        data = TimeSeries(numpy.empty(256*10) * numpy.nan, sample_rate=256)
        with pytest.raises(
            ValueError,
            match=r"^Input signal contains non-numerical values$",
        ):
            data.q_transform(method="median")

    def test_boolean_statetimeseries(self, array):
        """Test comparisons of `TimeSeries` returning `StateTimeSeries`."""
        comp = array >= 2 * array.unit
        assert isinstance(comp, StateTimeSeries)
        assert comp.unit is units.Unit("")
        assert comp.name == f"{array.name} >= 2.0"
        assert (array == array).name == f"{array.name} == {array.name}"  # noqa: PLR0124

    @pytest_skip_flaky_network
    def test_transfer_function(self, gw150914_h1_32, gw150914_l1_32):
        """Test `TimeSeries.transfer_function()`."""
        tf = gw150914_h1_32.transfer_function(
            gw150914_l1_32,
            fftlength=1.0,
            overlap=0.5,
        )
        assert tf.df == 1 * units.Hz
        assert tf.frequencies[abs(tf).argmax()] == 516 * units.Hz

    @pytest_skip_flaky_network
    def test_coherence(self, gw150914_h1_32, gw150914_l1_32):
        """Test `TimeSeries.coherence()`."""
        coh = gw150914_h1_32.coherence(
            gw150914_l1_32,
            fftlength=1.0,
        )
        assert coh.df == 1 * units.Hz
        assert coh.frequencies[coh.argmax()] == 60 * units.Hz

    @pytest_skip_flaky_network
    def test_coherence_spectrogram(self, gw150914_h1_32, gw150914_l1_32):
        """Test `TimeSeries.coherence_spectrogram()`."""
        cohsg = gw150914_h1_32.coherence_spectrogram(
            gw150914_l1_32,
            4,
            fftlength=1.0,
        )
        assert cohsg.t0 == gw150914_h1_32.t0
        assert cohsg.dt == 4 * units.second
        assert cohsg.df == 1 * units.Hz
        _, fmax = numpy.unravel_index(cohsg.argmax(), cohsg.shape)
        assert cohsg.frequencies[fmax] == 60 * units.Hz


# -- TimeSeriesDict ------------------

class TestTimeSeriesDict(_TestTimeSeriesBaseDict[TimeSeriesDict, TimeSeries]):
    """Tests for `gwpy.timeseries.TimeSeriesDict`."""

    channels: ClassVar[list[str]] = ["H1:LDAS-STRAIN", "L1:LDAS-STRAIN"]
    TEST_CLASS = TimeSeriesDict
    ENTRY_CLASS = TimeSeries

    @pytest.mark.requires("framel")
    def test_read_write_gwf(self, instance, tmp_path):
        """Test reading and writing GWF files."""
        tmp = tmp_path / "test.gwf"
        instance.write(tmp)
        new = self.TEST_CLASS.read(tmp, instance.keys())
        for key in new:
            utils.assert_quantity_sub_equal(new[key], instance[key],
                                            exclude=["channel"])

    def test_read_write_hdf5(self, instance, tmp_path):
        """Test reading and writing HDF5 files."""
        tmp = tmp_path / "test.h5"
        instance.write(tmp, overwrite=True)
        new = self.TEST_CLASS.read(tmp, instance.keys())
        for key in new:
            utils.assert_quantity_sub_equal(new[key], instance[key])
        # check auto-detection of names
        new = self.TEST_CLASS.read(tmp)
        for key in new:
            utils.assert_quantity_sub_equal(new[key], instance[key])


# -- TimeSeriesList ------------------

class TestTimeSeriesList(_TestTimeSeriesBaseList[TimeSeriesList, TimeSeries]):
    """Tests for :class:`gwpy.timeseries.TimeSeriesList`."""

    TEST_CLASS = TimeSeriesList
    ENTRY_CLASS = TimeSeries
