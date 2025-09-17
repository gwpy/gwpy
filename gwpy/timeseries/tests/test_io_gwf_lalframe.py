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

"""Tests for :mod:`gwpy.timeseries.io.gwf.lalframe`."""

import warnings
from pathlib import Path
from urllib.parse import urlparse

import numpy
import pytest
from gwdatafind.utils import file_segment

from ...io.cache import write_cache
from ...segments import Segment
from ...testing.utils import (
    TEST_GWF_FILE,
    assert_dict_equal,
    assert_quantity_sub_equal,
)
from ...timeseries import (
    TimeSeries,
    TimeSeriesDict,
)

# import optional dependencies
lal_utils = pytest.importorskip("lal.utils")
lalframe = pytest.importorskip("lalframe")
gwpy_lalframe = pytest.importorskip("gwpy.timeseries.io.gwf.lalframe")

# get URI to test against
TEST_GWF_PATH = Path(TEST_GWF_FILE).absolute()
TEST_GWF_URL = TEST_GWF_PATH.as_uri()

# get info corresponding to this file
TEST_GWF_SEGMENT = file_segment(TEST_GWF_FILE)
TEST_GWF_DELTA_T = 1 / 16384

# channels to read
CHANNELS = ["H1:LDAS-STRAIN", "L1:LDAS-STRAIN", "V1:h_16384Hz"]


@pytest.fixture
def stream():
    """Return a `lalframe.FrStream` pointing at the ``TEST_GWF_PATH``."""
    return lalframe.FrStreamOpen(str(TEST_GWF_PATH.parent), TEST_GWF_PATH.name)


def _test_open_data_source(source):
    """Test `open_data_source()`."""
    stream = gwpy_lalframe.open_data_source(source)
    assert stream.epoch == TEST_GWF_SEGMENT[0]
    assert Path(urlparse(stream.cache.list.url).path).samefile(TEST_GWF_PATH)


@pytest.mark.parametrize("source", [
    TEST_GWF_FILE,
    TEST_GWF_URL,
    [TEST_GWF_FILE],
    lal_utils.CacheEntry.from_T050017(TEST_GWF_FILE),
])
def test_open_data_source(source):
    """Test `open_data_source()`."""
    return _test_open_data_source(source)


@pytest.mark.requires("glue.lal")
def test_open_data_source_glue():
    """Test `open_data_source()` with a `glue.lal.Cache`."""
    from glue.lal import Cache
    Cache.entry_class = lal_utils.CacheEntry
    cache = Cache.from_urls([TEST_GWF_FILE])
    return _test_open_data_source(cache)


def test_open_data_source_cache(tmp_path):
    """Test `open_data_source()` with a cache file."""
    tmp = tmp_path / "test.lcf"
    write_cache([TEST_GWF_FILE], tmp, format="lal")
    return _test_open_data_source(tmp)


def test_open_data_source_error():
    """Check that an invalid source raises the right exception."""
    with pytest.raises(
        ValueError,
        match=r"^Don't know how to open data source of type 'NoneType'$",
    ):
        gwpy_lalframe.open_data_source(None)


def test_get_stream_duration(stream):
    """Test `get_stream_duration()`."""
    assert gwpy_lalframe.get_stream_duration(stream) == 1.


@pytest.mark.parametrize(("start", "end"), [
    (None, None),
    (None, TEST_GWF_SEGMENT[0] + .5),
    (TEST_GWF_SEGMENT[0] + .5, None),
    (TEST_GWF_SEGMENT[0] + .25, TEST_GWF_SEGMENT[1] - .25),
    (TEST_GWF_SEGMENT[1] - TEST_GWF_DELTA_T / 2, None),
])
def test_read(start, end):
    """Test that reading works for a variety of ``[start, stop)`` intervals."""
    data = gwpy_lalframe.read(
        TEST_GWF_FILE,
        CHANNELS,
        start=start,
        end=end,
    )
    assert isinstance(data, dict)

    start = TEST_GWF_SEGMENT[0] if start is None else start
    end = TEST_GWF_SEGMENT[1] if end is None else end

    for name in CHANNELS:
        ts = data[name]
        # check basic parameters
        assert ts.sample_rate.value == 16384
        assert ts.name == name

        # check data span is what we asked for
        assert numpy.allclose(ts.xspan, (start, end), atol=TEST_GWF_DELTA_T)


def test_read_channel_error():
    """Test that LALFrame raises a ValueError when the channel isn't found."""
    with pytest.raises(
        ValueError,
        match=r"^channel 'bad' not found$",
    ):
        gwpy_lalframe.read(TEST_GWF_FILE, ["bad"])


def test_write(tmp_path):
    """Test that LALFrame can write to GWF."""
    # read the data first
    data = gwpy_lalframe.read(TEST_GWF_FILE, CHANNELS)

    # write the data
    tmp = tmp_path / "test.gwf"
    gwpy_lalframe.write(
        data,
        tmp,
        *TEST_GWF_SEGMENT,
    )

    # read it back and check things
    data2 = gwpy_lalframe.read(tmp, CHANNELS)
    assert_dict_equal(data, data2, assert_quantity_sub_equal)


@pytest.mark.parametrize("data", [
    pytest.param(
        [],
        marks=pytest.mark.xfail(
            raises=RuntimeError,
            reason="Cannot add an empty series to a frame",
        ),
    ),
    [1, 2, 3, 4, 5],
])
def test_write_no_ifo(tmp_path, data):
    """Test that writing GWF without an IFO works fine."""
    # create timeseries with no IFO
    data = TimeSeries(data, dtype=float)
    tmp = tmp_path / "test.gwf"
    gwpy_lalframe.write(
        TimeSeriesDict({None: data}),
        tmp,
        *data.span,
    )


def test_read_missing_sample():
    """Check that giving non-sampled start time doesn't result in missing data.

    See https://git.ligo.org/computing/helpdesk/-/issues/4774 and
    https://git.ligo.org/lscsoft/lalsuite/-/issues/710.
    """
    data = TimeSeries.read(
        TEST_GWF_PATH,
        "H1:LDAS-STRAIN",
        start=968654552.6709,
        end=968654553,
        format="gwf",
        backend="lalframe",
    )
    assert data.span == Segment(968654552.670898432, 968654553.0)


def test_read_gwf_scaled_lalframe():
    """Check that LALFrame warns about using 'scaled'."""
    def _read(**kwargs):
        return TimeSeries.read(
            TEST_GWF_PATH,
            "L1:LDAS-STRAIN",
            format="gwf",
            backend="lalframe",
            **kwargs,
        )

    # check that it doesn't warn normally
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        data = _read()

    # and that it does warn when you use scaled=
    with pytest.warns(UserWarning):
        data2 = _read(scaled=True)

    # but the result should be the same
    assert_quantity_sub_equal(data, data2)
