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

"""Tests for :mod:`gwpy.timeseries.io.gwf.lalframe`
"""

from pathlib import Path

from six.moves.urllib.parse import urlparse

import pytest

from gwdatafind.utils import file_segment

from ...io.cache import write_cache
from ...testing.utils import (
    assert_dict_equal,
    assert_quantity_sub_equal,
    skip_missing_dependency,
    TemporaryFilename,
    TEST_GWF_FILE,
)
from ...timeseries import TimeSeries

# import optional dependencies
lal_utils = pytest.importorskip("lal.utils")
lalframe = pytest.importorskip("lalframe")
gwpy_lalframe = pytest.importorskip("gwpy.timeseries.io.gwf.lalframe")

# get URI to test against
TEST_GWF_PATH = Path(TEST_GWF_FILE).absolute()
TEST_GWF_URL = TEST_GWF_PATH.as_uri()

# get epoch corresponding to this file
TEST_GWF_SEGMENT = file_segment(TEST_GWF_FILE)

# channels to read
CHANNELS = ["H1:LDAS-STRAIN", "L1:LDAS-STRAIN", "V1:h_16384Hz"]


@pytest.fixture
def stream():
    return lalframe.FrStreamOpen(str(TEST_GWF_PATH.parent), TEST_GWF_PATH.name)


def _test_open_data_source(source):
    """This function actually performs the test
    """
    stream = gwpy_lalframe.open_data_source(source)
    assert stream.epoch == TEST_GWF_SEGMENT[0]
    assert TEST_GWF_PATH == Path(
        urlparse(stream.cache.list.url).path
    ).absolute()


@pytest.mark.parametrize("source", [
    TEST_GWF_FILE,
    TEST_GWF_URL,
    [TEST_GWF_FILE],
    lal_utils.CacheEntry.from_T050017(TEST_GWF_FILE),
])
def test_open_data_source(source):
    return _test_open_data_source(source)


@skip_missing_dependency("glue.lal")
def test_open_data_source_glue():
    from glue.lal import Cache
    Cache.entry_class = lal_utils.CacheEntry
    cache = Cache.from_urls([TEST_GWF_FILE])
    return _test_open_data_source(cache)


def test_open_data_source_cache():
    with TemporaryFilename(".lcf") as cache:
        write_cache([TEST_GWF_FILE], cache, format="lal")
        return _test_open_data_source(cache)


def test_open_data_source_error():
    with pytest.raises(ValueError) as exc:
        gwpy_lalframe.open_data_source(None)
    assert str(exc.value) == (
        "Don't know how to open data source of type {}".format(type(None))
    )


def test_get_stream_duration(stream):
    assert gwpy_lalframe.get_stream_duration(stream) == 1.


@pytest.mark.parametrize("start, end", [
    (None, None),
    (None, TEST_GWF_SEGMENT[0] + .5),
    (TEST_GWF_SEGMENT[0] + .5, None),
    (TEST_GWF_SEGMENT[0] + .25, TEST_GWF_SEGMENT[1] - .25)
])
def test_read(start, end):
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
        assert ts.channel.name == name

        # check data span is what we asked for
        assert ts.xspan == (start, end)


def test_read_channel_error():
    with pytest.raises(RuntimeError) as exc:
        gwpy_lalframe.read(TEST_GWF_FILE, ["bad"])
    assert str(exc.value) == "channel 'bad' not found"


def test_read_deprecated_scaled():
    with pytest.warns(UserWarning):
        gwpy_lalframe.read(
            TEST_GWF_FILE,
            ["L1:LDAS-STRAIN"],
            scaled=True,
        )


def test_write():
    # read the data first
    data = gwpy_lalframe.read(TEST_GWF_FILE, CHANNELS)

    with TemporaryFilename() as tmp:
        # write the data
        gwpy_lalframe.write(
            data,
            tmp,
        )

        # read it back and check things
        data2 = gwpy_lalframe.read(tmp, CHANNELS)
        assert_dict_equal(data, data2, assert_quantity_sub_equal)


def test_write_no_ifo():
    # create timeseries with no IFO
    data = TimeSeries([1, 2, 3, 4, 5])
    with TemporaryFilename() as tmp:
        gwpy_lalframe.write(
            {None: data},
            tmp
        )
