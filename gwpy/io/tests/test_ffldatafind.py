# Copyright (c) 2022-2025 Cardiff University
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

"""Tests for :mod:`gwpy.io.ffldatafind`."""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# ruff: noqa: S108

import os
from contextlib import nullcontext
from pathlib import Path
from unittest import mock

import pytest

from .. import (
    datafind as io_datafind,
    ffldatafind,
)

# -- test utilities ------------------

@pytest.mark.parametrize("content", [
    "abcdef",
    "line 1\nline2",
])
def test_read_last_line(tmp_path, content):
    """Test `_read_last_line()`."""
    ffldatafind._read_last_line.cache_clear()
    # write content to a file
    path = tmp_path / "tmp.txt"
    path.write_text(content)
    result = content.rsplit("\n", 1)[-1]
    assert ffldatafind._read_last_line(path) == result


def test_read_last_line_oserror(tmp_path):
    """Test `_read_last_line()` error handling."""
    ffldatafind._read_last_line.cache_clear()
    path = tmp_path / "tmp.txt"
    path.touch(exist_ok=False)
    with pytest.raises(
        OSError,
        match="Invalid argument",
    ):
        assert ffldatafind._read_last_line(path)


# -- test ffl utils ------------------

@pytest.mark.parametrize(("var", "value", "result"), [
    ("FFLPATH", "/path/to/ffl", "/path/to/ffl"),
    ("VIRGODATA", "/virgoData", "/virgoData/ffl"),
])
@mock.patch.dict("os.environ", clear=True)
def test_get_ffl_basedir(var, value, result):
    """Test `_get_ffl_basedir()`."""
    # note: use str(Path(x)) to convert to Posix->Windows
    os.environ[var] = str(Path(value))
    assert ffldatafind._get_ffl_basedir() == Path(result)


@mock.patch.dict("os.environ", clear=True)
def test_get_ffl_basedir_error():
    """Test `_get_ffl_basedir()` error handling."""
    with pytest.raises(KeyError):
        ffldatafind._get_ffl_basedir()


@pytest.mark.parametrize(("path", "result"), [
    (Path("/path/to/test.ffl"), True),
    (Path("test.txt"), False),
])
def test_is_ffl_file(path, result):
    """Test `_is_ffl_file()`."""
    assert ffldatafind._is_ffl_file(path) is result


# -- test ffl UI ---------------------

FFLS = {
    "a/test.ffl": [
        "/tmp/X-test-0-1.gwf 0 1 0 0",
        "/tmp/X-test-1-1.gwf 1 1 0 0",
    ],
    "b/test.ffl": [
        "/tmp/X-test-2-1.gwf 2 1 0 0",
    ],
    "test2.ffl": [
        "/tmp/X-test2-0-1.gwf 0 1 0 0",
        "/tmp/X-test2-1-1.gwf 1 1 0 0",
        "/tmp/X-test2-2-1.gwf 2 1 0 0",
    ],
    "test3.ffl": [
        "/tmp/Y-test3-0-1.gwf 0 1 0 0",
        "/tmp/Y-test3-1-1.gwf 1 1 0 0",
        "/tmp/Y-test3-2-1.gwf 2 1 0 0",
    ],
    "test-empty.ffl": [],
    "test-bad.ffl": ["badness"],
}
TEST_URLS = [
    x.split()[0]
    for key in ("a/test.ffl", "b/test.ffl")
    for x in FFLS[key]
]


@pytest.fixture(autouse=True)
def mock_ffl(tmp_path):
    """Create an FFL directory tree and mock it into the test environment."""
    for path, lines in FFLS.items():
        ffl = tmp_path / path
        ffl.parent.mkdir(parents=True, exist_ok=True)
        ffl.write_text("\n".join(lines))
    with mock.patch.dict(
        "os.environ",
        {"FFLPATH": str(tmp_path)},
        clear=True,
    ):
        yield


@pytest.mark.parametrize(("site", "match", "result"), [
    pytest.param(None, None, ["test", "test2", "test3"], id="all"),
    pytest.param("X", None, ["test", "test2"], id="X"),
    pytest.param("Y", "test", ["test3"], id="Y"),
    pytest.param(None, "2", ["test2"], id="2"),
])
def test_find_types(site, match, result):
    """Test `find_types()`."""
    assert sorted(ffldatafind.find_types(
        site=site,
        match=match,
    )) == sorted(result)


@pytest.mark.parametrize(("match", "ctx", "result"), [
    (None, nullcontext(), TEST_URLS),
    (r"\-0\-", pytest.warns(UserWarning, match="Missing segments"), TEST_URLS[:1]),
])
def test_find_urls(match, ctx, result):
    """Test `find_urls()`."""
    with ctx:
        assert sorted(ffldatafind.find_urls(
            "X",
            "test",
            0,
            3,
            match=match,
        )) == result


@pytest.mark.parametrize(("on_gaps", "ctx"), [
    ("ignore", nullcontext()),
    ("warn", pytest.warns(UserWarning, match="Missing segments")),
    ("raise", pytest.raises(RuntimeError, match="Missing segments")),
])
def test_find_urls_on_gaps(on_gaps, ctx):
    """Test `find_urls(..., on_gaps=...)`."""
    with ctx:
        assert ffldatafind.find_urls(
            "X",
            "test",
            100,
            101,
            on_gaps=on_gaps,
        ) == []


def test_find_latest():
    """`Test `find_latest()`."""
    assert ffldatafind.find_latest(
        "X",
        "test",
    ) == sorted(x.split()[0] for x in FFLS["b/test.ffl"])[-1:]


@pytest.mark.parametrize(("on_missing", "ctx"), [
    ("ignore", nullcontext()),
    ("warn", pytest.warns(UserWarning, match="No files found")),
    ("raise", pytest.raises(RuntimeError, match="No files found")),
])
def test_find_latest_on_missing(on_missing, ctx):
    """Test `find_latest(... on_missing=...)`."""
    with ctx:
        assert ffldatafind.find_latest(
            "BAD",
            "BAD",
            on_missing=on_missing,
        ) == []


# -- test gwpy.io.datafind interface -

@mock.patch(
    "gwpy.io.datafind.iter_channel_names",
    mock.MagicMock(return_value=["Y1:TEST-CHANNEL"]),
)
@mock.patch(
    "gwpy.io.datafind.on_tape",
    mock.MagicMock(return_value=False),
)
@mock.patch(
    "gwpy.io.datafind.num_channels",
    mock.MagicMock(return_value=1),
)
def test_datafind_find_frametype():
    """Test `gwpy.io.datafind.find_frametype` redirects to `ffldatafind`."""
    assert io_datafind.find_frametype(
        "Y1:TEST-CHANNEL",
        allow_tape=True,
    ) == "test3"
