# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2022)
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

"""Unit tests for :mod:`gwpy.io.ffldatafind`
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

from contextlib import nullcontext
from unittest import mock

import pytest

from .. import ffldatafind


# -- test utilities ---------

@pytest.mark.parametrize("content", [
    "abcdef",
    "line 1\nline2",
])
def test_read_last_line(tmp_path, content):
    """Check that the `_read_last_line` utility works
    """
    # write content to a file
    path = tmp_path / "tmp.txt"
    path.write_text(content)
    result = content.rsplit("\n", 1)[-1]
    assert ffldatafind._read_last_line(path) == result


def test_read_last_line_oserror(tmp_path):
    """Check that `_read_last_line` raises `OSError` for empty files.
    """
    path = tmp_path / "tmp.txt"
    path.touch(exist_ok=False)
    with pytest.raises(OSError):
        assert ffldatafind._read_last_line(path)


# -- test ffl UI ------------

FFLS = {
    "test.ffl": [
        "/tmp/X-test-0-1.gwf 0 1 0 0",
        "/tmp/X-test-1-1.gwf 1 1 0 0",
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
}
TEST_URLS = [x.split()[0] for x in FFLS["test.ffl"]]


@pytest.fixture(autouse=True)
def mock_ffl(tmp_path):
    """Create a temporary FFL file structure and mock it into
    the test environment.
    """
    for path, lines in FFLS.items():
        ffl = tmp_path / path
        ffl.write_text("\n".join(lines))
    with mock.patch.dict(
        "os.environ",
        {"FFLPATH": str(tmp_path)},
        clear=True,
    ):
        yield


@pytest.mark.parametrize(("site", "match", "result"), [
    (None, None, ["test", "test2", "test3"]),
    ("X", None, ["test", "test2"]),
    (None, "2", ["test2"]),
    ("Y", "test", ["test3"]),
])
def test_find_types(site, match, result):
    """Check that `ffldatafind.find_types` works.
    """
    assert sorted(ffldatafind.find_types(
        site=site,
        match=match,
    )) == sorted(result)


@pytest.mark.parametrize(("match", "ctx", "result"), (
    (None, nullcontext(), TEST_URLS),
    (r"\-0\-", pytest.warns(UserWarning), TEST_URLS[:1]),
))
def test_find_urls(match, ctx, result):
    """Check that `ffldatafind.find_urls` works.
    """
    with ctx:
        assert sorted(ffldatafind.find_urls(
            "X",
            "test",
            0,
            3,
            match=match,
        )) == result


@pytest.mark.parametrize(("on_gaps", "ctx"), (
    ("ignore", nullcontext()),
    ("warn", pytest.warns(UserWarning)),
    ("raise", pytest.raises(RuntimeError)),
))
def test_find_urls_on_missing(on_gaps, ctx):
    """Check that the ``on_missing`` keyword in `ffldatafind.find_latest`
    works in each case.
    """
    with ctx:
        urls = ffldatafind.find_urls(
            "X",
            "test",
            100,
            101,
            on_gaps=on_gaps,
        )
    if on_gaps != "raise":
        assert urls == []


def test_find_latest():
    """Check that `ffldatafind.find_latest` works.
    """
    assert ffldatafind.find_latest(
        "X",
        "test",
    ) == sorted(x.split()[0] for x in FFLS["test.ffl"])[-1:]
