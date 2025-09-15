# Copyright (c) 2025 Cardiff University
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

"""Tests for :mod:`gwpy.io.remote`."""

from pathlib import PureWindowsPath

import pytest

from ...testing.errors import pytest_skip_flaky_network
from .. import remote as io_remote

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

REMOTE_FILE = "https://gitlab.com/gwpy/gwpy/-/raw/main/README.md"
README_HEAD = "GWpy is a collaboration-driven Python package"


@pytest.mark.parametrize(("url", "result"), [
    pytest.param("/path/to/data", False, id="path"),
    pytest.param("file:///path/to/data", False, id="file"),
    pytest.param("https://example.com/path/to/data", True, id="http"),
    pytest.param("pelican://example.com/path/to/data", True, id="pelican"),
    pytest.param(PureWindowsPath(r"c:\Users\myname\mydata.txt"), False, id="windows"),
])
def test_is_remote(url, result):
    """Test `is_remote()`."""
    assert io_remote.is_remote(url) is result


@pytest_skip_flaky_network
def test_download_file():
    """Test `download_file()`."""
    path = io_remote.download_file(REMOTE_FILE, cache=False)
    with open(path) as file:
        assert next(file).strip().startswith(README_HEAD)


@pytest_skip_flaky_network
def test_open_remote_file():
    """Test `open_remote_file()`."""
    with io_remote.open_remote_file(
        REMOTE_FILE,
        cache=False,
        encoding="utf-8",
    ) as file:
        assert next(file).strip().startswith(README_HEAD)


@pytest_skip_flaky_network
@pytest.mark.requires("requests_pelican")
def test_download_file_pelican():
    """Test `download_file()` with a ``pelican://`` URI."""
    path = io_remote.download_file(
        "osdf:///gwdata/zenodo/README.zenodo",
        cache=False,
    )
    with open(path) as file:
        assert next(file).strip() == "## Mirror of IGWN Zenodo Communities"


@pytest_skip_flaky_network
@pytest.mark.requires("requests_pelican")
def test_open_remote_file_pelican():
    """Test `open_remote_file()` with a ``pelican://`` URI."""
    with io_remote.open_remote_file(
        "osdf:///gwdata/zenodo/README.zenodo",
        cache=False,
        encoding="utf-8",
    ) as file:
        assert next(file).strip() == "## Mirror of IGWN Zenodo Communities"
