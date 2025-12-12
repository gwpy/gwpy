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

"""Tests for :mod:`gwpy.io.pelican`."""

from unittest import mock

import pytest
from urllib3.util import parse_url

from ...testing.errors import pytest_skip_flaky_network
from .. import pelican as io_pelican

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _protocols():
    return ["pelican", "osdf"]


@mock.patch("gwpy.io.pelican._pelican_protocols", _protocols)
@pytest.mark.parametrize(("url", "result"), [
    pytest.param("/path/to/data", False, id="path"),
    pytest.param("https://example.com/path/to/data", False, id="http"),
    pytest.param("pelican://example.com/path/to/data", True, id="pelican"),
    pytest.param("osdf:///path/to/data", True, id="osdf"),
])
def test_is_pelican_url(url, result):
    """Test `is_pelican_url()`."""
    assert io_pelican.is_pelican_url(url) is result


@pytest_skip_flaky_network
@pytest.mark.requires("requests_pelican")
def test_query_director():
    """Test `query_director()`."""
    urls, needauth, authkw = io_pelican.query_director(
        "osdf:///igwn/ligo",
    )
    assert len(urls) >= 1
    for u in urls:
        parsed = parse_url(u)
        assert parsed.path == "/igwn/ligo"
    assert needauth is True
    # /igwn namespace doesn't get issuer info
    assert "issuer" not in authkw


@pytest_skip_flaky_network
@pytest.mark.requires("requests_pelican")
def test_download_file():
    """Test `download_file()`."""
    path = io_pelican.download_file(
        "osdf:///gwdata/zenodo/README.zenodo",
        cache=False,
    )
    with open(path) as file:
        assert next(file).strip() == "## Mirror of IGWN Zenodo Communities"


@pytest_skip_flaky_network
@pytest.mark.requires("requests_pelican")
def test_open_remote_file():
    """Test `open_remote_file()`."""
    with io_pelican.open_remote_file(
        "osdf:///gwdata/zenodo/README.zenodo",
        cache=False,
    ) as file:
        assert next(file).strip() == "## Mirror of IGWN Zenodo Communities"
