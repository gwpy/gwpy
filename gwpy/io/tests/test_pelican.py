# Copyright (C) Cardiff University (2025-)
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

from ...testing.errors import pytest_skip_network_error
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
    """Test `gwpy.io.pelican.is_pelican_url`."""
    assert io_pelican.is_pelican_url(url) is result


@pytest_skip_network_error
@pytest.mark.requires("requests_pelican")
def test_query_director():
    """Test that the `query_director` function works."""
    urls, needauth, authkw = io_pelican.query_director(
        "osdf:///igwn/ligo",
    )
    assert len(urls) >= 1
    assert all(u.endswith(":8443/igwn/ligo") for u in urls)
    assert needauth is True
    assert authkw.get("issuer", None)
