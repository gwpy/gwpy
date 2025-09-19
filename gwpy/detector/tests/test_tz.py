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

"""Tests for :mod:`gwpy.detector.tz`."""

import datetime

import pytest

from .. import tz

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def test_get_timezone():
    """Test `get_timezone()`."""
    assert tz.get_timezone("G1") == "Europe/Berlin"


def test_get_timezone_error():
    """Test `get_timezone()`."""
    with pytest.raises(
        ValueError,
        match=r"^Unrecognised ifo: 'ABC'$",
    ):
        tz.get_timezone("ABC")


@pytest.mark.parametrize(("ifo", "result"), [
    pytest.param("L1", -21600, id="L1"),
    pytest.param("K1", +32400, id="K1"),
])
def test_get_timezone_offset(ifo, result):
    """Test `get_timezone_offset()`."""
    assert tz.get_timezone_offset(
        ifo,
        datetime.datetime(2025, 1, 1, 0, 0, 0, tzinfo=datetime.UTC),
    ) == result


@pytest.mark.freeze_time("2025-1-1 00:00:00")
def test_get_timezone_offset_now():
    """Test `get_timezone_offset()` with default ``dt`` value."""
    assert tz.get_timezone_offset("G1") == 3600
