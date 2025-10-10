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

"""Tests for :mod:`gwpy.table.io.omega`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ...testing.utils import assert_table_equal
from ..table import EventTable

if TYPE_CHECKING:
    from pathlib import Path

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


OMEGA_ASCII = """
% time
% frequency
% duration
% bandwidth
% normalizedEnergy
1023927596.843750000 1.1766601562500000e+01 1.8193577533403701e-01 5.4964450953309427e+00 2.7199652890340332e+01
1023927378.250000000 2.3637695312500000e+00 9.0565757364173183e-01 1.1041700849239395e+00 2.1194302085268539e+01
1023927597.187500000 1.0293457031250000e+01 2.0797345069017714e-01 4.8083060442639054e+00 1.7678989962812786e+01
1023927433.250000000 2.3637695312500000e+00 9.0565757364173183e-01 1.1041700849239395e+00 1.6883206146877814e+01
""".strip()  # noqa: E501


@pytest.fixture
def omega_ascii(tmp_path: Path) -> Path:
    """Create a temporary file with example Omega ascii data."""
    dat = tmp_path / "EVENTS.txt"
    dat.write_text(OMEGA_ASCII)
    return dat


def test_read_omega_ascii(omega_ascii):
    """Check that the ``"ascii.omega"`` file reader works."""
    tab = EventTable.read(omega_ascii, format="ascii.omega")
    assert len(tab) == 4
    assert tab[0]["time"] == pytest.approx(1023927596.84375)
    assert tab[3]["normalizedEnergy"] == pytest.approx(16.8832)


def test_read_omega_ascii_noheader(tmp_path):
    """Check that reading an Omega file without headers results in an error."""
    # create empty file
    empty = tmp_path / "empty.txt"
    empty.write_text("")

    # read it back
    with pytest.raises(
        ValueError,
        match="No column names found in Omega header",
    ):
        EventTable.read(empty, format="ascii.omega")


def test_write_omega_ascii(omega_ascii):
    """Check that the ``"ascii.omega"`` file writer works."""
    # read the example data
    tab = EventTable.read(omega_ascii, format="ascii.omega")

    # write the table out in ascii.omega format
    out = omega_ascii.parent / "write.txt"
    tab.write(out, format="ascii.omega")

    # then read it from the written file and compare
    t2 = EventTable.read(out, format="ascii.omega")

    assert_table_equal(t2, tab)
