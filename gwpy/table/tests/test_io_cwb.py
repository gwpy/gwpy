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

"""Tests for :mod:`gwpy.table.io.cwb`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from gwpy.table import EventTable

if TYPE_CHECKING:
    from pathlib import Path

    from astropy.table import Table

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


CWB_ASCII = """
# correlation threshold = 0.000000
# network rho threshold = 0.000000
# -/+ - not passed/passed final selection cuts
#  1 - effective correlated amplitude rho
#  2 - correlation coefficient 0/1
#  3 - event norm
#  4 - event DoF
#  5 - correlated amplitude
#  6 - time shift
#  7 - time super shift
#  8 - likelihood
#  9 - normalized chi2
# 10 - cross-power statistic
# 11 - central frequency
# 12 - bandwidth
# 13 - duration
# 14 - number of pixels
# 15 - frequency resolution
# 16 - cwb run number
# 17 - time for L1 detector
# 18 - time for H1 detector
# 19 - sSNR for L1 detector
# 20 - sSNR for H1 detector
# 21 - hrss for L1 detector
# 22 - hrss for H1 detector
# 23 - PHI
# 24 - THETA
# 25 - PSI
+ - 72.5067 0.990 0.965 67.908 72.59   0.000       0 5.3e+03 0.980 152.705  123   67 0.023 1292   0     1 1420000000.2036 1420000000.2059 2.8e+03 2.5e+03 1.9e-22 1.7e-22 300.26 43.43 21.36
+ - 18.1662 0.828 0.723 81.677 19.00   0.000       0 4.1e+02 0.966 26.177  109   66 0.172 1140   0     1 1420000001.1299 1420000001.1361 2.5e+02 1.6e+02 6.2e-23 4.8e-23 274.22 91.79 -74.38
+ - 13.7103 0.940 0.842 31.190 13.73   0.000       0 2.0e+02 0.882 21.139  109   46 0.017 528   0     1 1420000002.7594 1420000002.7623 1.2e+02 8.5e+01 3.7e-23 3.2e-23 266.48 66.68 -79.10
+ - 13.2168 0.908 0.900 30.457 13.49   0.000       0 1.9e+02 1.034 21.953  104   44 0.019 517   0     1 1420000003.2002 1420000003.2079 1.2e+02 7.4e+01 3.7e-23 2.9e-23 276.33 115.94 -78.07
""".strip()  # noqa: E501


@pytest.fixture
def cwb_ascii(tmp_path: Path) -> Path:
    """Create a temporary cWB ASCII file."""
    dat = tmp_path / "EVENTS.txt"
    dat.write_text(CWB_ASCII)
    return dat


@pytest.fixture
def cwb_root(cwb_ascii: Path) -> Path:
    """Create a temporary cWB ROOT file from the ASCII file."""
    tab = EventTable.read(
        cwb_ascii,
        columns=["central frequency", "PHI", "THETA"],
        format="ascii.cwb",
    )
    rootf = cwb_ascii.parent / "cwb.root"
    tab.write(rootf, format="root", treename="waveburst")
    return rootf


def _check_cwb_table(table: Table):
    """Check that a cWB table has been read correctly."""
    assert len(table) == 4
    assert table[0]["central frequency"] == pytest.approx(123)
    assert table[3]["PHI"] == pytest.approx(276.33)


def test_read_cwb_ascii(cwb_ascii: Path):
    """Check that the ``"ascii.cwb"`` file reader works."""
    tab = EventTable.read(cwb_ascii, format="ascii.cwb")
    _check_cwb_table(tab)


def test_read_cwb_ascii_columns_where(cwb_ascii: Path):
    """Check that the ``"ascii.cwb"`` file reader works."""
    columns = ["central frequency", "bandwidth", "duration"]
    tab = EventTable.read(
        cwb_ascii,
        format="ascii.cwb",
        columns=columns,
        where=["PSI > 0"],
    )
    assert set(tab.colnames) == set(columns)
    assert list(tab["bandwidth"]) == [67]  # only one event with PSI > 0


@pytest.mark.requires("uproot")
def test_read_cwb_root(cwb_root: Path):
    """Check that the ``"ascii.root"`` file reader works."""
    tab = EventTable.read(cwb_root, format="root.cwb")
    _check_cwb_table(tab)
