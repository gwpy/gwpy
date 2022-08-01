# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2020)
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

"""Tests for :mod:`gwpy.table.io.ligolw`
"""

import pytest

import numpy
from numpy.testing import assert_array_equal

from .. import EventTable


# -- fixtures -----------------------------------------------------------------

@pytest.fixture()
def llwtable():
    from ligo.lw.lsctables import (New, SnglBurstTable)
    llwtab = New(SnglBurstTable, columns=["peak_frequency", "snr"])
    for i in range(10):
        row = llwtab.RowType()
        row.peak_frequency = float(i)
        row.snr = float(i)
        llwtab.append(row)
    return llwtab


# -- test to_astropy_table() via EventTable conversions -----------------------

@pytest.mark.requires("ligo.lw.lsctables")
def test_to_astropy_table(llwtable):
    tab = EventTable(llwtable)
    assert set(tab.colnames) == {"peak_frequency", "snr"}
    assert_array_equal(tab["snr"], llwtable.getColumnByName("snr"))


@pytest.mark.requires("ligo.lw.lsctables")
def test_to_astropy_table_rename(llwtable):
    tab = EventTable(llwtable, rename={"peak_frequency": "frequency"})
    assert set(tab.colnames) == {"frequency", "snr"}
    assert_array_equal(
        tab["frequency"],
        llwtable.getColumnByName("peak_frequency"),
    )


@pytest.mark.requires("ligo.lw.lsctables")
def test_to_astropy_table_empty():
    from ligo.lw.lsctables import (New, SnglBurstTable)
    llwtable = New(
        SnglBurstTable,
        columns=["peak_time", "peak_time_ns", "ifo"],
    )
    tab = EventTable(llwtable, columns=["peak", "ifo"])
    assert set(tab.colnames) == {"peak", "ifo"}
    assert tab['peak'].dtype.type is numpy.object_
    assert tab['ifo'].dtype.type is numpy.unicode_


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_process_table():
    """Regression test against gwpy/gwpy#1367
    """
    from ligo.lw.lsctables import (New, ProcessTable)
    llwtable = New(
        ProcessTable,
        columns=["ifos", "username"],
    )
    llwtable.appendRow(instruments=("G1", "H1"), username="testuser")
    llwtable.appendRow(instruments=("H1", "L1"), username="testuser")
    tab = EventTable(llwtable)
    assert len(tab) == 2
    assert tab[0]["ifos"] == "G1,H1"
