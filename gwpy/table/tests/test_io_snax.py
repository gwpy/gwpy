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

from contextlib import nullcontext

import h5py
import pytest

from ...testing.utils import assert_table_equal
from ..filter import filter_table
from ..table import EventTable
from .utils import random_table

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


@pytest.fixture
def snaxtable():
    """Create a table of SNAX-like data."""
    snax = random_table(
        names=[
            "time",
            "snr",
            "frequency",
        ],
        length=100,
    )
    snax["channel"] = "X1:SNAX"
    return snax


@pytest.fixture
def snaxfile(snaxtable, tmp_path):
    """Create a temporary file with example SNAX HDF5 data."""
    tmp = tmp_path / "SNAX-0-0.h5"
    channel = snaxtable[0]["channel"]
    tmptable = snaxtable.copy()
    tmptable.remove_column("channel")
    with h5py.File(tmp, "w") as h5f:
        group = h5f.create_group(channel)
        group.create_dataset(data=tmptable, name="0.0_20.0")
    return tmp


def test_read_snax(snaxtable, snaxfile):
    """Check that we can read a SNAX-format HDF5 file."""
    table = EventTable.read(snaxfile, format="hdf5.snax")
    assert_table_equal(snaxtable, table)


def test_read_snax_channel(snaxtable, snaxfile):
    """Check that we can read a SNAX-format HDF5 file specifying the channel."""
    table = EventTable.read(
        snaxfile,
        format="hdf5.snax",
        channels="X1:SNAX",
    )
    assert_table_equal(snaxtable, table)


def test_read_snax_columns_where(snaxtable, snaxfile):
    """Check that ``where`` and ``columns`` work when reading from a SNAX file."""
    # test with where and columns
    table = EventTable.read(
        snaxfile,
        channels="X1:SNAX",
        format="hdf5.snax",
        where="snr>.5",
        columns=("time", "snr"),
    )
    assert_table_equal(
        table,
        filter_table(snaxtable, "snr>.5")[("time", "snr")],
    )


def test_read_snax_compact(snaxtable, snaxfile):
    """Check that ``columns`` and ``where`` work when reading from a SNAX file."""
    # test compact representation of channel column
    t2 = EventTable.read(snaxfile, compact=True, format="hdf5.snax")

    # group by channel and drop channel column
    tables = {}
    t2 = t2.group_by("channel")
    t2.remove_column("channel")
    for key, group in zip(t2.groups.keys, t2.groups, strict=True):
        channel = t2.meta["channel_map"][key["channel"]]
        tables[channel] = EventTable(group, copy=True)

    # verify table groups are identical
    t_ref = snaxtable.copy().group_by("channel")
    t_ref.remove_column("channel")
    for key, group in zip(t_ref.groups.keys, t_ref.groups, strict=True):
        channel = key["channel"]
        assert_table_equal(group, tables[channel])


_missing_msg = "requested channels not found in SNAX file: 'X1:MISSING'"

@pytest.mark.parametrize(("on_missing", "ctx"), [
    pytest.param(
        "error",
        pytest.raises(ValueError, match=_missing_msg),
        id="error",
    ),
    pytest.param(
        "warn",
        pytest.warns(UserWarning, match=_missing_msg),
        id="warn",
    ),
    pytest.param(
        "ignore",
        nullcontext(),
        id="ignore",
    ),
])
def test_read_snax_errors(on_missing, ctx, snaxtable, snaxfile):
    """Check error handling when reading from a SNAX-format file."""
    missing = ["X1:SNAX", "X1:MISSING"]
    with ctx:
        table = EventTable.read(
            snaxfile,
            channels=missing,
            format="hdf5.snax",
            on_missing=on_missing,
        )
    if on_missing != "error":
        assert_table_equal(snaxtable, table)
