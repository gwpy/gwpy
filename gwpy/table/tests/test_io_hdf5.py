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

"""Test of `Table.read` and `Table.write` with ``format='hdf5'``."""

import pytest

from ...testing.utils import assert_table_equal
from ..filter import filter_table
from .utils import (
    TABLE_CLASSES,
    random_table,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


@pytest.mark.parametrize("table_class", TABLE_CLASSES[1:])
def test_read_write_hdf5(table_class, tmp_path):
    """Test `EventTable.read` and `EventTable.write` with ``format='hdf5'``."""
    table = random_table(
        names=["time", "frequency", "snr"],
        table_class=table_class,
    )

    # check that we can write a table and read it back
    tmp = tmp_path / "table.h5"
    table.write(tmp, path="/data")
    t2 = type(table).read(tmp, path="/data")
    assert_table_equal(t2, table)

    # check that we can read with columns and where
    t2 = type(table).read(
        tmp,
        path="/data",
        where="frequency>500",
        columns=["time", "snr"],
    )
    assert_table_equal(
        t2,
        filter_table(table, "frequency>500")[("time", "snr")],
    )
