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

"""Tests for :mod:`gwpy.table.io.root`."""

import pytest

from ...segments import (
    Segment,
    SegmentList,
)
from ...testing.utils import assert_table_equal
from .. import filters
from ..filter import filter_table
from .utils import (
    TABLE_CLASSES,
    random_table,
)

# -- test config ---------------------

def pytest_generate_tests(metafunc):
    """Parametrize the ``table`` fixture for each valid table type."""
    if "table" in metafunc.fixturenames:
        metafunc.parametrize("table", TABLE_CLASSES, indirect=True)


@pytest.fixture
def table(request):
    """Create a random table with the right columns."""
    return random_table(
        length=100,
        names=["time", "frequency", "snr"],
        table_class=request.param,
    )


@pytest.fixture
def root_file(table, tmp_path):
    """Write the ``table`` to a (temporary) file and return the path.

    Enables read tests.
    """
    tmp = tmp_path / "table.root"
    table.write(tmp)
    return tmp


# -- tests ---------------------------

@pytest.mark.requires("uproot")
def test_read_write_root(table, tmp_path):
    """Test `Table.read` and `Table.write` with `format='root'`.

    Default options.
    """
    tmp = tmp_path / "table.root"

    # check write
    table.write(tmp)

    # check read gives back same table
    t2 = type(table).read(tmp)
    assert_table_equal(t2, table)


@pytest.mark.requires("uproot")
def test_read_root_where(table, root_file):
    """Test `Table.read(format='root')` with ``where``."""
    # test where works
    segs = SegmentList([Segment(100, 200), Segment(400, 500)])
    t2 = type(table).read(
        root_file,
        where=[
            "200 < frequency < 500",
            ("time", filters.in_segmentlist, segs),
        ],
    )
    assert_table_equal(
        t2,
        filter_table(
            table,
            "frequency > 200",
            "frequency < 500",
            ("time", filters.in_segmentlist, segs),
        ),
    )


@pytest.mark.requires("uproot")
def test_write_root_overwrite(table, tmp_path):
    """Test `Table.write(format='root')` with ``overwrite=True``."""
    tmp = tmp_path / "table.root"
    table.write(tmp)

    # assert failure with overwrite=False (default)
    with pytest.raises(OSError, match="path exists and refusing to overwrite"):
        table.write(tmp)

    # assert works with overwrite=True
    table.write(tmp, overwrite=True)


@pytest.mark.requires("uproot")
def test_write_root_append(table, tmp_path):
    """Test `Table.write(format='root')` with ``append=True``."""
    # write once
    tmp = tmp_path / "table.root"
    table.write(tmp, treename="a")

    # write a second tree
    table.write(tmp, treename="b", append=True)

    # check that we can't read without specifying a tree
    with pytest.raises(
        ValueError,
        match=r"^Multiple trees found",
    ):
        type(table).read(tmp)

    # check that we can read both trees
    type(table).read(tmp, treename="a")
    type(table).read(tmp, treename="b")


@pytest.mark.requires("uproot")
def test_write_root_append_not_found(table, tmp_path):
    """Test `Table.write(format='root', append=True)` when the source doesn't exist."""
    tmp = tmp_path / "table.root"
    with pytest.raises(OSError, match="could not read"):
        table.write(tmp, treename="a", append=True)


@pytest.mark.requires("uproot")
def test_write_root_append_overwrite(table, tmp_path):
    """Test `Table.write(format='root', append=True, overwrite=True)`."""
    # write once
    tmp = tmp_path / "table.root"
    table.write(tmp, treename="a")

    # write a second tree
    table.write(tmp, treename="b", append=True, overwrite=True)

    # check that we can read the new tree
    type(table).read(tmp, treename="b")

    # if overwriting, check that we can't read the old tree any more
    with pytest.raises(KeyError):
        type(table).read(tmp, treename="a")
