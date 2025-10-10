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

"""Tests for :mod:`gwpy.table.io.hacr`."""

from unittest import mock

import numpy
import pytest
from astropy.table import Table

from ...testing.utils import assert_table_equal
from ..filter import filter_table
from ..io import hacr as io_hacr
from ..table import EventTable
from .utils import random_table

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

requires_db_libraries = pytest.mark.requires(
    "sqlalchemy",
    "pandas",
    "pymysql",
)


@pytest.fixture(scope="module")
def hacr_table():
    """Create a table of HACR-like data."""
    names, types = zip(*{  # correct as of Jan 2025 (DMM)
        "refId": "int64",
        "freq_central": "float64",
        "bandwidth": "float64",
        "duration": "float64",
        "num_pixels": "int64",
        "snr": "float64",
        "pksnr": "float64",
        "cluster_type": "U4",
        "cluster_density": "float64",
        "segmentNumber": "int64",
        "maxPower": "float64",
        "totPower": "float64",
        "veto": "int64",
    }.items(), strict=False)
    table = random_table(
        length=100,
        names=names,
        dtypes=list(types),
    )

    # add timing information
    table.add_column(
        numpy.linspace(0, 10, num=len(table), dtype="int64", endpoint=False),
        name="gps_start",
    )
    table.add_column(
        numpy.tile(
            numpy.linspace(0, 1, num=10, dtype="float64", endpoint=False),
            10,
        ),
        name="gps_offset",
    )

    # add process_id with ints
    table.add_column(
        numpy.linspace(0, 10, num=len(table), dtype="int64", endpoint=False),
        name="process_id",
    )

    return table


@pytest.fixture(scope="module")
def hacr_process_table():
    """Create a HACR process table."""
    return Table(
        rows=[
            (0, "X1:HACR-1", 0, 1, "chacr"),
            (1, "X1:HACR-1", 1, 2, "chacr"),
            (2, "X1:HACR-1", 2, 3, "chacr"),
            (3, "X1:HACR-1", 3, 4, "chacr"),
            (4, "X1:HACR-1", 4, 5, "chacr"),
            (5, "X1:HACR-1", 5, 6, "chacr"),
            (6, "X1:HACR-1", 6, 7, "chacr"),
            (7, "X1:HACR-1", 7, 8, "chacr"),
            (8, "X1:HACR-1", 8, 9, "chacr"),
            (9, "X1:HACR-2", 9, 10, "chacr"),
            (10, "X1:HACR-3", 9, 10, "something else"),
            (11, "X1:HACR-3", 9, 10, "something else"),
        ],
        names=(
            "process_id",
            "channel",
            "gps_start",
            "gps_stop",
            "monitorName",
        ),
    )


@pytest.fixture(scope="module")
def hacr_sqlite(hacr_table, hacr_process_table, tmp_path_factory):
    """Create a HACR database in SQLite."""
    from sqlalchemy import create_engine

    db = tmp_path_factory.mktemp("hacr") / "hacr.sqlite"
    dburl = f"sqlite:///{db}"

    engine = create_engine(dburl)
    hacr_table.to_pandas().to_sql("mhacr", engine)
    hacr_process_table.to_pandas().to_sql("job", engine)
    engine.dispose()
    return dburl


@pytest.fixture(scope="module")
def hacr_engine(hacr_sqlite):
    """Create an `sqlalchemy.Engine` connected to the HACR SQLite database."""
    from sqlalchemy import create_engine

    return create_engine(hacr_sqlite)


@pytest.mark.parametrize(("start", "end", "result"), [
    ("Jan 1 2024", "Jan 1 2024 00:01", ["geo202401"]),
    ("Jan 1 2024", "Mar 1 2024", ["geo202401", "geo202402"]),
])
def test_get_database_names(start, end, result):
    """Test `get_database_names`."""
    assert io_hacr.get_database_names(start, end) == result


@requires_db_libraries
def test_get_hacr_channels(hacr_engine):
    """Test `get_hacr_channels`."""
    with mock.patch(
        "gwpy.table.io.hacr.create_engine",
        return_value=hacr_engine,
    ):
        assert set(io_hacr.get_hacr_channels()) == {
            "X1:HACR-1",
            "X1:HACR-2",
        }


@requires_db_libraries
def test_fetch_hacr(hacr_table, hacr_engine):
    """Test `EventTable.fetch(source='hacr')` with basic options."""
    t2 = EventTable.fetch(
        source="hacr",
        engine=hacr_engine,
        index_col="index",  # test sqlite database injects an index
    )

    # check response is correct
    assert_table_equal(t2, hacr_table)


@requires_db_libraries
def test_fetch_hacr_columns_where(hacr_table, hacr_engine):
    """Test that `EventTable.fetch(source='hacr')` works with complex queries."""
    with mock.patch(
        "gwpy.table.io.hacr.create_engine",
        return_value=hacr_engine,
    ):
        t2 = EventTable.fetch(
            source="hacr",
            channel="X1:HACR-1",
            start=4,
            end=6,
            columns=[
                "bandwidth",
                "snr",
            ],
            where="freq_central>500",
        )

    expect = filter_table(
        hacr_table,
        "freq_central>500",
        "gps_start >= 4",
        "gps_start < 6",
    )["bandwidth", "snr"]

    assert_table_equal(t2, expect)
