# Copyright (c) 2020-2025 Cardiff University
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

"""Tests for :mod:`gwpy.table.io.ligolw`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest
from astropy.table import (
    Table,
    vstack,
)
from numpy.testing import (
    assert_allclose,
    assert_array_equal,
)

from ...testing import utils
from ...testing.errors import pytest_skip_network_error
from .. import EventTable
from .utils import random_table

if TYPE_CHECKING:
    from igwn_ligolw.lsctables import SnglBurstTable

TEST_XML_PATH = utils.TEST_DATA_PATH / "H1-LDAS_STRAIN-968654552-10.xml.gz"
REMOTE_XML_FILE = (
    "https://gitlab.com/gwpy/gwpy/-/raw/v3.0.10/gwpy/testing/data/"
    + TEST_XML_PATH.name
)


@pytest.fixture
def table() -> Table:
    """Create a table to use in testing."""
    return random_table(
        names=["peak_time", "peak_time_ns", "snr", "central_freq"],
        dtypes=["i4", "i4", "f4", "f4"],
        length=100,
    )


@pytest.fixture
def llwtable() -> SnglBurstTable:
    """Create a LIGO_LW table to use in testing."""
    from igwn_ligolw.lsctables import SnglBurstTable
    llwtab = SnglBurstTable.new(columns=["peak_frequency", "snr"])
    for i in range(10):
        row = llwtab.RowType()
        row.peak_frequency = float(i)
        row.snr = float(i)
        llwtab.append(row)
    return llwtab


# -- conversions ---------------------

@pytest.mark.requires("igwn_ligolw.lsctables")
def test_to_astropy_table(llwtable):
    """Test converting a LIGO_LW table to an Astropy Table."""
    tab = EventTable(llwtable)
    assert set(tab.colnames) == {"peak_frequency", "snr"}
    assert_array_equal(tab["snr"], llwtable.getColumnByName("snr"))


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_to_astropy_table_rename(llwtable):
    """Test converting a LIGO_LW table to an Astropy Table with renaming."""
    tab = EventTable(llwtable, rename={"peak_frequency": "frequency"})
    assert set(tab.colnames) == {"frequency", "snr"}
    assert_array_equal(
        tab["frequency"],
        llwtable.getColumnByName("peak_frequency"),
    )


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_to_astropy_table_empty():
    """Test converting an empty LIGO_LW table to an Astropy Table."""
    from igwn_ligolw.lsctables import SnglBurstTable
    llwtable = SnglBurstTable.new(
        columns=["peak_time", "peak_time_ns", "ifo"],
    )
    tab = EventTable(llwtable, columns=["peak", "ifo"])
    assert set(tab.colnames) == {"peak", "ifo"}
    assert tab["peak"].dtype.type is numpy.object_
    assert tab["ifo"].dtype.type is numpy.str_


# -- i/o -----------------------------

@pytest.mark.parametrize("ext", ["xml", "xml.gz"])
@pytest.mark.parametrize("table_class", [Table, EventTable])
@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_write_ligolw(table, ext, table_class):
    """Test `Table.read` and `Table.write` with ``format='ligolw'``."""
    table = table_class(table)
    utils.test_read_write(
        table,
        "ligolw",
        ext,
        write_kw={
            "tablename": "sngl_burst",
        },
        autoidentify=False,
        assert_equal=utils.assert_table_equal,
        assert_kw={
            "almost_equal": True,
        },
    )


@pytest.mark.parametrize("use_numpy_dtypes", [False, True])
@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_write_ligolw_types(use_numpy_dtypes):
    """Test `Table.read(format='ligolw')` with ``use_numpy_dtypes``."""
    t2 = Table.read(
        TEST_XML_PATH,
        tablename="sngl_burst",
        columns=["peak"],
        use_numpy_dtypes=use_numpy_dtypes,
    )
    peak = t2["peak"][0]
    if use_numpy_dtypes:
        assert isinstance(peak, float)
    else:
        from igwn_ligolw.lsctables import LIGOTimeGPS as LigolwGps
        assert isinstance(peak, LigolwGps)


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_ligolw_multiple():
    """Test `Table.read(format='ligolw')` with multiple files."""
    t = EventTable.read(
        TEST_XML_PATH,
        tablename="sngl_burst",
    )
    t2 = EventTable.read(
        [TEST_XML_PATH, TEST_XML_PATH],
        tablename="sngl_burst",
    )
    utils.assert_table_equal(t2, vstack((t, t)))


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_write_ligolw_overwrite(table, tmp_path):
    """Test `Table.write(format='ligolw', overwrite=True)` overwrites a table."""
    # write the table once
    xml = tmp_path / "test.xml"
    table.write(xml, format="ligolw", tablename="sngl_burst")

    # check that writing again to the same file fails
    with pytest.raises(
        OSError,
        match=f"^File exists: {xml}$",
    ):
        table.write(xml, format="ligolw", tablename="sngl_burst")

    # but overwrite=True forces the new write
    table.write(
        xml,
        format="ligolw",
        tablename="sngl_burst",
        overwrite=True,
    )

    # check that the table is the same
    utils.assert_table_equal(
        type(table).read(xml, format="ligolw"),
        table,
        almost_equal=True,
    )


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_write_ligolw_append(table, tmp_path):
    """Test `Table.write(format='ligolw', append=True)` extends a table."""
    # write the table once
    xml = tmp_path / "test.xml"
    table.write(xml, format="ligolw", tablename="sngl_burst")

    # test that append=True extends the table in-place
    table.write(xml, format="ligolw", tablename="sngl_burst", append=True)
    utils.assert_table_equal(
        type(table).read(xml, format="ligolw"),
        vstack((table, table)),
        almost_equal=True,
    )


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_write_ligolw_append_multiple_tables(table, tmp_path):
    """Test `Table.write(format='ligolw', append=True)` with different tables."""
    # write the first table
    xml = tmp_path / "test.xml"
    table.write(xml, format="ligolw", tablename="sngl_burst")

    # write another table into the same file
    insp = random_table(
        names=["end_time", "snr", "chisq_dof"],
        length=10,
    )
    insp.write(
        xml,
        format="ligolw",
        tablename="sngl_inspiral",
        append=True,
    )

    # check that we can get back the first, but not have to specify tablename
    with pytest.raises(
        ValueError,
        match="Multiple tables found in LIGO_LW document",
    ):
        type(table).read(xml, format="ligolw")
    utils.assert_table_equal(
        type(table).read(xml, format="ligolw", tablename="sngl_burst"),
        table,
        almost_equal=True,
    )


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_write_ligolw_property_columns(tmp_path):
    """Test reading/writing LIGO_LW files handles property columns properlty.

    As below the 'peak' column is a `gpsproperty` made by dynamically combining
    the ``'peak_time'`` and ``'peak_time_ns'`` columns.
    This test checks that the writer handles this silently, and that the reader
    can handle it.
    """
    # create table that uses a property column ('peak')
    table = random_table(
        names=["peak", "snr", "central_freq"],
        dtypes=["f8", "f4", "f4"],
        length=100,
    )

    # write table to file
    xml = tmp_path / "test.xml"
    table.write(
        xml,
        format="ligolw",
        tablename="sngl_burst",
    )

    # read it back and check that the gpsproperty was unpacked properly
    t2 = type(table).read(
        xml,
        format="ligolw",
        tablename="sngl_burst",
    )
    assert set(t2.colnames) == {
        "peak_time",
        "peak_time_ns",
        "snr",
        "central_freq",
    }
    assert_allclose(
        t2["peak_time"] + t2["peak_time_ns"] * 1e-9,
        table["peak"],
    )

    # test that reading property columns directly works.
    t3 = type(table).read(
        xml,
        format="ligolw",
        tablename="sngl_burst",
        columns=["peak", "peak_time"],
    )
    assert set(t3.colnames) == {"peak", "peak_time"}
    assert_array_equal(
        t3["peak"],
        table["peak"],
    )


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_ligolw_get_as_exclude(tmp_path):
    """Test that reading LIGO_LW files handles columns whose names don't
    correspond to the `get_{name}` method on the parent table.

    E.g. the ``'time_slide'`` table has a column 'time_slide_id' and a
    method `~igwn_ligolw.lsctables.TimeSlideTable.get_time_slide_id` that have
    nothing to do with each other.
    """  # noqa: D205
    # create a time_slide table with a 'time_slide_id' column
    table = Table(
        rows=[
            ("H1", 0.0, 4, 0),
            ("L1", 0.62831, 4, 0),
            ("V1", 0.31415, 4, 0),
        ],
        names=("instrument", "offset", "process_id", "time_slide_id"),
    )

    # write it
    xml = tmp_path / "test.xml"
    table.write(
        xml,
        format="ligolw",
        tablename="time_slide",
    )

    # read it back and check that the `time_slide_id` column is read properly
    t2 = type(table).read(
        xml,
        tablename="time_slide",
    )
    t2.sort("instrument")
    utils.assert_table_equal(t2, table)


@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_process_table():
    """Regression test against gwpy/gwpy#1367."""
    from igwn_ligolw.lsctables import ProcessTable
    llwtable = ProcessTable.new(
        columns=["ifos", "username"],
    )
    llwtable.appendRow(instruments=("G1", "H1"), username="testuser")
    llwtable.appendRow(instruments=("H1", "L1"), username="testuser")
    tab = EventTable(llwtable)
    assert len(tab) == 2
    assert tab[0]["ifos"] == "G1,H1"


@pytest_skip_network_error
@pytest.mark.requires("igwn_ligolw.lsctables")
def test_read_remote_file():
    """Test that we can read remote files over HTTP."""
    tab = EventTable.read(
        REMOTE_XML_FILE,
        cache=False,
        format="ligolw",
        tablename="sngl_burst",
    )
    assert len(tab) == 2052
    assert tab[0]["snr"] == pytest.approx(0.69409615)
