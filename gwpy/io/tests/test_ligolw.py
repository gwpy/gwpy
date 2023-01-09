# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Unit test for `gwpy.io.ligolw` module
"""

import tempfile
from pathlib import Path

import pytest

import numpy

from .. import ligolw as io_ligolw

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OLD_FORMAT_LIGO_LW_XML = """
<?xml version='1.0' encoding='utf-8'?>
<!DOCTYPE LIGO_LW SYSTEM "http://ldas-sw.ligo.caltech.edu/doc/ligolwAPI/html/ligolw_dtd.txt">
<LIGO_LW>
  <Table Name="sngl_burst:table">
    <Column Type="lstring" Name="sngl_burst:ifo"/>
    <Column Type="int_4s" Name="sngl_burst:peak_time"/>
    <Column Type="int_4s" Name="sngl_burst:peak_time_ns"/>
    <Column Type="int_4s" Name="sngl_burst:start_time"/>
    <Column Type="int_4s" Name="sngl_burst:start_time_ns"/>
    <Column Type="real_4" Name="sngl_burst:duration"/>
    <Column Type="lstring" Name="sngl_burst:search"/>
    <Column Type="ilwd:char" Name="sngl_burst:event_id"/>
    <Column Type="ilwd:char" Name="sngl_burst:process_id"/>
    <Column Type="real_4" Name="sngl_burst:central_freq"/>
    <Column Type="lstring" Name="sngl_burst:channel"/>
    <Column Type="real_4" Name="sngl_burst:amplitude"/>
    <Column Type="real_4" Name="sngl_burst:snr"/>
    <Column Type="real_4" Name="sngl_burst:confidence"/>
    <Column Type="real_8" Name="sngl_burst:chisq"/>
    <Column Type="real_8" Name="sngl_burst:chisq_dof"/>
    <Column Type="real_4" Name="sngl_burst:bandwidth"/>
    <Stream Delimiter="," Type="Local" Name="sngl_burst:table">
      "H1",100000000,000000000,100000000,000000000,1,"gstlal_excesspower","sngl_burst:event_id:4696","process:process_id:0",162,"LDAS-STRAIN",2.3497068e-25,0.69409615,16.811825,0,512,256,
      "H1",100000000,000000001,100000000,000000001,1,"gstlal_excesspower","sngl_burst:event_id:4697","process:process_id:0",162,"LDAS-STRAIN",2.3497797e-25,0.69416249,16.816761,0,512,256,
      "H1",100000000,000000001,100000000,000000002,1,"gstlal_excesspower","sngl_burst:event_id:4698","process:process_id:0",162,"LDAS-STRAIN",2.3479907e-25,0.69253588,16.696106,0,512,256,
    </Stream>
  </Table>
</LIGO_LW>
""".strip()  # noqa: E501


# -- fixtures -----------------------------------------------------------------

@pytest.fixture(scope="function")
def llwdoc():
    """Build an empty LIGO_LW Document
    """
    try:
        from ligo.lw.ligolw import (Document, LIGO_LW)
    except ImportError as exc:
        pytest.skip(str(exc))
    xmldoc = Document()
    xmldoc.appendChild(LIGO_LW())
    return xmldoc


def new_table(tablename, data=None, **new_kw):
    """Create a new LIGO_LW Table with data
    """
    try:
        from ligo.lw import lsctables
    except ImportError as exc:
        pytest.skip(str(exc))
    from ligo.lw.table import Table

    table = lsctables.New(
        lsctables.TableByName[Table.TableName(tablename)],
        **new_kw,
    )
    for dat in data or list():
        row = table.RowType()
        for key, val in dat.items():
            setattr(row, key, val)
        table.append(row)
    return table


@pytest.fixture(scope="function")
def llwdoc_with_tables(llwdoc):
    """Build a LIGO_LW Document with some tables
    """
    llw = llwdoc.childNodes[-1]  # get ligolw element
    for t in [
        new_table('process'),
        new_table('sngl_ringdown'),
    ]:
        llw.appendChild(t)
    return llwdoc


# -- tests --------------------------------------------------------------------

def test_read_table(llwdoc_with_tables):
    tab = io_ligolw.read_table(llwdoc_with_tables, tablename="process")
    assert tab is llwdoc_with_tables.childNodes[0].childNodes[0]


def test_read_table_empty(llwdoc):
    with pytest.raises(ValueError) as exc:
        io_ligolw.read_table(llwdoc)
    assert str(exc.value) == "No tables found in LIGO_LW document(s)"


@pytest.mark.requires("ligo.lw.lsctables")
def test_read_table_ilwd(tmp_path):
    xmlpath = tmp_path / "test.xml"
    with open(xmlpath, "w") as f:
        f.write(OLD_FORMAT_LIGO_LW_XML)
    tab = io_ligolw.read_table(xmlpath, tablename="sngl_burst")
    assert len(tab) == 3


def test_read_table_multiple(llwdoc_with_tables):
    # check that read_table correctly errors on ambiguity
    with pytest.raises(ValueError) as exc:
        io_ligolw.read_table(llwdoc_with_tables)
    assert str(exc.value).startswith("Multiple tables")
    assert "'process', 'sngl_ringdown'" in str(exc.value)


def test_open_xmldoc(tmp_path, llwdoc_with_tables):
    tmp = tmp_path / "test.xml"
    # write a LIGO_LW file
    with open(tmp, "w") as fobj:
        llwdoc_with_tables.write(fobj)
    # and check that we can read it again (from Path, str, and file)
    for obj in (tmp, str(tmp), open(tmp, "rb")):
        copy = io_ligolw.open_xmldoc(obj)
        # and that the contents are the same
        assert (
            copy.childNodes[0].childNodes
            == llwdoc_with_tables.childNodes[0].childNodes
        )


def test_open_xmldoc_new(tmp_path, llwdoc):
    from ligo.lw.ligolw import Document
    new = io_ligolw.open_xmldoc(tmp_path / "new.xml")
    assert isinstance(new, Document)
    assert not new.childNodes  # empty


def test_get_ligolw_element(llwdoc):
    llw = llwdoc.childNodes[0]
    assert io_ligolw.get_ligolw_element(llw) is llw
    assert io_ligolw.get_ligolw_element(llwdoc) is llw


def test_get_ligolw_element_error(llwdoc):
    # check that blank document raises an error
    with pytest.raises(ValueError):
        io_ligolw.get_ligolw_element(type(llwdoc)())


def test_iter_tables(llwdoc_with_tables):
    expected = llwdoc_with_tables.childNodes[0].childNodes
    assert list(io_ligolw.iter_tables(llwdoc_with_tables)) == expected


def test_list_tables(llwdoc_with_tables):
    names = [
        t.TableName(t.Name)
        for t in llwdoc_with_tables.childNodes[0].childNodes
    ]

    # check that tables are listed properly
    assert io_ligolw.list_tables(llwdoc_with_tables) == names


def test_list_tables_file(llwdoc_with_tables):
    # check that we can list from files
    names = [
        t.TableName(t.Name)
        for t in llwdoc_with_tables.childNodes[0].childNodes
    ]
    with tempfile.NamedTemporaryFile(mode='w') as f:
        llwdoc_with_tables.write(f)
        f.seek(0)
        assert io_ligolw.list_tables(f) == names


@pytest.mark.requires("ligo.lw.lsctables")
@pytest.mark.parametrize('value, name, result', [
    (None, 'peak_time', None),
    (1.0, 'peak_time', numpy.int32(1)),
    (1, 'process_id', 1),
    (1.0, 'invalidname', 1.0),
])
def test_to_table_type(value, name, result):
    from ligo.lw.lsctables import SnglBurstTable
    out = io_ligolw.to_table_type(value, SnglBurstTable, name)
    assert isinstance(out, type(result))
    assert out == result


def test_write_tables_to_document(llwdoc_with_tables):
    # create new table
    def _new():
        return new_table(
            'segment',
            [{'segment': (1, 2)}, {'segment': (3, 4)}, {'segment': (5, 6)}],
            columns=('start_time', 'start_time_ns', 'end_time', 'end_time_ns'),
        )

    # get ligolw element
    llw = llwdoc_with_tables.childNodes[-1]

    # check we can add a new table
    tab = _new()
    io_ligolw.write_tables_to_document(llwdoc_with_tables, [tab])
    assert llw.childNodes[-1] is tab
    assert len(tab) == 3

    # check that adding to an existing table extends
    io_ligolw.write_tables_to_document(llwdoc_with_tables, [_new()])
    assert len(llw.childNodes[-1]) == 6

    # check overwrite=True gives a fresh table
    io_ligolw.write_tables_to_document(
        llwdoc_with_tables,
        [_new()],
        overwrite=True,
    )
    assert len(llw.childNodes[-1]) == 3


def test_write_tables(tmp_path):
    stab = new_table(
        'segment',
        [{'segment': (1, 2)}, {'segment': (3, 4)}, {'segment': (5, 6)}],
        columns=('start_time', 'start_time_ns', 'end_time', 'end_time_ns'),
    )
    ptab = new_table(
        'process',
        [{'program': 'gwpy'}],
        columns=('program',),
    )

    tmp = tmp_path / "test.xml"

    # write writing works
    io_ligolw.write_tables(tmp, [stab, ptab])
    assert io_ligolw.list_tables(tmp) == ['segment', 'process']

    # check writing to existing file raises IOError
    with pytest.raises(IOError):
        io_ligolw.write_tables(tmp, [stab, ptab])

    # check overwrite=True works
    io_ligolw.write_tables(tmp, [stab], overwrite=True)
    xmldoc = io_ligolw.open_xmldoc(tmp)
    assert io_ligolw.list_tables(xmldoc) == ['segment']
    stab2 = io_ligolw.read_table(xmldoc, 'segment')
    assert len(stab2) == len(stab)

    io_ligolw.write_tables(tmp, [stab, ptab], overwrite=True)  # rewrite

    # check append=True works
    io_ligolw.write_tables(tmp, [stab], append=True)
    xmldoc = io_ligolw.open_xmldoc(tmp)
    assert sorted(io_ligolw.list_tables(xmldoc)) == ['process', 'segment']
    stab2 = io_ligolw.read_table(xmldoc, 'segment')
    assert len(stab2) == len(stab) * 2

    # check append=True, overwrite=True works
    io_ligolw.write_tables(tmp, [stab], append=True, overwrite=True)
    xmldoc = io_ligolw.open_xmldoc(tmp)
    assert sorted(io_ligolw.list_tables(xmldoc)) == ['process', 'segment']
    stab2 = io_ligolw.read_table(xmldoc, 'segment')
    assert len(stab2) == len(stab)


@pytest.mark.requires("ligo.lw.lsctables")
def test_is_ligolw_false():
    assert not io_ligolw.is_ligolw("read", None, None, 1)


def test_is_ligolw_obj(llwdoc):
    assert io_ligolw.is_ligolw("read", None, None, llwdoc)


def test_is_ligolw_file(llwdoc):
    with tempfile.TemporaryDirectory() as tmpdir:
        f = str(Path(tmpdir) / "test.xml")
        with open(f, "w"):  # create an empty file
            pass
        # assert that it isn't identified as LIGO_LW XML
        with open(f, "r") as fobj:
            assert not io_ligolw.is_ligolw("read", f, fobj)

        # now write a LIGO_LW file
        with open(f, "w") as fobj:
            llwdoc.write(fobj)
        # and check that it is identified as LIGO_LW XML
        with open(f, "r") as fobj:
            assert io_ligolw.is_ligolw("read", f, fobj)
