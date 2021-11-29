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

"""Unit test for `io` module
"""

import importlib
import tempfile
from pathlib import Path

import pytest

import numpy

from ...testing.utils import skip_missing_dependency
from .. import ligolw as io_ligolw

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

LIGOLW_LIBS = (
    "glue.ligolw",  # old
    "ligo.lw",  # new
)

parametrize_ilwdchar_compat = pytest.mark.parametrize("ilwdchar_compat", [
    pytest.param(False, marks=skip_missing_dependency("ligo.lw.lsctables")),
    pytest.param(True, marks=skip_missing_dependency("glue.ligolw.lsctables")),
])


# -- fixtures -----------------------------------------------------------------

@pytest.fixture(
    scope="function",
    params=[
        pytest.param(
            lib,
            marks=skip_missing_dependency(f"{lib}.lsctables"))
        for lib in LIGOLW_LIBS
    ],
)
def ligolw_lib(request):
    """Parametrize which ligolw library to use
    """
    return importlib.import_module(request.param)


@pytest.fixture(scope="function")
def llwdoc(ligolw_lib):
    """Build an empty LIGO_LW Document
    """
    # build empty LIGO_LW document
    ligolw_pkg = ligolw_lib.__name__
    ligolw = importlib.import_module(".ligolw", package=ligolw_pkg)
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    return xmldoc


@pytest.fixture(scope="function")
def llwdoc_with_tables(llwdoc):
    """Build a LIGO_LW Document with some tables
    """
    llw = llwdoc.childNodes[-1]  # get ligolw element
    ilwdchar_compat = llwdoc.__module__.startswith("glue")
    for t in [
        new_table('process', ilwdchar_compat=ilwdchar_compat),
        new_table('sngl_ringdown', ilwdchar_compat=ilwdchar_compat),
    ]:
        llw.appendChild(t)
    return llwdoc


@io_ligolw.ilwdchar_compat
def new_table(tablename, data=None, **new_kw):
    from ligo.lw import lsctables
    from ligo.lw.table import Table

    table = lsctables.New(lsctables.TableByName[Table.TableName(tablename)],
                          **new_kw)
    for dat in data or list():
        row = table.RowType()
        for key, val in dat.items():
            setattr(row, key, val)
        table.append(row)
    return table


# -- tests --------------------------------------------------------------------

@parametrize_ilwdchar_compat
def test_ilwdchar_compat(ilwdchar_compat):
    if ilwdchar_compat:
        from glue.ligolw.table import Table
    else:
        from ligo.lw.table import Table
    # test that our new_table function actually returns properly
    tab = new_table("sngl_burst", ilwdchar_compat=ilwdchar_compat)
    assert isinstance(tab, Table)


def test_read_table(llwdoc_with_tables):
    tab = io_ligolw.read_table(llwdoc_with_tables, tablename="process")
    assert tab is llwdoc_with_tables.childNodes[0].childNodes[0]


def test_read_table_empty(llwdoc):
    with pytest.raises(ValueError) as exc:
        io_ligolw.read_table(llwdoc)
    assert str(exc.value) == "No tables found in LIGO_LW document(s)"


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


@skip_missing_dependency('ligo.lw.lsctables')  # check for LAL
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


@skip_missing_dependency('glue.ligolw.lsctables')
@pytest.mark.parametrize('value, name, result', [
    (None, 'peak_time', None),
    (1.0, 'peak_time', numpy.int32(1)),
    (1, 'process_id', 'sngl_burst:process_id:1'),
    (1.0, 'invalidname', 1.0),
    ('process:process_id:100', 'process_id', 'process:process_id:100'),
])
def test_to_table_type_glue_ligolw(value, name, result):
    from glue.ligolw.lsctables import SnglBurstTable
    from glue.ligolw.ilwd import ilwdchar
    from glue.ligolw._ilwd import ilwdchar as IlwdChar
    out = io_ligolw.to_table_type(value, SnglBurstTable, name)
    if isinstance(out, IlwdChar):
        result = ilwdchar(result)
    assert isinstance(out, type(result))
    assert out == result


def test_write_tables_to_document(llwdoc_with_tables):
    # create new table
    def _new():
        return new_table(
            'segment',
            [{'segment': (1, 2)}, {'segment': (3, 4)}, {'segment': (5, 6)}],
            columns=('start_time', 'start_time_ns', 'end_time', 'end_time_ns'),
            ilwdchar_compat=llwdoc_with_tables.__module__.startswith("glue"),
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


@parametrize_ilwdchar_compat
def test_write_tables(ilwdchar_compat, tmp_path):
    stab = new_table(
        'segment',
        [{'segment': (1, 2)}, {'segment': (3, 4)}, {'segment': (5, 6)}],
        columns=('start_time', 'start_time_ns', 'end_time', 'end_time_ns'),
        ilwdchar_compat=ilwdchar_compat
    )
    ptab = new_table(
        'process',
        [{'program': 'gwpy'}],
        columns=('program',),
        ilwdchar_compat=ilwdchar_compat,
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
