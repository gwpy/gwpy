# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

import tempfile

import pytest

import numpy

from ...testing.utils import (skip_missing_dependency, TemporaryFilename)
from .. import ligolw as io_ligolw

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


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


@pytest.mark.parametrize("ilwdchar_compat", [
    pytest.param(False, marks=skip_missing_dependency("ligo.lw.lsctables")),
    pytest.param(True, marks=skip_missing_dependency("glue.ligolw.lsctables")),
])
def test_ilwdchar_compat(ilwdchar_compat):
    if ilwdchar_compat:
        from glue.ligolw.table import Table
    else:
        from ligo.lw.table import Table
    # test that our new_table function actually returns properly
    tab = new_table("sngl_burst", ilwdchar_compat=ilwdchar_compat)
    assert isinstance(tab, Table)


@skip_missing_dependency('ligo.lw.lsctables')  # check for LAL
@pytest.fixture
def llwtable():
    from ligo.lw.ligolw import (Document, LIGO_LW)

    # build dummy document with two tables
    xmldoc = Document()
    llw = xmldoc.appendChild(LIGO_LW())
    tables = [new_table('process'), new_table('sngl_ringdown')]
    for t in tables:
        llw.appendChild(t)
    return xmldoc


@skip_missing_dependency('ligo.lw.lsctables')  # check for LAL
def test_open_xmldoc(llwtable):
    from ligo.lw.ligolw import Document
    assert isinstance(io_ligolw.open_xmldoc(tempfile.mktemp()), Document)

    with tempfile.TemporaryFile(mode='w') as f:
        llwtable.write(f)
        f.seek(0)
        assert isinstance(io_ligolw.open_xmldoc(f), Document)


@skip_missing_dependency('ligo.lw')
def test_get_ligolw_element():
    from ligo.lw.ligolw import (Document, LIGO_LW)
    xmldoc = Document()
    llw = xmldoc.appendChild(LIGO_LW())
    assert io_ligolw.get_ligolw_element(llw) is llw
    assert io_ligolw.get_ligolw_element(xmldoc) is llw
    with pytest.raises(ValueError):
        io_ligolw.get_ligolw_element(Document())


@skip_missing_dependency('ligo.lw.lsctables')  # check for LAL
def test_list_tables(llwtable):
    names = [t.TableName(t.Name) for t in llwtable.childNodes[0].childNodes]

    # check that tables are listed properly
    assert io_ligolw.list_tables(llwtable) == names

    # check that we can list from files
    with tempfile.NamedTemporaryFile(mode='w') as f:
        llwtable.write(f)
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


@skip_missing_dependency('ligo.lw.lsctables')  # check for LAL
def test_write_tables_to_document(llwtable):
    # create new table
    def _new():
        return new_table(
            'segment',
            [{'segment': (1, 2)}, {'segment': (3, 4)}, {'segment': (5, 6)}],
            columns=('start_time', 'start_time_ns', 'end_time', 'end_time_ns'))

    llw = llwtable.childNodes[-1]

    # check we can add a new table
    tab = _new()
    io_ligolw.write_tables_to_document(llwtable, [tab])
    assert llw.childNodes[-1] is tab
    assert len(tab) == 3

    # check that adding to an existing table extends
    io_ligolw.write_tables_to_document(llwtable, [_new()])
    assert len(llw.childNodes[-1]) == 6

    # check overwrite=True gives a fresh table
    io_ligolw.write_tables_to_document(llwtable, [_new()], overwrite=True)
    assert len(llw.childNodes[-1]) == 3


@skip_missing_dependency('ligo.lw.lsctables')  # check for LAL
def test_write_tables():
    stab = new_table(
        'segment',
        [{'segment': (1, 2)}, {'segment': (3, 4)}, {'segment': (5, 6)}],
        columns=('start_time', 'start_time_ns', 'end_time', 'end_time_ns'),
    )
    ptab = new_table('process', [{'program': 'gwpy'}], columns=('program',))

    with TemporaryFilename() as tmp:
        # write writing works
        io_ligolw.write_tables(tmp, [stab, ptab])
        assert io_ligolw.list_tables(tmp) == ['segment', 'process']

        # check writing to existing file raises IOError
        with pytest.raises(IOError):
            io_ligolw.write_tables(tmp, [stab, ptab])

        # check overwrite=True works
        io_ligolw.write_tables(tmp, [stab], overwrite=True)
        print(open(tmp, 'rb').read())
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
