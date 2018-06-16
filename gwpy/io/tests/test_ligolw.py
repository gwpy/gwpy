# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

from ...tests.utils import skip_missing_dependency
from .. import ligolw as io_ligolw

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@skip_missing_dependency('glue.ligolw.lsctables')  # check for LAL
def test_open_xmldoc():
    from glue.ligolw.ligolw import (Document, LIGO_LW)
    assert isinstance(io_ligolw.open_xmldoc(tempfile.mktemp()), Document)
    with tempfile.TemporaryFile(mode='w') as f:
        xmldoc = Document()
        xmldoc.appendChild(LIGO_LW())
        xmldoc.write(f)
        f.seek(0)
        assert isinstance(io_ligolw.open_xmldoc(f), Document)


@skip_missing_dependency('glue.ligolw')
def test_get_ligolw_element():
    from glue.ligolw.ligolw import (Document, LIGO_LW)
    xmldoc = Document()
    llw = xmldoc.appendChild(LIGO_LW())
    assert io_ligolw.get_ligolw_element(llw) is llw
    assert io_ligolw.get_ligolw_element(xmldoc) is llw
    with pytest.raises(ValueError):
        io_ligolw.get_ligolw_element(Document())


@skip_missing_dependency('glue.ligolw.lsctables')  # check for LAL
def test_list_tables():
    from glue.ligolw import lsctables
    from glue.ligolw.ligolw import (Document, LIGO_LW)

    # build dummy document with two tables
    xmldoc = Document()
    llw = xmldoc.appendChild(LIGO_LW())
    tables = [lsctables.New(lsctables.ProcessTable),
              lsctables.New(lsctables.SnglRingdownTable)]
    names = [t.TableName(t.Name) for t in tables]
    [llw.appendChild(t) for t in tables]  # add tables to xmldoc

    # check that tables are listed properly
    assert io_ligolw.list_tables(xmldoc) == names

    # check that we can list from files
    with tempfile.NamedTemporaryFile(mode='w') as f:
        xmldoc.write(f)
        f.seek(0)
        assert io_ligolw.list_tables(f) == names


@skip_missing_dependency('glue.ligolw.lsctables')  # check for LAL
@pytest.mark.parametrize('value, name, result', [
    (None, 'peak_time', None),
    (1.0, 'peak_time', numpy.int32(1)),
    (1, 'process_id', 'sngl_burst:process_id:1'),
    (1.0, 'invalidname', 1.0),
    ('process:process_id:100', 'process_id', 'process:process_id:100'),
])
def test_to_table_type(value, name, result):
    from glue.ligolw.lsctables import SnglBurstTable
    from glue.ligolw.ilwd import ilwdchar
    from glue.ligolw._ilwd import ilwdchar as IlwdChar
    out = io_ligolw.to_table_type(value, SnglBurstTable, name)
    if isinstance(out, IlwdChar):
        result = ilwdchar(result)
    assert isinstance(out, type(result))
    assert out == result


@skip_missing_dependency('glue.ligolw.lsctables')  # check for LAL
def test_to_table_type_ilwd():
    from glue.ligolw.ilwd import ilwdchar
    from glue.ligolw.lsctables import SnglBurstTable
    ilwd = ilwdchar('process:process_id:0')
    with pytest.raises(ValueError) as exc:
        io_ligolw.to_table_type(ilwd, SnglBurstTable, 'event_id')
    assert str(exc.value) == ('ilwdchar \'process:process_id:0\' doesn\'t '
                              'match column \'event_id\'')
