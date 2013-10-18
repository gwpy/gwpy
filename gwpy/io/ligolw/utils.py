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

"""Read LIGO_LW-format XML into Tables
"""


from glue.ligolw import (table as ligolw_table, types as ligolw_types,
                         utils as ligolw_utils, lsctables)

from ... import version
from ...table import Table, Column

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ['to_table', 'from_table']

def to_table(llwtable, columns=None):
    """Convert a LIGO_LW `glue.ligolw.table.Table` object into an
    astropy `~astropy.table.Table` object.
    """
    out = Table()
    out.name = llwtable.tableName
    if columns is None:
        columns = llwtable.columnnames
    for i,column in enumerate(columns):
        incol = llwtable.getColumnByName(column)
        #incol = map(ligolw_types.ToPyType[incol.getAttribute('Type')], incol)
        outcol = Column(data=incol, name=str(column))
        out.add_column(outcol)
    return out


def from_table(t, table_name=None):
    columns = t.colnames
    dtype = t.dtype
    #types = map(lambda x: lsctables.From
    if table_name is None:
        table_name = t.name
    out = lsctables.New(lsctables.TableByName[table_name], columns=columns)
    append = out.append
    RowType = out.RowType
    for row in t:
        llwrow = RowType()
        for col,val in zip(columns, row.data):
            setattr(llwrow, col, val)
        append(llwrow)
    return out
