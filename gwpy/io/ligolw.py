# coding=utf-8
# Copyright (C) Duncan Macleod (2014)
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

"""Basic utilities for reading/writing LIGO_LW-format XML files.

All specific unified input/output for class objecst should be placed in
an 'io' subdirectory of the containing directory for that class.
"""

from glue.lal import CacheEntry
from glue.ligolw.ligolw import LIGOLWContentHandler
from glue.ligolw import (utils as llwutils, table as llwtable, lsctables)

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class GWpyContentHandler(LIGOLWContentHandler):
    pass


def table_from_file(f, tablename, columns=None,
                    contenthandler=GWpyContentHandler):
    """Read a :class:`~glue.ligolw.table.Table` from a LIGO_LW file.

    Parameters
    ----------
    f : `file`, `str`
        open `file` in memory, or path to file on disk.
    tablename : `str`
        name of the table to read.
    columns : `list`, optional
        list of column name strings to read, default all.
    contenthandler : :class:`~glue.ligolw.ligolw.LIGOLWContentHandler`
        SAX content handler for parsing LIGO_LW documents.

    Returns
    -------
    table : :class:`~glue.ligolw.table.Table`
        `Table` of data with given columns filled

    """
    # find table class
    tableclass = lsctables.TableByName[llwtable.StripTableName(tablename)]
    # set columns to read
    if columns is not None:
        _oldcols = tableclass.loadcolumns
        tableclass.loadcolumns = columns
    # load file
    if isinstance(f, CacheEntry):
        f = f.path
    if isinstance(f, (str, unicode)):
        xmldoc = llwutils.load_filename(f, contenthandler=contenthandler)
    else:
        xmldoc, _ = llwutils.load_fileobj(f, contenthandler=contenthandler)
    out = tableclass.get_table(xmldoc)
    if columns is not None:
        tableclass.loadcolumns = _oldcols
    return out


def identify_ligolw_file(*args, **kwargs):
    """Determine an input object as either a LIGO_LW format file.
    """
    fp = args[3]
    if isinstance(fp, file):
        fp = fp.name
    elif isinstance(fp, CacheEntry):
        fp = fp.path
    # identify string
    if isinstance(fp, (unicode, str)) and fp.endswith(('xml', 'xml.gz')):
        return True
    else:
        return False
