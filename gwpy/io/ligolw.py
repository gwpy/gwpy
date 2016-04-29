# -*- coding: utf-8 -*-
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
from glue.ligolw.ligolw import (Document, LIGOLWContentHandler,
                                PartialLIGOLWContentHandler)
from glue.ligolw.table import CompareTableNames as compare_table_names
from glue.ligolw.utils.ligolw_add import ligolw_add
from glue.ligolw import (table, lsctables)

from ..utils import gprint
from .cache import file_list
from .utils import identify_factory

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class GWpyContentHandler(LIGOLWContentHandler):
    """Empty sub-class of `~glue.ligolw.ligolw.LIGOLWContentHandler`
    """
    pass


def get_partial_contenthandler(table):
    """Build a `~glue.ligolw.ligolw.PartialLIGOLWContentHandler` for this table

    Parameters
    ----------
    table : `type`
        the table class to be read

    Returns
    -------
    contenthandler : `type`
        a subclass of `~glue.ligolw.ligolw.PartialLIGOLWContentHandler` to
        read only the given `table`
    """
    def _filter_func(name, attrs):
        if name == table.tagName and attrs.has_key('Name'):
            return compare_table_names(attrs.get('Name'), table.tableName) == 0
        else:
            return False

    class _ContentHandler(PartialLIGOLWContentHandler):
        def __init__(self, document):
            super(_ContentHandler, self).__init__(document, _filter_func)

    return _ContentHandler


def table_from_file(f, tablename, columns=None, filt=None,
                    contenthandler=None, nproc=1, verbose=False):
    """Read a `~glue.ligolw.table.Table` from a LIGO_LW file.

    Parameters
    ----------
    f : `file`, `str`, `CacheEntry`, `list`, `Cache`
        object representing one or more files. One of

        - an open `file`
        - a `str` pointing to a file path on disk
        - a formatted `~glue.lal.CacheEntry` representing one file
        - a `list` of `str` file paths
        - a formatted `~glue.lal.Cache` representing many files

    tablename : `str`
        name of the table to read.
    columns : `list`, optional
        list of column name strings to read, default all.
    filt : `function`, optional
        function by which to `filter` events. The callable must accept as
        input a row of the table event and return `True`/`False`.
    contenthandler : `~glue.ligolw.ligolw.LIGOLWContentHandler`
        SAX content handler for parsing LIGO_LW documents.

    Returns
    -------
    table : `~glue.ligolw.table.Table`
        `Table` of data with given columns filled
    """
    # find table class
    tableclass = lsctables.TableByName[table.StripTableName(tablename)]

    # get content handler
    if contenthandler is None:
        contenthandler = get_partial_contenthandler(tableclass)

    # allow cache multiprocessing
    if nproc != 1:
        return tableclass.read(f, columns=columns,
                               contenthandler=contenthandler,
                               nproc=nproc, format='cache')

    # set columns to read
    if columns is not None:
        _oldcols = tableclass.loadcolumns
        tableclass.loadcolumns = columns

    # generate Document and populate
    files = file_list(f)
    xmldoc = Document()
    ligolw_add(xmldoc, files, non_lsc_tables_ok=True,
               contenthandler=contenthandler, verbose=verbose)

    # extract table
    out = tableclass.get_table(xmldoc)
    if verbose:
        gprint('%d rows found in %s table' % (len(out), out.tableName))

    # filter output
    if filt:
        if verbose:
            gprint('filtering rows ...', end=' ')
        try:
            out_ = out.copy()
        except AttributeError:
            out_ = table.new_from_template(out)
        out_.extend(filter(filt, out))
        out = out_
        if verbose:
            gprint('%d rows remaining\n' % len(out))

    # reset loadcolumns and return
    if columns is not None:
        tableclass.loadcolumns = _oldcols

    return out


identify_ligolw = identify_factory('xml', 'xml.gz')
