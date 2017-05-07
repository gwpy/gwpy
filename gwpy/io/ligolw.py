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

All specific unified input/output for class objects should be placed in
an 'io' subdirectory of the containing directory for that class.
"""

import os.path
import warnings

from six import string_types

from ..utils import gprint
from .cache import (file_list, FILE_LIKE)
from .utils import identify_factory

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- content handling ---------------------------------------------------------

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
    from glue.ligolw.ligolw import PartialLIGOLWContentHandler

    def _element_filter(name, attrs):
        return table.CheckProperties(name, attrs)

    class _ContentHandler(PartialLIGOLWContentHandler):
        def __init__(self, document):
            super(_ContentHandler, self).__init__(document, _element_filter)

    return _ContentHandler


# -- reading ------------------------------------------------------------------

def table_from_file(f, tablename, columns=None, filt=None,
                    contenthandler=None, nproc=1, verbose=False):
    """Read a `~glue.ligolw.table.Table` from a LIGO_LW file.

    Parameters
    ----------
    f : `file`, `str`, `CacheEntry`, `list`, `Cache`
        object representing one or more files. One of

        - an open `file`
        - a `str` pointing to a file path on disk
        - a formatted `~lal.utils.CacheEntry` representing one file
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
    from glue.ligolw.ligolw import Document
    from glue.ligolw import (table, lsctables)
    from glue.ligolw.utils.ligolw_add import ligolw_add

    # find table class
    tableclass = lsctables.TableByName[table.Table.TableName(tablename)]

    # get content handler
    if contenthandler is None:
        contenthandler = get_partial_contenthandler(tableclass)

    # allow cache multiprocessing
    if nproc != 1:
        return tableclass.read(f, columns=columns,
                               contenthandler=contenthandler,
                               nproc=nproc, format='cache')

    lsctables.use_in(contenthandler)

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


# -- writing ------------------------------------------------------------------

def open_xmldoc(f, **kwargs):
    """Try and open an existing LIGO_LW-format file, or create a new Document
    """
    from glue.ligolw.lsctables import use_in
    from glue.ligolw.ligolw import Document
    from glue.ligolw.utils import load_filename, load_fileobj
    use_in(kwargs['contenthandler'])
    try:  # try and load existing file
        if isinstance(f, string_types):
            return load_filename(f, **kwargs)
        if isinstance(f, FILE_LIKE):
            return load_fileobj(f, **kwargs)[0]
    except (OSError, IOError):  # or just create a new Document
        return Document()


def get_ligolw_element(xmldoc):
    """Find an existing <LIGO_LW> element in this XML Document
    """
    from glue.ligolw.ligolw import LIGO_LW
    if isinstance(xmldoc, LIGO_LW):
        return xmldoc
    else:
        for node in xmldoc.childNodes:
            if isinstance(node, LIGO_LW):
                return node
    raise ValueError("Cannot find LIGO_LW element in XML Document")


def write_tables_to_document(xmldoc, tables, overwrite=False):
    """Write the given LIGO_LW table into a :class:`Document`

    Parameters
    ----------
    xmldoc : :class:`~glue.ligolw.ligolw.Document`
        the document to write into

    tables : `list` of :class:`~glue.ligolw.table.Table`
        the set of tables to write

    overwrite : `bool`, optional, default: `False`
        if `True`, delete an existing instance of the table type, otherwise
        append new rows
    """
    from glue.ligolw.ligolw import LIGO_LW
    from glue.ligolw import lsctables

    # find or create LIGO_LW tag
    try:
        llw = get_ligolw_element(xmldoc)
    except ValueError:
        llw = LIGO_LW()
        xmldoc.appendChild(llw)

    for table in tables:
        try:  # append new data to existing table
            old = lsctables.TableByName[
                table.TableName(table.tableName)].get_table(xmldoc)
        except ValueError:  # or create a new table
            llw.appendChild(table)
        else:
            if overwrite:
                llw.removeChild(old)
                old.unlink()
                llw.appendChild(table)
            else:
                old.extend(table)

    return xmldoc


def write_tables(f, tables, append=False, overwrite=False, **kwargs):
    """Write an LIGO_LW table to file

    Parameters
    ----------
    f : `str`, `file`, :class:`~glue.ligolw.ligolw.Document`
        the file or document to write into

    tables : `list` of :class:`~glue.ligolw.table.Table`
        the tables to write

    append : `bool`, optional, default: `False`
        if `True`, append to an existing file/table, otherwise `overwrite`

    overwrite : `bool`, optional, default: `False`
        if `True`, delete an existing instance of the table type, otherwise
        append new rows
    """
    from glue.ligolw.ligolw import (Document, LIGO_LW, LIGOLWContentHandler)
    from glue.ligolw import utils as ligolw_utils

    # allow writing directly to XML
    if isinstance(f, (Document, LIGO_LW)):
        xmldoc = f
    # open existing document, if possible
    elif append and not overwrite:
        xmldoc = open_xmldoc(
            f, contenthandler=kwargs.pop('contenthandler',
                                         LIGOLWContentHandler))
    # fail on existing document and not overwriting
    elif not overwrite and isinstance(f, string_types) and os.path.isfile(f):
        raise IOError("File exists: %s" % f)
    else:  # or create a new document
        xmldoc = Document()

    # convert table to format
    write_tables_to_document(xmldoc, tables, overwrite=overwrite)

    # write file
    if isinstance(f, string_types):
        kwargs.setdefault('gz', f.endswith('.gz'))
        ligolw_utils.write_filename(xmldoc, f, **kwargs)
    elif not isinstance(f, Document):
        kwargs.setdefault('gz', f.name.endswith('.gz'))
        ligolw_utils.write_fileobj(xmldoc, f, **kwargs)


# -- identify -----------------------------------------------------------------

identify_ligolw = identify_factory('xml', 'xml.gz')
