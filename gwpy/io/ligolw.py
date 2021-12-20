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

"""Basic utilities for reading/writing LIGO_LW-format XML files.

All specific unified input/output for class objects should be placed in
an 'io' subdirectory of the containing directory for that class.
"""

import os
import os.path
from contextlib import contextmanager
from functools import wraps
from importlib import import_module

import numpy

try:
    from ligo.lw.ligolw import (
        ElementError as LigolwElementError,
        LIGOLWContentHandler,
    )
except ImportError:  # no ligo.lw
    LigolwElementError = None
    LIGOLWContentHandler = None

from .utils import (file_list, FILE_LIKE)
from ..utils.decorators import deprecated_function

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# XML elements
XML_SIGNATURE = b'<?xml'
LIGOLW_SIGNATURE = b'<!doctype ligo_lw'
LIGOLW_ELEMENT = b'<ligo_lw>'


# -- hack around around TypeError from LIGOTimeGPS(numpy.int32(...)) ----------

def _ligotimegps(s, ns=0):
    """Catch TypeError and cast `s` and `ns` to `int`
    """
    from lal import LIGOTimeGPS
    try:
        return LIGOTimeGPS(s, ns)
    except TypeError:
        return LIGOTimeGPS(int(s), int(ns))


@contextmanager
def patch_ligotimegps(module="ligo.lw.lsctables"):
    """Context manager to on-the-fly patch LIGOTimeGPS to accept all int types
    """
    module = import_module(module)
    orig = module.LIGOTimeGPS
    module.LIGOTimeGPS = _ligotimegps
    try:
        yield
    finally:
        module.LIGOTimeGPS = orig


# -- content handling ---------------------------------------------------------


def strip_ilwdchar(_ContentHandler):
    """Wrap a LIGO_LW content handler to swap ilwdchar for int on-the-fly
    when reading a document

    This is adapted from :func:`ligo.skymap.utils.ilwd`, copyright
    Leo Singer (GPL-3.0-or-later).
    """
    from ligo.lw.lsctables import TableByName
    from ligo.lw.table import (Column, TableStream)
    from ligo.lw.types import (FromPyType, ToPyType)

    class IlwdMapContentHandler(_ContentHandler):

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._idconverter = {}

        @wraps(_ContentHandler.startColumn)
        def startColumn(self, parent, attrs):
            result = super().startColumn(parent, attrs)

            # if an old ID type, convert type definition to an int
            if result.Type == "ilwd:char":
                old_type = ToPyType[result.Type]

                def converter(old):
                    return int(old_type(old))

                self._idconverter[(id(parent), result.Name)] = converter
                result.Type = FromPyType[int]

            try:
                validcolumns = TableByName[parent.Name].validcolumns
            except KeyError:  # parent.Name not in TableByName
                return result
            if result.Name not in validcolumns:
                stripped_column_to_valid_column = {
                    Column.ColumnName(name): name
                    for name in validcolumns
                }
                if result.Name in stripped_column_to_valid_column:
                    result.setAttribute(
                        'Name',
                        stripped_column_to_valid_column[result.Name],
                    )

            return result

        @wraps(_ContentHandler.startStream)
        def startStream(self, parent, attrs):
            result = super().startStream(parent, attrs)
            if isinstance(result, TableStream):
                loadcolumns = set(parent.columnnames)
                if parent.loadcolumns is not None:
                    loadcolumns &= set(parent.loadcolumns)
                pid = id(parent)
                result._tokenizer.set_types([
                    self._idconverter.pop((pid, colname), pytype)
                    if colname in loadcolumns else None
                    for pytype, colname in zip(
                        parent.columnpytypes,
                        parent.columnnames,
                    )
                ])
            return result

    return IlwdMapContentHandler


def _wrap_content_handler(contenthandler):
    from ligo.lw.lsctables import use_in

    @strip_ilwdchar
    @use_in
    class ContentHandler(contenthandler):
        pass

    return ContentHandler


def default_content_handler():
    """Return a standard content handler to read LIGO_LW documents

    This handler knows how to parse LSCTables, and automatically converts
    old-style ilwdchar ID types to `int`.

    Returns
    -------
    contenthandler : subclass of `ligo.lw.ligolw.LIGOLWContentHandler`
    """
    from ligo.lw.ligolw import LIGOLWContentHandler
    return _wrap_content_handler(LIGOLWContentHandler)


def get_partial_contenthandler(element):
    """Build a `PartialLIGOLWContentHandler` to read only this element

    Parameters
    ----------
    element : `type`, subclass of :class:`~ligo.lw.ligolw.Element`
        the element class to be read

    Returns
    -------
    contenthandler : `type`
        a subclass of `~ligo.lw.ligolw.PartialLIGOLWContentHandler`
        to read only the given `element`
    """
    from ligo.lw.ligolw import PartialLIGOLWContentHandler
    from ligo.lw.table import Table

    if issubclass(element, Table):
        def _element_filter(name, attrs):
            return element.CheckProperties(name, attrs)
    else:
        def _element_filter(name, _):
            return name == element.tagName

    return build_content_handler(PartialLIGOLWContentHandler, _element_filter)


def get_filtering_contenthandler(element):
    """Build a `FilteringLIGOLWContentHandler` to exclude this element

    Parameters
    ----------
    element : `type`, subclass of :class:`~ligo.lw.ligolw.Element`
        the element to exclude (and its children)

    Returns
    -------
    contenthandler : `type`
        a subclass of `~ligo.lw.ligolw.FilteringLIGOLWContentHandler`
        to exclude an element and its children
    """
    from ligo.lw.ligolw import FilteringLIGOLWContentHandler
    from ligo.lw.table import Table

    if issubclass(element, Table):
        def _element_filter(name, attrs):
            return ~element.CheckProperties(name, attrs)
    else:
        def _element_filter(name, _):
            # pylint: disable=unused-argument
            return name != element.tagName

    return build_content_handler(
        FilteringLIGOLWContentHandler,
        _element_filter,
    )


def build_content_handler(parent, filter_func):
    """Build a `~xml.sax.handler.ContentHandler` with a given filter

    Parameters
    ----------
    parent : `type`, subclass of `xml.sax.handler.ContentHandler`
        a class of contenthandler to use

    filter_func : `callable`
        the filter function to pass to the content handler creation

    Returns
    -------
    contenthandler : subclass of ``parent``
        a new content handler that applies the filter function and the
        default parsing extras from :func:`_wrap_content_handler`.
    """
    class ContentHandler(parent):
        # pylint: disable=too-few-public-methods
        def __init__(self, document):
            super().__init__(document, filter_func)

    return _wrap_content_handler(ContentHandler)


# -- reading ------------------------------------------------------------------

def read_ligolw(source, contenthandler=None, **kwargs):
    """Read one or more LIGO_LW format files

    Parameters
    ----------
    source : `str`, `file`
        the open file or file path to read

    contenthandler : `~xml.sax.handler.ContentHandler`, optional
        content handler used to parse document

    verbose : `bool`, optional
        be verbose when reading files, default: `False`

    Returns
    -------
    xmldoc : :class:`~ligo.lw.ligolw.Document`
        the document object as parsed from the file(s)
    """
    from ligo.lw.ligolw import Document
    from ligo.lw import types
    from ligo.lw.utils import (load_url, ligolw_add)

    # mock ToPyType to link to numpy dtypes
    topytype = types.ToPyType.copy()
    for key in types.ToPyType:
        if key in types.ToNumPyType:
            types.ToPyType[key] = numpy.dtype(types.ToNumPyType[key]).type

    # set contenthandler
    if contenthandler is None:
        contenthandler = default_content_handler()

    # read one or more files into a single Document
    source = file_list(source)
    try:
        if len(source) == 1:
            return load_url(
                source[0],
                contenthandler=contenthandler,
                **kwargs
            )
        return ligolw_add.ligolw_add(
            Document(),
            source,
            contenthandler=contenthandler,
            **kwargs
        )
    finally:  # replace ToPyType
        types.ToPyType = topytype


# -- reading ------------------------------------------------------------------

def read_table(
    source,
    tablename=None,
    columns=None,
    contenthandler=None,
    **kwargs,
):
    """Read a :class:`~ligo.lw.table.Table` from one or more LIGO_LW files

    Parameters
    ----------
    source : `Document`, `file`, `str`, `CacheEntry`, `list`
        object representing one or more files. One of

        - a LIGO_LW :class:`~ligo.lw.ligolw.Document`
        - an open `file`
        - a `str` pointing to a file path on disk
        - a formatted :class:`~lal.utils.CacheEntry` representing one file
        - a `list` of `str` file paths or :class:`~lal.utils.CacheEntry`

    tablename : `str`
        name of the table to read.

    columns : `list`, optional
        list of column name strings to read, default all.

    contenthandler : `~xml.sax.handler.ContentHandler`, optional
        SAX content handler for parsing LIGO_LW documents.

    **kwargs
        other keyword arguments are passed to `~gwpy.io.ligolw.read_ligolw`

    Returns
    -------
    table : :class:`~ligo.lw.table.Table`
        `Table` of data
    """
    from ligo.lw.ligolw import Document
    from ligo.lw import (table, lsctables)

    # get content handler to read only this table (if given)
    if tablename is not None:
        tableclass = lsctables.TableByName[
            table.Table.TableName(tablename)
        ]
        if contenthandler is None:
            contenthandler = get_partial_contenthandler(tableclass)

        # overwrite loading column names to get just what was asked for
        _oldcols = tableclass.loadcolumns
        if columns is not None:
            tableclass.loadcolumns = columns

    # read document
    if isinstance(source, Document):
        xmldoc = source
    else:
        if contenthandler is None:
            contenthandler = default_content_handler()
        try:
            xmldoc = read_ligolw(
                source,
                contenthandler=contenthandler,
                **kwargs)
        finally:  # reinstate original set of loading column names
            if tablename is not None:
                tableclass.loadcolumns = _oldcols

    # now find the right table
    if tablename is None:
        tables = list_tables(xmldoc)
        if not tables:
            raise ValueError("No tables found in LIGO_LW document(s)")
        if len(tables) > 1:
            raise ValueError(
                "Multiple tables found in LIGO_LW document(s), please specify "
                "the table to read via the ``tablename=`` keyword argument. "
                "The following tables were found: "
                "'{}'".format("', '".join(tables)),
            )
        tableclass = lsctables.TableByName[table.Table.TableName(tables[0])]

    # extract table
    return tableclass.get_table(xmldoc)


# -- writing ------------------------------------------------------------------

def open_xmldoc(fobj, contenthandler=None, **kwargs):
    """Try and open an existing LIGO_LW-format file, or create a new Document

    Parameters
    ----------
    fobj : `str`, `file`
        file path or open file object to read

    contenthandler : `~xml.sax.handler.ContentHandler`, optional
        the content handler with which to parse the document, if not given
        a default handler will be created using
        :func:`default_content_handler`.

    **kwargs
        other keyword arguments to pass to
        :func:`~ligo.lw.utils.load_fileobj` as appropriate

    Returns
    --------
    xmldoc : :class:`~ligo.lw.ligolw.Document`
        either the `Document` as parsed from an existing file, or a new, empty
        `Document`
    """
    from ligo.lw.ligolw import Document
    from ligo.lw.utils import load_fileobj

    if contenthandler is None:
        contenthandler = default_content_handler()

    # read from an existing Path/filename
    if not isinstance(fobj, FILE_LIKE):
        try:
            with open(fobj, "rb") as fobj2:
                return open_xmldoc(
                    fobj2,
                    contenthandler=contenthandler,
                    **kwargs,
                )
        except (OSError, IOError):
            # or just create a new Document
            return Document()

    return load_fileobj(
        fobj,
        contenthandler=contenthandler,
        **kwargs,
    )


def get_ligolw_element(xmldoc):
    """Find an existing <LIGO_LW> element in this XML Document
    """
    from ligo.lw.ligolw import (LIGO_LW, WalkChildren)

    if isinstance(xmldoc, LIGO_LW):
        return xmldoc
    for elem in WalkChildren(xmldoc):
        if isinstance(elem, LIGO_LW):
            return elem
    raise ValueError("Cannot find LIGO_LW element in XML Document")


def write_tables_to_document(xmldoc, tables, overwrite=False):
    """Write the given LIGO_LW table into a :class:`Document`

    Parameters
    ----------
    xmldoc : :class:`~ligo.lw.ligolw.Document`
        the document to write into

    tables : `list` of :class:`~ligo.lw.table.Table`
        the set of tables to write

    overwrite : `bool`, optional, default: `False`
        if `True`, delete an existing instance of the table type, otherwise
        append new rows
    """
    from ligo.lw.ligolw import LIGO_LW
    from ligo.lw import lsctables

    # find or create LIGO_LW tag
    try:
        llw = get_ligolw_element(xmldoc)
    except ValueError:
        llw = LIGO_LW()
        xmldoc.appendChild(llw)

    for table in tables:
        try:  # append new data to existing table
            old = lsctables.TableByName[
                table.TableName(table.Name)].get_table(xmldoc)
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


def write_tables(
    target,
    tables,
    append=False,
    overwrite=False,
    contenthandler=None,
    **kwargs,
):
    """Write an LIGO_LW table to file

    Parameters
    ----------
    target : `str`, `file`, :class:`~ligo.lw.ligolw.Document`
        the file or document to write into

    tables : `list`, `tuple` of :class:`~ligo.lw.table.Table`
        the tables to write

    append : `bool`, optional, default: `False`
        if `True`, append to an existing file/table, otherwise `overwrite`

    overwrite : `bool`, optional, default: `False`
        if `True`, delete an existing instance of the table type, otherwise
        append new rows

    contenthandler : `~xml.sax.handler.ContentHandler`, optional
        the content handler with which to parse the document, if not given
        a default handler will be created using
        :func:`default_content_handler`.

    **kwargs
        other keyword arguments to pass to
        :func:`~ligo.lw.utils.load_fileobj` as appropriate
    """
    from ligo.lw.ligolw import Document, LIGO_LW
    from ligo.lw import utils as ligolw_utils

    # allow writing directly to XML
    if isinstance(target, (Document, LIGO_LW)):
        xmldoc = target
    # open existing document, if possible
    elif append:
        if contenthandler is None:
            contenthandler = default_content_handler()
        xmldoc = open_xmldoc(
            target,
            contenthandler=contenthandler,
        )
    # fail on existing document and not overwriting
    elif (
        not overwrite
        and isinstance(target, (str, os.PathLike))
        and os.path.exists(target)
    ):
        raise IOError(f"File exists: {target}")
    else:  # or create a new document
        xmldoc = Document()

    # convert table to format
    write_tables_to_document(xmldoc, tables, overwrite=overwrite)

    # find writer function and target filename
    if isinstance(target, FILE_LIKE):
        writer = ligolw_utils.write_fileobj
        name = target.name
    else:
        writer = ligolw_utils.write_filename
        name = target = str(target)

    # handle gzip compression kwargs
    if name.endswith('.gz'):
        kwargs.setdefault('compress', 'gz')

    # write XML
    writer(xmldoc, target, **kwargs)


# -- utilities ----------------------------------------------------------------

def iter_tables(source):
    """Iterate over all tables in the given document(s)

    Parameters
    ----------
    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document`s

    Yields
    ------
    ligo.lw.table.Table
        a table structure from the document(s)
    """
    from ligo.lw.ligolw import (Element, Stream, WalkChildren)

    # get LIGO_LW object
    if not isinstance(source, Element):
        filt = get_filtering_contenthandler(Stream)
        source = read_ligolw(source, contenthandler=filt)
    llw = get_ligolw_element(source)

    # yield tables
    for elem in WalkChildren(llw):
        if elem.tagName == "Table":
            yield elem


def list_tables(source):
    """List the names of all tables in this file(s)

    Parameters
    ----------
    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one or more open files, file paths, or LIGO_LW `Document`s

    Examples
    --------
    >>> from gwpy.io.ligolw import list_tables
    >>> print(list_tables('H1-LDAS_STRAIN-968654552-10.xml.gz'))
    ['process', 'process_params', 'sngl_burst', 'search_summary', 'segment_definer', 'segment_summary', 'segment']
    """  # noqa: E501
    return [tbl.TableName(tbl.Name) for tbl in iter_tables(source)]


def to_table_type(val, cls, colname):
    """Cast a value to the correct type for inclusion in a LIGO_LW table

    This method returns the input unmodified if a type mapping for ``colname``
    isn't found.

    Parameters
    ----------
    val : `object`
        The input object to convert, of any type

    cls : `type`, subclass of :class:`~ligo.lw.table.Table`
        the table class to map against

    colname : `str`
        The name of the mapping column

    Returns
    -------
    obj : `object`
        The input ``val`` cast to the correct type

    Examples
    --------
    >>> from gwpy.io.ligolw import to_table_type as to_ligolw_type
    >>> from ligo.lw.lsctables import SnglBurstTable
    >>> x = to_ligolw_type(1.0, SnglBurstTable, 'central_freq'))
    >>> print(type(x), x)
    <class 'numpy.float32'> 1.0
    """
    from ligo.lw.types import (
        ToNumPyType as numpytypes,
        ToPyType as pytypes,
    )

    # if nothing to do...
    if val is None or colname not in cls.validcolumns:
        return val

    llwtype = cls.validcolumns[colname]

    # map to numpy or python types
    try:
        return numpy.sctypeDict[numpytypes[llwtype]](val)
    except KeyError:
        return pytypes[llwtype](val)


# -- identify -----------------------------------------------------------------

def is_ligolw(origin, filepath, fileobj, *args, **kwargs):
    """Identify a file object as LIGO_LW-format XML
    """
    # pylint: disable=unused-argument
    if fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            line1 = fileobj.readline().lower()
            line2 = fileobj.readline().lower()
            try:
                return (
                    line1.startswith(XML_SIGNATURE)
                    and line2.startswith((LIGOLW_SIGNATURE, LIGOLW_ELEMENT))
                )
            except TypeError:  # bytes vs str
                return (
                    line1.startswith(XML_SIGNATURE.decode('utf-8'))
                    and line2.startswith((
                        LIGOLW_SIGNATURE.decode('utf-8'),
                        LIGOLW_ELEMENT.decode('utf-8'),
                    ))
                )
        finally:
            fileobj.seek(loc)

    try:
        from ligo.lw.ligolw import Element
    except ImportError:
        return
    return len(args) > 0 and isinstance(args[0], Element)


@deprecated_function
def is_xml(origin, filepath, fileobj, *args, **kwargs):  # pragma: no cover
    """Identify a file object as XML (any format)
    """
    # pylint: disable=unused-argument
    if fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            sig = fileobj.read(5).lower()
            return sig == XML_SIGNATURE
        finally:
            fileobj.seek(loc)
    elif filepath is not None:
        return filepath.endswith(('.xml', '.xml.gz'))
