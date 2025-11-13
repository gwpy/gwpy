# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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


from __future__ import annotations

import os
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy

from .utils import (
    FileLike,
    file_list,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Collection,
        Generator,
        Iterable,
    )
    from typing import Literal
    from xml.sax.handler import ContentHandler
    from xml.sax.xmlreader import AttributesImpl

    from igwn_ligolw.ligolw import (
        Document,
        Element,
        FilteringLIGOLWContentHandler,
        PartialLIGOLWContentHandler,
        Stream,
        Table,
    )

    from .utils import (
        FileSystemPath,
        NamedReadable,
        Readable,
        Writable,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "build_content_handler",
    "default_content_handler",
    "get_filtering_contenthandler",
    "get_ligolw_element",
    "get_partial_contenthandler",
    "is_ligolw",
    "iter_tables",
    "list_tables",
    "open_xmldoc",
    "read_ligolw",
    "read_table",
    "to_table_type",
    "write_tables",
    "write_tables_to_document",
]

# XML elements
XML_SIGNATURE = b"<?xml"
LIGOLW_SIGNATURE = b"<!doctype ligo_lw"
LIGOLW_ELEMENT = b"<ligo_lw>"


# -- content handling ----------------

def _int_ilwd(ilwd: str) -> int:
    """Convert an ``ilwd`` string into an integer."""
    try:
        _, _, i = ilwd.strip().split(":")
    except ValueError as exc:
        msg = f"invalid ilwd:char '{ilwd}'"
        raise ValueError(msg) from exc
    return int(i)


def strip_ilwdchar(content_handler: type[ContentHandler]) -> type[ContentHandler]:
    """Wrap a contenthandler to swap ilwdchar for int when reading a document.

    This is adapted from :func:`ligo.skymap.utils.ilwd`, copyright
    Leo Singer (GPL-3.0-or-later).
    """
    from igwn_ligolw.ligolw import (
        Column,
        Table,
    )
    from igwn_ligolw.lsctables import TableByName
    from igwn_ligolw.types import FromPyType

    class IlwdMapContentHandler(content_handler):  # type: ignore[misc,valid-type]

        def __init__(self, *args, **kwargs) -> None:  # noqa: ANN002
            super().__init__(*args, **kwargs)
            self._idconverter: dict[tuple[int, str], Callable[[str], int]] = {}

        def startColumn(  # noqa: N802
            self,
            parent: Element,
            attrs: AttributesImpl,
        ) -> Column:
            result = super().startColumn(parent, attrs)

            # if an old ID type, convert type definition to an int
            if result.Type == "ilwd:char":
                self._idconverter[(id(parent), result.Name)] = _int_ilwd
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
                        "Name",
                        stripped_column_to_valid_column[result.Name],
                    )

            return result

        def startStream(  # noqa: N802
            self,
            parent: Element,
            attrs: AttributesImpl,
        ) -> Stream:
            result = super().startStream(parent, attrs)
            if isinstance(result, Table.Stream):
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
                        strict=True,
                    )
                ])
            return result

    return IlwdMapContentHandler


def _wrap_content_handler(contenthandler: type[ContentHandler]) -> type[ContentHandler]:

    @strip_ilwdchar
    class ContentHandler(contenthandler):  # type: ignore[misc,valid-type]
        pass

    return ContentHandler


def default_content_handler() -> type[ContentHandler]:
    """Return a standard content handler to read LIGO_LW documents.

    This handler knows how to parse LSCTables, and automatically converts
    old-style ilwdchar ID types to `int`.

    Returns
    -------
    contenthandler : subclass of `igwn_ligolw.ligolw.LIGOLWContentHandler`
    """
    from igwn_ligolw.ligolw import LIGOLWContentHandler
    return _wrap_content_handler(LIGOLWContentHandler)


def get_partial_contenthandler(
    element: type[Element],
) -> type[PartialLIGOLWContentHandler]:
    """Build a `PartialLIGOLWContentHandler` to read only this element.

    Parameters
    ----------
    element : `type`, subclass of :class:`~igwn_ligolw.ligolw.Element`
        the element class to be read

    Returns
    -------
    contenthandler : `type`
        a subclass of `~igwn_ligolw.ligolw.PartialLIGOLWContentHandler`
        to read only the given `element`
    """
    from igwn_ligolw.ligolw import (
        PartialLIGOLWContentHandler,
        Table,
    )

    if issubclass(element, Table):
        def _element_filter(name: str, attrs: AttributesImpl) -> bool:
            return element.CheckProperties(name, attrs)
    else:
        def _element_filter(name: str, attrs: AttributesImpl) -> bool:  # noqa: ARG001
            return name == element.tagName

    return build_content_handler(PartialLIGOLWContentHandler, _element_filter)


def get_filtering_contenthandler(
    element: type[Element],
) -> type[FilteringLIGOLWContentHandler]:
    """Build a `FilteringLIGOLWContentHandler` to exclude this element.

    Parameters
    ----------
    element : `type`, subclass of :class:`~igwn_ligolw.ligolw.Element`
        the element to exclude (and its children)

    Returns
    -------
    contenthandler : `type`
        a subclass of `~igwn_ligolw.ligolw.FilteringLIGOLWContentHandler`
        to exclude an element and its children
    """
    from igwn_ligolw.ligolw import (
        FilteringLIGOLWContentHandler,
        Table,
    )

    if issubclass(element, Table):
        def _element_filter(name: str, attrs: AttributesImpl) -> bool:
            return ~element.CheckProperties(name, attrs)
    else:
        def _element_filter(name: str, attrs: AttributesImpl) -> bool:  # noqa: ARG001
            return name != element.tagName

    return build_content_handler(
        FilteringLIGOLWContentHandler,
        _element_filter,
    )


def build_content_handler(
    parent: type[PartialLIGOLWContentHandler],
    filter_func: Callable[[str, AttributesImpl], bool],
) -> type[PartialLIGOLWContentHandler]:
    """Build a `~xml.sax.handler.ContentHandler` with a given filter.

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
    class ContentHandler(parent):  # type: ignore[misc,valid-type]
        def __init__(self, document: Document) -> None:
            super().__init__(document, filter_func)

    return _wrap_content_handler(ContentHandler)


# -- reading -------------------------

def read_ligolw(
    source: NamedReadable | list[NamedReadable],
    contenthandler: type[ContentHandler] | None = None,
    **kwargs,
) -> Document:
    """Read one or more LIGO_LW format files.

    Parameters
    ----------
    source : `str`, `file`, `list` of `str` or `file`
        The open file or file path to read.

    contenthandler : `~xml.sax.handler.ContentHandler`, optional
        Content handler used to parse document.

    kwargs
        Other keyword arguments to pass to `igwn_ligolw.utils.load_url`.

    Returns
    -------
    xmldoc : `~igwn_ligolw.ligolw.Document`
        the document object as parsed from the file(s)
    """
    from igwn_ligolw import types
    from igwn_ligolw.ligolw import Document
    from igwn_ligolw.utils import ligolw_add, load_url

    # mock ToPyType to link to numpy dtypes
    topytype = types.ToPyType.copy()
    for key in types.ToPyType:
        if key in types.ToNumPyType:
            types.ToPyType[key] = numpy.dtype(types.ToNumPyType[key]).type

    # set contenthandler
    if contenthandler is None:
        contenthandler = default_content_handler()

    # read one or more files into a single Document
    sources = file_list(source)
    try:
        if len(sources) == 1:
            return load_url(
                sources[0],
                contenthandler=contenthandler,
                **kwargs,
            )
        return ligolw_add.ligolw_add(
            Document(),
            sources,
            contenthandler=contenthandler,
            **kwargs,
        )
    finally:  # replace ToPyType
        types.ToPyType = topytype


# -- reading -------------------------

def read_table(
    source: Document | NamedReadable | list[NamedReadable],
    tablename: str | None = None,
    columns: Collection[str] | None = None,
    contenthandler: type[ContentHandler] | None = None,
    **kwargs,
) -> Table:
    """Read a :class:`~igwn_ligolw.ligolw.Table` from one or more LIGO_LW files.

    Parameters
    ----------
    source : `Document`, `file`, `str`, `CacheEntry`, `list`
        object representing one or more files. One of

        - a LIGO_LW :class:`~igwn_ligolw.ligolw.Document`
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
    table : :class:`~igwn_ligolw.ligolw.Table`
        `Table` of data
    """
    from igwn_ligolw import lsctables
    from igwn_ligolw.ligolw import (
        Document,
        Table,
    )

    # get content handler to read only this table (if given)
    if tablename is not None:
        tableclass = lsctables.TableByName[
            Table.TableName(tablename)
        ]
        if contenthandler is None:
            contenthandler = get_partial_contenthandler(tableclass)

        # overwrite loading column names to get just what was asked for
        _oldcols = tableclass.loadcolumns
        if columns is not None:
            tableclass.loadcolumns = set(columns)

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
            msg = "No tables found in LIGO_LW document(s)"
            raise ValueError(msg)
        if len(tables) > 1:
            msg = (
                "Multiple tables found in LIGO_LW document(s), please specify "
                "the table to read via the ``tablename=`` keyword argument. "
                "The following tables were found: "
                "'{}'".format("', '".join(tables))
            )
            raise ValueError(msg)
        tableclass = lsctables.TableByName[Table.TableName(tables[0])]

    # extract table
    return tableclass.get_table(xmldoc)


# -- writing -------------------------

def open_xmldoc(
    fobj: Readable,
    contenthandler: type[ContentHandler] | None = None,
    **kwargs,
) -> Document:
    """Try and open an existing LIGO_LW-format file, or create a new Document.

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
        :func:`~igwn_ligolw.utils.load_fileobj` as appropriate

    Returns
    -------
    xmldoc : :class:`~igwn_ligolw.ligolw.Document`
        either the `Document` as parsed from an existing file, or a new, empty
        `Document`
    """
    from igwn_ligolw.ligolw import Document
    from igwn_ligolw.utils import load_fileobj

    if contenthandler is None:
        contenthandler = default_content_handler()

    # read from an existing Path/filename
    if not isinstance(fobj, FileLike):
        try:
            with open(fobj, "rb") as fobj2:
                return open_xmldoc(
                    fobj2,
                    contenthandler=contenthandler,
                    **kwargs,
                )
        except OSError:
            # or just create a new Document
            return Document()

    return load_fileobj(
        fobj,
        contenthandler=contenthandler,
        **kwargs,
    )


def get_ligolw_element(xmldoc: Document) -> Element:
    """Find an existing <LIGO_LW> element in this XML Document."""
    from igwn_ligolw.ligolw import LIGO_LW, WalkChildren

    if isinstance(xmldoc, LIGO_LW):
        return xmldoc
    for elem in WalkChildren(xmldoc):
        if isinstance(elem, LIGO_LW):
            return elem
    msg = "Cannot find LIGO_LW element in XML Document"
    raise ValueError(msg)


def write_tables_to_document(
    xmldoc: Document,
    tables: Iterable[Table],
    *,
    overwrite: bool = False,
) -> Document:
    """Write the given LIGO_LW table into a :class:`Document`.

    Parameters
    ----------
    xmldoc : :class:`~igwn_ligolw.ligolw.Document`
        the document to write into

    tables : `list` of :class:`~igwn_ligolw.ligolw.Table`
        the set of tables to write

    overwrite : `bool`, optional, default: `False`
        if `True`, delete an existing instance of the table type, otherwise
        append new rows
    """
    from igwn_ligolw import lsctables
    from igwn_ligolw.ligolw import LIGO_LW

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
        except ValueError:
            # or create a new table
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
    target: Writable | Document,
    tables: Iterable[Table],
    *,
    append: bool = False,
    overwrite: bool = False,
    contenthandler: type[ContentHandler] | None = None,
    **kwargs,
) -> None:
    """Write an LIGO_LW table to file.

    Parameters
    ----------
    target : `str`, `file`, :class:`~igwn_ligolw.ligolw.Document`
        the file or document to write into

    tables : `list`, `tuple` of :class:`~igwn_ligolw.ligolw.Table`
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
        :func:`~igwn_ligolw.utils.load_fileobj` as appropriate
    """
    from igwn_ligolw import utils as ligolw_utils
    from igwn_ligolw.ligolw import LIGO_LW, Document

    # allow writing directly to XML
    if isinstance(target, Document | LIGO_LW):
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
        and isinstance(target, str | os.PathLike)
        and Path(target).exists()
    ):
        msg = f"File exists: {target}"
        raise OSError(msg)
    else:  # or create a new document
        xmldoc = Document()

    # convert table to format
    write_tables_to_document(xmldoc, tables, overwrite=overwrite)

    # find writer function and target filename
    if isinstance(target, FileLike):
        writer = ligolw_utils.write_fileobj
        try:
            name = target.name  # type: ignore[union-attr]
        except AttributeError:
            name = ""
    else:
        writer = ligolw_utils.write_filename
        name = target = str(target)

    # handle gzip compression kwargs
    if name.endswith(".gz"):
        kwargs.setdefault("compress", "gz")

    # write XML
    writer(xmldoc, target, **kwargs)


# -- utilities -----------------------

def iter_tables(
    source: Document | NamedReadable | list[NamedReadable],
) -> Generator[Table, None, None]:
    """Iterate over all tables in the given document(s).

    Parameters
    ----------
    source : `file`, `str`, :class:`~igwn_ligolw.ligolw.Document`, `list`
        One or more open files, file paths,
        or `LIGO_LW documents <igwn_ligolw.ligolw.Document>`.

    Yields
    ------
    igwn_ligolw.ligolw.Table
        A table structure from the document(s).
    """
    from igwn_ligolw.ligolw import Element, Stream, WalkChildren

    # get LIGO_LW object
    if not isinstance(source, Element):
        filt = get_filtering_contenthandler(Stream)
        source = read_ligolw(source, contenthandler=filt)
    llw = get_ligolw_element(source)

    # yield tables
    for elem in WalkChildren(llw):
        if elem.tagName == "Table":
            yield elem


def list_tables(source: FileLike | str | Document | list) -> list[str]:
    """List the names of all tables in this file(s).

    Parameters
    ----------
    source : `file`, `str`, :class:`~igwn_ligolw.ligolw.Document`, `list`
        One or more open files, file paths,
        or `LIGO_LW documents <igwn_ligolw.ligolw.Document>`.

    Examples
    --------
    >>> from gwpy.io.ligolw import list_tables
    >>> print(list_tables('H1-LDAS_STRAIN-968654552-10.xml.gz'))
    ['process', 'process_params', 'sngl_burst', 'search_summary', 'segment_definer', 'segment_summary', 'segment']
    """  # noqa: E501
    return [tbl.TableName(tbl.Name) for tbl in iter_tables(source)]


def to_table_type(
    val: object,
    cls: type[Table],
    colname: str,
) -> object:
    """Cast a value to the correct type for inclusion in a LIGO_LW table.

    This method returns the input unmodified if a type mapping for ``colname``
    isn't found.

    Parameters
    ----------
    val : `object`
        The input object to convert, of any type

    cls : `type`, subclass of :class:`~igwn_ligolw.ligolw.Table`
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
    >>> from igwn_ligolw.lsctables import SnglBurstTable
    >>> x = to_ligolw_type(1.0, SnglBurstTable, 'central_freq'))
    >>> print(type(x), x)
    <class 'numpy.float32'> 1.0
    """
    from igwn_ligolw.types import (
        ToNumPyType as numpytypes,  # noqa: N813
        ToPyType as pytypes,  # noqa: N813
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


# -- identify ------------------------

def is_ligolw(
    origin: Literal["read", "write"],  # noqa: ARG001
    filepath: FileSystemPath | None,  # noqa: ARG001
    fileobj: FileLike | None,
    *args,  # noqa: ANN002
    **kwargs,  # noqa: ARG001
) -> bool:
    """Identify a file object as LIGO_LW-format XML."""
    if fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            line1 = fileobj.readline().lower()
            line2 = fileobj.readline().lower()
            try:
                # binary format
                return (
                    line1.startswith(XML_SIGNATURE)
                    and line2.startswith((LIGOLW_SIGNATURE, LIGOLW_ELEMENT))
                )
            except TypeError:
                # text format
                line1 = cast("str", line1)
                line2 = cast("str", line2)
                return (
                    line1.startswith(XML_SIGNATURE.decode("utf-8"))
                    and line2.startswith((
                        LIGOLW_SIGNATURE.decode("utf-8"),
                        LIGOLW_ELEMENT.decode("utf-8"),
                    ))
                )
        finally:
            fileobj.seek(loc)

    try:
        from igwn_ligolw.ligolw import Element
    except ImportError:
        return False
    return len(args) > 0 and isinstance(args[0], Element)
