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

"""Read LIGO_LW documents into :class:`~igwn_ligolw.ligolw.Table` objects."""

from __future__ import annotations

import inspect
from functools import cache
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy
from astropy.utils.compat.numpycompat import COPY_IF_NEEDED

from ...io.ligolw import (
    is_ligolw,
    read_table as read_ligolw_table,
    to_table_type as to_ligolw_table_type,
    write_tables as write_ligolw_tables,
)
from ...segments import Segment
from .. import (
    EventTable,
    Table,
)
from .utils import (
    dynamic_columns,
    read_with_where,
)

if TYPE_CHECKING:
    from collections.abc import (
        Collection,
        Iterable,
    )
    from typing import Any

    from astropy.table import Column
    from igwn_ligolw import ligolw
    from numpy.typing import (
        ArrayLike,
        DTypeLike,
    )

    from ...io.utils import (
        NamedReadable,
        Writable,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# methods to exclude from get_as_columns conversions
GET_AS_EXCLUDE = [
    "column",
    "username",
    "table",
    "time_slide_id",
]

# mappings
NUMPY_TYPE_MAP: dict[type, type] = {}
LIGOLW_TABLES: set[type] = set()
LIGOLW_PROPERTY_TYPES: tuple[type, ...] = (property,)

try:
    from igwn_ligolw import lsctables
except ImportError:
    HAVE_LSCTABLES = False
else:
    HAVE_LSCTABLES = True
    LIGOLW_TABLES = set(lsctables.TableByName.values())
    LIGOLW_PROPERTY_TYPES = (
        *LIGOLW_PROPERTY_TYPES,
        lsctables.gpsproperty,
        lsctables.gpsproperty_with_gmst,
        lsctables.instrumentsproperty,
        lsctables.segmentproperty,
    )
    NUMPY_TYPE_MAP[lsctables.LIGOTimeGPS] = numpy.float64


# -- utilities -----------------------

@cache
def _property_columns(prop: property) -> tuple[str, ...]:
    """Return the columns associated with a property descriptor."""
    from igwn_ligolw import lsctables
    if isinstance(prop, lsctables.gpsproperty_with_gmst):
        return (prop.s_name, prop.ns_name, prop.gmst_name)
    if isinstance(prop, lsctables.gpsproperty):
        return (prop.s_name, prop.ns_name)
    if isinstance(prop, lsctables.segmentproperty):
        return (prop.start, prop.stop)
    if isinstance(prop, lsctables.instrumentsproperty):
        return (prop.name,)
    return ()


def _get_property_columns(
    tabletype: type[ligolw.Table],
    columns: Iterable[str] | None,
) -> dict[str, set[str]]:
    """Return the `set` of columns required to read properties for a table.

    Examples
    --------
    >>> _get_property_columns(lsctables.SnglBurstTable, ['peak'])
    {'peak': {'peak_time', 'peak_time_ns'}}
    """
    rowtype = tabletype.RowType

    if columns is None:  # get all property columns
        try:
            columns = next(zip(
                *inspect.getmembers(
                    rowtype,
                    predicate=lambda x: isinstance(x, LIGOLW_PROPERTY_TYPES),
                ),
                strict=True,
            ))
        except StopIteration:
            # this object doesn't have any property columns
            return {}

    # get properties for row object
    rowvars = vars(rowtype)

    # build list of real column names for fancy properties
    extracols: dict[str, set[str]] = {}
    for key in columns:
        try:
            prop = rowvars[key]
        except KeyError:
            continue
        if propcols := _property_columns(prop):
            extracols.setdefault(key, set()).update(propcols)
    return extracols


@cache
def _get_property_type(
    tabletype: type[ligolw.Table],
    column: str,
) -> type | None:
    """Return the type of values in the given property.

    Examples
    --------
    >>> _get_property_type(lsctables.SnglBurstTable, 'peak')
    lal.LIGOTimeGPS
    """
    from igwn_ligolw import lsctables
    prop = vars(tabletype.RowType)[column]
    if isinstance(prop, lsctables.gpsproperty):
        return lsctables.LIGOTimeGPS
    if isinstance(prop, lsctables.segmentproperty):
        return Segment
    if isinstance(prop, lsctables.instrumentsproperty):
        return str
    return None


@cache
def table_columns(ligolw_table_class: type[ligolw.Table]) -> list[str]:
    """Return the list of column names for this table."""
    from igwn_ligolw import ligolw
    return list(map(ligolw.Column.ColumnName, ligolw_table_class.validcolumns))


# -- conversions ---------------------

def to_astropy_table(
    llwtable: ligolw.Table,
    table_class: type[Table],
    copy: bool | None = COPY_IF_NEEDED,  # noqa: FBT001
    *,
    columns: Collection[str] | None =None,
    use_numpy_dtypes: bool = False,
    rename: dict[str, str] | None = None,
) -> Table:
    """Convert a `~igwn_ligolw.ligolw.Table` to an `~astropy.tableTable`.

    This method is designed as an internal method to be attached to
    :class:`~igwn_ligolw.ligolw.Table` objects as `__astropy_table__`.

    Parameters
    ----------
    llwtable : `~igwn_ligolw.ligolw.Table`
        The LIGO_LW table to convert from.

    table_class : `type`
        Table type to convert to.

    copy : `bool`, optional
        If `True` copy the input data, otherwise return a reference.
        Default is `False`

    columns : `list` of `str`, optional
        The columns to populate.
        Defaults to all columns.

    use_numpy_dtypes : `bool`, optional
        If `True` then force all columns in the output to have standard
        Numpy types.

    rename : `dict`, optional
        Mapping of ``('old name' -> 'new name')`` pairs to rename columns
        from the original ``LIGO_LW`` table.

    Returns
    -------
    table : `Table`
        The newly transformed table. An instance of ``table_class``.
    """
    # set default keywords
    if rename is None:
        rename = {}
    if columns is None:
        columns = llwtable.columnnames

    # extract columns from LIGO_LW table as astropy.table.Column
    data = []
    for colname in columns:
        arr = _get_column(llwtable, colname)

        # transform to astropy.table.Column
        copythis = copy or isinstance(arr, numpy.ndarray)
        data.append(to_astropy_column(
            arr,
            table_class.Column,
            copy=copythis,
            use_numpy_dtype=use_numpy_dtypes,
            name=rename.get(colname, colname),
        ))

    # build table and return
    return table_class(
        data,
        copy=copy,
        meta={"tablename": str(llwtable.Name)},
    )


def _get_column(
    llwtable: ligolw.Table,
    name: str,
) -> ligolw.Column:
    """Get a column from a ``LIGO_LW`` table."""
    # handle empty tables as special (since we can't introspect the types)
    if not len(llwtable):
        dtype: type | None
        if name in llwtable.validcolumns:
            dtype = _get_pytype(llwtable.validcolumns[name])
        else:
            dtype = _get_property_type(type(llwtable), name)  # type: ignore[arg-type]
        if dtype:
            return numpy.empty((0,), dtype=dtype)

    try:
        # try normal column access
        return llwtable.getColumnByName(name)
    except KeyError:  # bad name
        # try get_{}
        get_ = f"get_{name}"
        if (
            hasattr(llwtable, get_)
            and name not in GET_AS_EXCLUDE
        ):
            return getattr(llwtable, get_)()

        # try array of property values
        try:
            return numpy.asarray([getattr(row, name) for row in llwtable])
        except AttributeError:  # no property
            pass

        raise  # KeyError from getColumnByName


def to_astropy_column(
    llwcol: ligolw.Column,
    cls: type[Column],
    *,
    copy: bool | None = COPY_IF_NEEDED,
    dtype: DTypeLike | None = None,
    use_numpy_dtype: bool = False,
    **kwargs,
) -> Column:
    """Convert a :class:`~igwn_ligolw.ligolw.Column` to `astropy.table.Column`.

    Parameters
    ----------
    llwcol : `~igwn_ligolw.ligolw.Column`, `numpy.ndarray`, iterable
        The ``LIGO_LW`` column to convert, or an iterable.

    cls : `~astropy.table.Column`
        The Astropy `~astropy.table.Column` or subclass to convert to.

    copy : `bool`, optional
        If `True` copy the input data, otherwise return a reference.
        Default is `False`.

    dtype : `type`, optional
        The data type to convert to when creating the `~astropy.table.Column`.

    use_numpy_dtype : `bool`, optional
        If `True` then force the output column to only have standard
        Numpy types.

    kwargs
        Other keyword arguments are passed to the `~astropy.table.Column`
        creator.

    Returns
    -------
    column : `~astropy.table.Column`
        An Astropy version of the given ``LIGO_LW`` column.
        An instance of ``cls``.
    """
    if dtype is None:  # try and find dtype
        dtype = _get_column_dtype(llwcol)
        if (
            use_numpy_dtype
            and numpy.dtype(dtype).type is numpy.object_
        ):
            dtype = cast("type", dtype)
            # dtype maps to 'object' in numpy, try and resolve real numpy type
            try:
                dtype = NUMPY_TYPE_MAP[dtype]
            except KeyError as exc:
                msg = f"no mapping from object type '{dtype}' to numpy type"
                raise TypeError(msg) from exc
    return cls(
        data=llwcol,
        copy=copy,
        dtype=dtype,
        **kwargs,
    )


def _get_column_dtype(
    llwcol: ligolw.Column | ArrayLike,
) -> type | None:
    """Get the data type of a LIGO_LW `Column`.

    Parameters
    ----------
    llwcol : `~igwn_ligolw.ligolw.Column`, `numpy.ndarray`, iterable
        A ``LIGO_LW`` column, a numpy array, or an iterable.

    Returns
    -------
    dtype : `type`, `None`
        The object data type for values in the given column, `None` is
        returned if ``llwcol`` is a `numpy.ndarray` with `numpy.object_`
        dtype, or no data type can be parsed (e.g. empty list).

    Examples
    --------
    >>> _get_column_dtype(numpy.zeros(1))
    dtype('float64')
    >>> _get_column_dtype(llwtable.getColumnByName('peak_time')
    'int32'
    """
    try:
        dtype = llwcol.dtype  # type: ignore[union-attr]
        if dtype is numpy.dtype("O"):
            raise AttributeError  # goto below
    except AttributeError:
        try:  # igwn_ligolw.ligolw.Column
            name = str(llwcol.getAttribute("Name"))  # type: ignore[union-attr]
            if name.startswith(f"{llwcol.parentNode.Name}:"):  # type: ignore[union-attr]
                name = name.split(":", 1)[-1]
            llwtype = llwcol.parentNode.validcolumns[name]  # type: ignore[union-attr]
        except AttributeError:  # not a column
            try:
                return type(llwcol[0])  # type: ignore[index]
            except IndexError:
                return None
        # map column type str to python type
        return _get_pytype(llwtype)
    return dtype


def _get_pytype(llwtype: type) -> type:
    """Return a dtype-compatible type for the given LIGO_LW type string."""
    from igwn_ligolw.types import (
        ToNumPyType,
        ToPyType,
    )
    try:
        return numpy.dtype(ToNumPyType[llwtype]).type
    except KeyError:
        return ToPyType[llwtype]


def table_to_ligolw(
    table: Table,
    tablename: str,
) -> ligolw.Table:
    """Convert a `astropy.table.Table` to a `igwn_ligolw.ligolw.Table`."""
    from igwn_ligolw import lsctables

    # -- work out columns for LIGO_LW table
    # this is overly complicated because of the way that we magically
    # map properties of igwn_ligolw.ligolw.Table objects to real columns in
    # astropy.table.Table objects, but also because igwn_ligolw internally
    # combines table names and column names for some columns
    # (e.g. process:process_id)
    columns = table.columns.keys()
    cls = lsctables.TableByName[tablename]
    inst = cls.new()
    try:
        columnnamesreal = dict(zip(
            inst.columnnames,
            inst.columnnamesreal,
            strict=False,
        ))
    except AttributeError:  # glue doesn't have these attributes
        columnnamesreal = {}
    llwcolumns = [columnnamesreal.get(n, n) for n in columns]
    for col, llwcols in _get_property_columns(cls, columns).items():
        idx = llwcolumns.index(col)
        llwcolumns.pop(idx)
        for name in list(llwcols)[::-1]:
            llwcolumns.insert(idx, name)

    # create new LIGO_LW table
    llwtable = cls.new(columns=llwcolumns)

    # map rows across
    for row in table:
        llwrow = llwtable.RowType()
        for name in columns:
            setattr(llwrow, name,
                    to_ligolw_table_type(row[name], llwtable, name))
        llwtable.append(llwrow)

    return llwtable


# -- read ----------------------------

@read_with_where
def read_table(
    source: NamedReadable | ligolw.Document | list[NamedReadable],
    tablename: str | None = None,
    columns: Collection[str] | None = None,
    ligolw_columns: Collection[str] | None = None,
    **kwargs,
) -> Table:
    """Read a `Table` from one or more LIGO_LW XML documents.

    source : `file`, `str`, `~igwn_ligolw.ligolw.Document`, `list`
        One or more open files or file paths, or a single LIGO_LW ``Document``.

    tablename : `str`, optional
        The `Name` of the relevant ``LIGO_LW`` `Table` to read.
        Required if the document doesn't contain exactly one table.

    columns : `list` of `str`, optional
        The list of columns to include in the returned Table.
        Default is all columns.
        This list may include:

        - any column given in the ``validcolumns`` mapping
          for the table (in :mod:`igwn_ligolw.lsctables`)
        - any of the properties defined on the associated ``RowType`` object

    ligolw_columns : `list` of `str`, optional
        The list of columns to read from the ``LIGO_LW`` document.
        Default is all columns.
        This should be a super-set of ``columns``, including any
        ``validcolumns`` for the relevant table that are required to resolve
        dynamic columns from a row `property`.

    use_numpy_dtypes : `bool`, optional
        If `True` then force all columns in the output to have standard
        Numpy types.

    rename : `dict`, optional
        Mapping of ``('old name' -> 'new name')`` pairs to rename columns
        from the original ``LIGO_LW`` table.

    copy : `bool`, optional
        If `True` copy the input data, otherwise return a reference.
        Default is `False`

    See Also
    --------
    gwpy.io.ligolw.read_table
        For details of keyword arguments for the read operation.

    gwpy.table.io.ligolw.to_astropy_table
        For details of keyword arguments for the conversion operation.
    """
    from igwn_ligolw import ligolw
    from igwn_ligolw.lsctables import TableByName

    # -- keyword handling ------------

    # separate keywords for reading from LIGO_LW and converting to Astropy
    convert_kw: dict[str, Any] = {
        "rename": None,
        "use_numpy_dtypes": False,
    }
    for key in filter(kwargs.__contains__, convert_kw):
        convert_kw[key] = kwargs.pop(key)
    if convert_kw["rename"] is None:
        convert_kw["rename"] = {}
    read_kw = kwargs  # rename for readability

    # allow user to specify LIGO_LW columns to read to provide the
    # desired output columns
    read_kw["columns"] = ligolw_columns or columns
    convert_kw["columns"] = columns or read_kw["columns"]

    if tablename:
        tableclass = TableByName[ligolw.Table.TableName(tablename)]
        read_kw["columns"] = dynamic_columns(
            convert_kw["columns"],
            table_columns(tableclass),
            _get_property_columns(tableclass, None),
        )[0]

    # -- read ------------------------

    return Table(
        read_ligolw_table(
            source,
            tablename=tablename,
            **read_kw,
        ),
        **convert_kw,
    )


# -- write ---------------------------

def write_table(  # noqa: D417
    table: Table,
    target: Writable | ligolw.Document,
    tablename: str | None = None,
    **kwargs,
) -> None:
    """Write a `~astropy.table.Table` to file in LIGO_LW XML format.

    Parameters
    ----------
    target : `str`, `pathlib.Path`, `file`
        The file path or file object to write to.

    tablename : `str`, optional
        The Name of the ``LIGO_LW`` table to write to.
        Default is taken from the ``tabename`` metadata attribute of
        the table being written, otherwise this argument is required.

    kwargs
        All keyword arguments are passed to
        `gwpy.io.ligolw.write_tables`.

    See Also
    --------
    gwpy.io.ligolw.write_tables
        For details of the table writing implementation and any
        valid keyword arguments.
    """
    if tablename is None:  # try and get tablename from metadata
        tablename = table.meta.get("tablename", None)
    if tablename is None:  # panic
        msg = (
            "please pass ``tablename=`` to specify the target "
            "LIGO_LW Table Name"
        )
        raise ValueError(msg)
    llwtable = table_to_ligolw(
        table,
        tablename,
    )
    return write_ligolw_tables(target, [llwtable], **kwargs)


# -- register ------------------------

for table_ in LIGOLW_TABLES:
    # register conversion from LIGO_LW to astropy Table
    table_.__astropy_table__ = to_astropy_table  # type: ignore[attr-defined]

for klass in (Table, EventTable):
    registry = klass.read.registry
    registry.register_reader("ligolw", klass, read_table)
    registry.register_writer("ligolw", klass, write_table)
    registry.register_identifier("ligolw", klass, is_ligolw)
