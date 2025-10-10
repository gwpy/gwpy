# Copyright (c) 2017-2025 Cardiff University
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

"""Utilies for filtering a `Table` using column slice definitions."""

from __future__ import annotations

import operator
import re
import token
from io import StringIO
from tokenize import generate_tokens
from typing import (
    TYPE_CHECKING,
    Generic,
    NamedTuple,
    TypeVar,
    cast,
)

import numpy

FilterOperandType = TypeVar("FilterOperandType")

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )

    from astropy.table import Column

    from . import Table

    T = TypeVar("T", bound=Table)

    FilterTuple = tuple[
        str | tuple[str, ...],
        Callable[[Table | Column, FilterOperandType], numpy.ndarray],
        FilterOperandType,
    ]
    FilterLike = str | FilterTuple

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

OPERATORS: dict[str, Callable] = {
    "<": operator.lt,
    "<=": operator.le,
    "=": operator.eq,
    "==": operator.eq,
    ">=": operator.ge,
    ">": operator.gt,
    "!=": operator.ne,
}

OPERATORS_INV: dict[str, Callable] = {
    "<=": operator.ge,
    "<": operator.gt,
    ">": operator.lt,
    ">=": operator.le,
}

QUOTE_REGEX = re.compile(r"^[\s\"\']+|[\s\"\']+$")
DELIM_REGEX = re.compile(r"(and|&+)", re.IGNORECASE)


# -- filter parsing ------------------

FILTER_SIMPLE_ARG_COUNT = 3
FILTER_COMPOUND_ARG_COUNT = 5

class FilterSpec(NamedTuple, Generic[FilterOperandType]):
    """A table column filter definition.

    The generic type parameter _O represents the type of the operand,
    which should match the second parameter of the operator callable.
    """

    column: str | tuple[str, ...]
    operator: Callable[[Table | Column, FilterOperandType], numpy.ndarray]
    operand: FilterOperandType


def _float_or_str(value: str) -> float | str:
    """Attempt `float(value)` handling a `ValueError`."""
    # remove any surrounding quotes
    value = QUOTE_REGEX.sub("", value)
    try:  # attempt `float()` conversion
        return float(value)
    except ValueError:  # just return the input
        return value


def parse_operator(mathstr: str) -> Callable:
    """Parse a `str` as a function from the `operator` module.

    Parameters
    ----------
    mathstr : `str`
        A `str` representing a mathematical operator.

    Returns
    -------
    op : `func`
        One of the functions from the `operator` module.

    Raises
    ------
    KeyError
        If input `str` cannot be mapped to an `operator` function.

    Examples
    --------
    >>> parse_operator('>')
    <built-in function gt>
    """
    try:
        return OPERATORS[mathstr]
    except KeyError as exc:
        exc.args = (
            f"Unrecognised operator '{mathstr}'",
        )
        raise


def parse_column_filter(definition: str) -> list[FilterSpec]:
    """Parse a `str` of the form 'column>50'.

    Parameters
    ----------
    definition : `str`
        a column filter definition of the form ``<name><operator><threshold>``
        or ``<threshold><operator><name><operator><threshold>``, e.g.
        ``frequency >= 10``, or ``50 < snr < 100``

    Returns
    -------
    filters : `list` of `FilterSpec`
        A `list` of filter 3-`tuple`s, where each `tuple` contains the
        following elements:

        - ``column`` (`str`) - the name of the column on which to operate
        - ``operator`` (`callable`) - the operator to call when evaluating
          the filter
        - ``operand`` (anything) - the argument to the operator function

    Raises
    ------
    ValueError
        If the filter definition cannot be parsed.

    KeyError
        If any parsed operator string cannnot be mapped to a function from
        the `operator` module.

    Notes
    -----
    Strings that contain non-alphanumeric characters (e.g. hyphen `-`) should
    be quoted inside the filter definition, to prevent such characters
    being interpreted as operators, e.g. ``channel = X1:TEST`` should always
    be passed as ``channel = "X1:TEST"``.

    Examples
    --------
    >>> parse_column_filter("frequency>10")
    FilterSpec(column='frequency', operator=<built-in function gt>, operand=10.0)
    >>> parse_column_filter("50 < snr < 100")
    [FilterSpec(column='snr', operator=<built-in function gt>, operand=50.0),
     FilterSpec(column='snr', operator=<built-in function lt>, operand=100.0)]
    >>> parse_column_filter('channel = "H1:TEST"')
    FilterSpec(column='channel', operator=<built-in function eq>, operand='H1:TEST')
    """
    # parse definition into parts (skipping null tokens)
    parts = list(generate_tokens(StringIO(definition.strip()).readline))
    while parts[-1].type in {token.ENDMARKER, token.NEWLINE}:
        parts = parts[:-1]

    # parse simple definition: e.g: snr > 5 or 5 < snr
    if len(parts) == FILTER_SIMPLE_ARG_COUNT:
        left, op_, right = parts
        if left.type in {token.NAME, token.STRING}:  # snr > 5
            name = QUOTE_REGEX.sub("", left.string)
            oprtr = OPERATORS[op_.string]
            value = _float_or_str(right.string)
            return [FilterSpec(name, oprtr, value)]
        if right.type in {token.NAME, token.STRING}:  # 5 < snr
            name = QUOTE_REGEX.sub("", right.string)
            oprtr = OPERATORS_INV[op_.string]
            value = _float_or_str(left.string)
            return [FilterSpec(name, oprtr, value)]

    # parse between definition: e.g: 5 < snr < 10
    elif len(parts) == FILTER_COMPOUND_ARG_COUNT:
        left, op1, mid, op2, right = parts
        name = QUOTE_REGEX.sub("", mid.string)
        return [
            FilterSpec(
                name,
                OPERATORS_INV[op1.string],
                _float_or_str(left.string),
            ),
            FilterSpec(
                name,
                OPERATORS[op2.string],
                _float_or_str(right.string),
            ),
        ]

    msg = f"cannot parse filter definition from '{definition}'"
    raise ValueError(msg)


def parse_column_filters(
    *definitions: FilterLike | Iterable[FilterLike],
) -> list[FilterSpec]:
    """Parse multiple compound column filter definitions.

    Examples
    --------
    >>> parse_column_filters('snr > 10', 'frequency < 1000')
    [FilterSpec(column='snr', operator=<built-in function gt>, operand=10.0),
     FilterSpec(column='frequency', operator=<built-in function lt>, operand=1000.0)]
    >>> parse_column_filters('snr > 10 && frequency < 1000')
    [FilterSpec(column='snr', operator=<built-in function gt>, operand=10.0),
     FilterSpec(column='frequency', operator=<built-in function lt>, operand=1000.0)]
    """
    fltrs = []
    for def_ in _flatten(definitions):
        if isinstance(def_, str):
            for splitdef in DELIM_REGEX.split(def_)[::2]:
                fltrs.extend(parse_column_filter(splitdef))
        else:
            fltrs.append(FilterSpec(*def_))
    return fltrs


def _flatten(
    container: Iterable,
) -> Iterator[FilterLike]:
    """Flatten arbitrary nested list of filters into a 1-D list."""
    for elem in container:
        if isinstance(elem, str) or is_filter_tuple(elem):
            yield elem
        else:
            yield from _flatten(elem)


def is_filter_tuple(tup: object) -> bool:
    """Return whether a `tuple` matches the format for a column filter."""
    if isinstance(tup, FilterSpec):
        return True
    try:
        names, func, _ = cast("tuple[str | tuple[str, ...], Callable, tuple]", tup)  # type: ignore[misc]
        return (
            (isinstance(names, str) or all(isinstance(x, str) for x in names))
            and callable(func)
        )
    except (TypeError, ValueError):
        return False


# -- filter --------------------------

def filter_table(
    table: T,
    *column_filters: FilterLike | Iterable[FilterLike],
) -> T:
    """Apply one or more column slice filters to a `Table`.

    Multiple column filters can be given, and will be applied serially.

    Parameters
    ----------
    table : `~astropy.table.Table`
        The table to filter.

    column_filters : `str`, `tuple`
        One or more column slice filter definition, in one of two formats:

        - `str` - e.g. ``'snr > 10``
        - `FilterSpec` (`tuple`) - ``(<column(s)>, <operator>, <operand>)``,
          e.g. ``('snr', operator.gt, 10)``

        Multiple filters can be given and will be applied in order.

    Returns
    -------
    table : `~astropy.table.Table`
        A view of the input table with only those rows matching the filters.

    Examples
    --------
    >>> filter(my_table, 'snr>10', 'frequency<1000')

    Custom operations can be defined using filter tuple definitions:

    >>> from gwpy.table.filters import in_segmentlist
    >>> filter(my_table, ('time', in_segmentlist, segs))
    """
    keep = numpy.ones(len(table), dtype=bool)
    for name, op_func, operand in parse_column_filters(*column_filters):
        col = table[name]
        keep &= op_func(col, operand)
    return table[keep]
