# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Utilies for filtering a `Table` using column slice definitions
"""

import operator
import re
import token
from collections import OrderedDict
from io import StringIO
from tokenize import generate_tokens

import numpy

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OPERATORS = OrderedDict([
    ('<', operator.lt),
    ('<=', operator.le),
    ('=', operator.eq),
    ('==', operator.eq),
    ('>=', operator.ge),
    ('>', operator.gt),
    ('!=', operator.ne),
])

OPERATORS_INV = OrderedDict([
    ('<=', operator.ge),
    ('<', operator.gt),
    ('>', operator.lt),
    ('>=', operator.le),
])

QUOTE_REGEX = re.compile(r'^[\s\"\']+|[\s\"\']+$')
DELIM_REGEX = re.compile(r'(and|&+)', re.I)


# -- filter parsing -----------------------------------------------------------

def _float_or_str(value):
    """Internal method to attempt `float(value)` handling a `ValueError`
    """
    # remove any surrounding quotes
    value = QUOTE_REGEX.sub('', value)
    try:  # attempt `float()` conversion
        return float(value)
    except ValueError:  # just return the input
        return value


def parse_operator(mathstr):
    """Parse a `str` as a function from the `operator` module

    Parameters
    ----------
    mathstr : `str`
        a `str` representing a mathematical operator

    Returns
    -------
    op : `func`
        a callable `operator` module function

    Raises
    ------
    KeyError
        if input `str` cannot be mapped to an `operator` function

    Examples
    --------
    >>> parse_operator('>')
    <built-in function gt>
    """
    try:
        return OPERATORS[mathstr]
    except KeyError as exc:
        exc.args = ('Unrecognised operator %r' % mathstr,)
        raise


def parse_column_filter(definition):
    """Parse a `str` of the form 'column>50'

    Parameters
    ----------
    definition : `str`
        a column filter definition of the form ``<name><operator><threshold>``
        or ``<threshold><operator><name><operator><threshold>``, e.g.
        ``frequency >= 10``, or ``50 < snr < 100``

    Returns
    -------
    filters : `list` of `tuple`
        a `list` of filter 3-`tuple`s, where each `tuple` contains the
        following elements:

        - ``column`` (`str`) - the name of the column on which to operate
        - ``operator`` (`callable`) - the operator to call when evaluating
          the filter
        - ``operand`` (`anything`) - the argument to the operator function

    Raises
    ------
    ValueError
        if the filter definition cannot be parsed

    KeyError
        if any parsed operator string cannnot be mapped to a function from
        the `operator` module

    Notes
    -----
    Strings that contain non-alphanumeric characters (e.g. hyphen `-`) should
    be quoted inside the filter definition, to prevent such characters
    being interpreted as operators, e.g. ``channel = X1:TEST`` should always
    be passed as ``channel = "X1:TEST"``.

    Examples
    --------
    >>> parse_column_filter("frequency>10")
    [('frequency', <function operator.gt>, 10.)]
    >>> parse_column_filter("50 < snr < 100")
    [('snr', <function operator.gt>, 50.), ('snr', <function operator.lt>, 100.)]
    >>> parse_column_filter("channel = "H1:TEST")
    [('channel', <function operator.eq>, 'H1:TEST')]
    """  # noqa
    # parse definition into parts (skipping null tokens)
    parts = list(generate_tokens(StringIO(definition.strip()).readline))
    while parts[-1][0] in (token.ENDMARKER, token.NEWLINE):
        parts = parts[:-1]

    # parse simple definition: e.g: snr > 5
    if len(parts) == 3:
        a, b, c = parts  # pylint: disable=invalid-name
        if a[0] in [token.NAME, token.STRING]:  # string comparison
            name = QUOTE_REGEX.sub('', a[1])
            oprtr = OPERATORS[b[1]]
            value = _float_or_str(c[1])
            return [(name, oprtr, value)]
        elif b[0] in [token.NAME, token.STRING]:
            name = QUOTE_REGEX.sub('', b[1])
            oprtr = OPERATORS_INV[b[1]]
            value = _float_or_str(a[1])
            return [(name, oprtr, value)]

    # parse between definition: e.g: 5 < snr < 10
    elif len(parts) == 5:
        a, b, c, d, e = list(zip(*parts))[1]  # pylint: disable=invalid-name
        name = QUOTE_REGEX.sub('', c)
        return [(name, OPERATORS_INV[b], _float_or_str(a)),
                (name, OPERATORS[d], _float_or_str(e))]

    raise ValueError("Cannot parse filter definition from %r" % definition)


def parse_column_filters(*definitions):
    """Parse multiple compound column filter definitions

    Examples
    --------
    >>> parse_column_filters('snr > 10', 'frequency < 1000')
    [('snr', <function operator.gt>, 10.), ('frequency', <function operator.lt>, 1000.)]
    >>> parse_column_filters('snr > 10 && frequency < 1000')
    [('snr', <function operator.gt>, 10.), ('frequency', <function operator.lt>, 1000.)]
    """  # noqa: E501
    fltrs = []
    for def_ in _flatten(definitions):
        if is_filter_tuple(def_):
            fltrs.append(def_)
        else:
            for splitdef in DELIM_REGEX.split(def_)[::2]:
                fltrs.extend(parse_column_filter(splitdef))
    return fltrs


def _flatten(container):
    """Flatten arbitrary nested list of filters into a 1-D list
    """
    if isinstance(container, str):
        container = [container]
    for elem in container:
        if isinstance(elem, str) or is_filter_tuple(elem):
            yield elem
        else:
            for elem2 in _flatten(elem):
                yield elem2


def is_filter_tuple(tup):
    """Return whether a `tuple` matches the format for a column filter
    """
    try:
        names, func, args = tup
        return (
            (isinstance(names, str) or all(isinstance(x, str) for x in names))
            and callable(func)
        )
    except (TypeError, ValueError):
        return False


# -- filter -------------------------------------------------------------------

def filter_table(table, *column_filters):
    """Apply one or more column slice filters to a `Table`

    Multiple column filters can be given, and will be applied
    concurrently

    Parameters
    ----------
    table : `~astropy.table.Table`
        the table to filter

    column_filter : `str`, `tuple`
        a column slice filter definition, in one of two formats:

        - `str` - e.g. ``'snr > 10``
        - `tuple` - ``(<column(s)>, <operator>, <operand>)``, e.g.
          ``('snr', operator.gt, 10)``

        multiple filters can be given and will be applied in order

    Returns
    -------
    table : `~astropy.table.Table`
        a view of the input table with only those rows matching the filters

    Examples
    --------
    >>> filter(my_table, 'snr>10', 'frequency<1000')

    custom operations can be defined using filter tuple definitions:

    >>> from gwpy.table.filters import in_segmentlist
    >>> filter(my_table, ('time', in_segmentlist, segs))
    """
    keep = numpy.ones(len(table), dtype=bool)
    for name, op_func, operand in parse_column_filters(*column_filters):
        col = table[name]
        keep &= op_func(col, operand)
    return table[keep]
