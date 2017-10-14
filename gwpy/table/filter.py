# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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
import token
import re
from tokenize import generate_tokens
from collections import OrderedDict

from six.moves import StringIO

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

re_quote = re.compile(r'^[\s\"\']+|[\s\"\']+$')
re_delim = re.compile(r'(and|&+)', re.I)


def _float_or_str(value):
    """Internal method to attempt `float(value)` handling a `ValueError`
    """
    # remove any surrounding quotes
    value = re_quote.sub('', value)
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
    except KeyError as e:
        e.args = ('Unrecognised operator %r' % mathstr,)
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
    column : `str`
        the name of the column on which to operate

    math : `list` of (`str`, `callable`) pairs
        the list of thresholds and their associated math operators

    Raises
    ------
    ValueError
        if the filter definition cannot be parsed

    KeyError
        if any parsed operator string cannnot be mapped to a function from
        the `operator` module

    Examples
    --------
    >>> parse_column_filter("frequency>10")
    ('frequency', [(10.0, <built-in function gt>)])
    >>> parse_column_filter("50 < snr < 100")
    ('snr', [(50.0, <built-in function ge>), (100.0, <build-in function lt>)])
    """
    # parse definition into parts
    parts = list(generate_tokens(StringIO(definition.strip()).readline))
    if parts[-1][0] == token.ENDMARKER:  # remove end marker
        parts = parts[:-1]

    # parse simple definition: e.g: snr > 5
    if len(parts) == 3:
        a, b, c = parts
        if a[0] in [token.NAME, token.STRING]:  # string comparison
            name = re_quote.sub('', a[1])
            op = OPERATORS[b[1]]
            value = _float_or_str(c[1])
        elif b[0] in [token.NAME, token.STRING]:
            name = re_quote.sub('', b[1])
            op = OPERATORS_INV[b[1]]
            value = _float_or_str(a[1])
        return name, [(value, op)]

    # parse between definition: e.g: 5 < snr < 10
    elif len(parts) == 5:
        a, b, c, d, e = zip(*parts)[1]
        return re_quote.sub('', c), [(_float_or_str(a), OPERATORS_INV[b]),
                                     (_float_or_str(e), OPERATORS[d])]

    raise ValueError("Cannot parse filter definition from %r" % definition)


def parse_column_filters(*definitions):
    """Parse multiple compound column filter definitions

    Examples
    --------
    >>> parse_column_filters('snr > 10', 'frequency < 1000')
    [('snr', [(10.0, <built-in function gt>)]),
     ('frequency', [(1000.0, <built-in function lt>)])]
    >>> parse_column_filters('snr > 10 && frequency < 1000')
    [('snr', [(10.0, <built-in function gt>)]),
     ('frequency', [(1000.0, <built-in function lt>)])]
    """
    fltrs = []
    for def_ in _flatten(definitions):
        for splitdef in re_delim.split(def_)[::2]:
            fltrs.append(parse_column_filter(splitdef))
    return fltrs


def _flatten(container):
    """Flatten arbitrary nested list into strings
    """
    for i in container:
        if isinstance(i, (list, tuple)):
            for j in _flatten(i):
                yield j
        else:
            yield i


def filter_table(table, *column_filters):
    """Apply one or more column slice filters to a `Table`

    Multiple column filters can be given, and will be applied
    concurrently

    Parameters
    ----------
    table : `~astropy.table.Table`
        the table to filter

    column_filter : `str`
        a column slice filter definition, e.g. ``'snr > 10``

    Returns
    -------
    table : `~astropy.table.Table`
        a view of the input table with only those rows matching the filters

    Examples
    --------
    >>> filter(my_table, 'snr>10', 'frequency<1000')
    """
    keep = numpy.ones(len(table), dtype=bool)
    for name, math in parse_column_filters(*column_filters):
        col = table[name].view(numpy.ndarray)
        for threshold, oprtr in math:
            keep &= oprtr(col, threshold)
    return table[keep]
