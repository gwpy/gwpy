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
from tokenize import generate_tokens

from six.moves import StringIO

import numpy

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OPERATORS = {
    '<': operator.lt,
    '<=': operator.le,
    '==': operator.eq,
    '>=': operator.ge,
    '>': operator.gt,
    '!=': operator.ne,
}

OPERATORS_INV = {
    '<': operator.ge,
    '<=': operator.gt,
    '>=': operator.lt,
    '>': operator.le,
}


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
    parts = list(generate_tokens(StringIO(definition).readline))

    # find column name
    names = filter(lambda t: t[0] == token.NAME, parts)
    if len(names) != 1:
        raise ValueError("Multiple column names parsed from "
                         "column filter definition %r" % definition)
    name = names[0][1]

    # parse thresholds and operators and divide into single operation pairs
    thresholds = list(zip(*filter(lambda t: t[0] == token.NUMBER, parts)))[1]
    operators = list(zip(*filter(lambda t: t[0] == token.OP, parts)))[1]
    if len(thresholds) != len(operators):  # sanity check
        ValueError("Number of thresholds doesn't match number of operators "
                   "in column filter definition %r" % definition)
    math = []
    for lim, op in zip(thresholds, operators):
        try:
            # parse '1 < snr' as 'snr >= 1'
            if (definition.find(lim) < definition.find(op) and
                    op in OPERATORS_INV):
                math.append((float(lim), OPERATORS_INV[op]))
            else:
                math.append((float(lim), OPERATORS[op]))
        except KeyError as e:
            e.args = ('Unrecognised math operator %r' % op,)
            raise

    return name, math


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
    for def_ in definitions:
        for splitdef in def_.replace('&&', '&').split('&'):
            fltrs.append(parse_column_filter(splitdef))
    return fltrs


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
