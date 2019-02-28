# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""Read events from ROOT trees into Tables
"""

from six import string_types

from ...io import registry
from ...io.utils import identify_factory
from ..filter import (OPERATORS, parse_column_filters, filter_table)
from .. import (Table, EventTable)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def table_from_root(source, treename=None, columns=None, **kwargs):
    """Read a Table from a ROOT tree
    """
    import root_numpy

    # parse column filters into tree2array ``selection`` keyword
    # NOTE: not all filters can be passed directly to root_numpy, so we store
    #       those separately and apply them after-the-fact before returning
    try:
        selection = kwargs.pop('selection')
    except KeyError:  # no filters
        filters = None
    else:
        rootfilters = []
        filters = []
        for col, op_, value in parse_column_filters(selection):
            try:
                opstr = [key for key in OPERATORS if OPERATORS[key] is op_][0]
            except (IndexError, KeyError):  # cannot filter with root_numpy
                filters.append((col, op_, value))
            else:  # can filter with root_numpy
                rootfilters.append('{0} {1} {2!r}'.format(col, opstr, value))
        kwargs['selection'] = ' && '.join(rootfilters)

    # pass file name (not path)
    if not isinstance(source, string_types):
        source = source.name

    # find single tree (if only one tree present)
    if treename is None:
        trees = root_numpy.list_trees(source)
        if len(trees) == 1:
            treename = trees[0]
        elif not trees:
            raise ValueError("No trees found in %s" % source)
        else:
            raise ValueError("Multiple trees found in %s, please select on "
                             "via the `treename` keyword argument, e.g. "
                             "`treename='events'`. Available trees are: %s."
                             % (source, ', '.join(map(repr, trees))))

    # read, filter, and return
    t = Table(root_numpy.root2array(
        source,
        treename,
        branches=columns,
        **kwargs
    ))
    if filters:
        return filter_table(t, *filters)
    return t


def table_to_root(table, filename, **kwargs):
    """Write a Table to a ROOT file
    """
    import root_numpy
    root_numpy.array2root(table.as_array(), filename, **kwargs)


# register I/O
for table_class in (Table, EventTable):
    registry.register_reader('root', table_class, table_from_root)
    registry.register_writer('root', table_class, table_to_root)
    registry.register_identifier('root', table_class,
                                 identify_factory('.root'))
