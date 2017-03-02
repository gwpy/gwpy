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

"""Read events from ROOT trees into Tables
"""

import warnings

from ...io import registry
from ...io.utils import identify_factory
from ...io.cache import (file_list, read_cache)
from ...utils import with_import
from .. import (Table, EventTable)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@with_import('root_numpy')
def table_from_root(f, treename=None, include_names=None, **kwargs):
    if include_names is None:
        try:
            include_names = kwargs.pop('columns')
        except KeyError:
            pass
        else:
            warnings.warn("Keyword argument `columns` has been renamed to "
                          "`include_names` to better match default "
                          "astropy.table.Table.read kwargs, please update "
                          "your call.", DeprecationWarning)

    files = file_list(f)
    if treename is None:
        trees = root_numpy.list_trees(files[0])
        if len(trees) == 1:
            treename = trees[0]
        elif len(trees) == 0:
            raise ValueError("No trees found in %s" % files[0])
        else:
            raise ValueError("Multiple trees found in %s, please select on "
                             "via the `treename` keyword argument, e.g. "
                             "`treename='events'`. Available trees are: %s."
                             % (files[0], ', '.join(map(repr, trees))))

    return Table(root_numpy.root2array(files, treename, branches=include_names,
                                       **kwargs))


@with_import('root_numpy')
def table_to_root(table, filename, **kwargs):
    root_numpy.array2root(table.as_array(), filename, **kwargs)


# register I/O
_identifier = identify_factory('.root')
for table_class in (Table, EventTable):
    registry.register_reader('root', table_class, table_from_root)
    registry.register_writer('root', table_class, table_to_root)
    registry.register_identifier('root', table_class, _identifier)
