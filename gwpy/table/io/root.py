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

from ...io import registry
from ...io.utils import identify_factory
from ...io.cache import (file_list, read_cache)
from ...utils import with_import
from .. import Table

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@with_import('root_numpy')
def table_from_root(f, tree, columns=None, **kwargs):
    return Table(root_numpy.root2array(file_list(f), tree,
                                            branches=columns, **kwargs))


@with_import('root_numpy')
def table_to_root(table, filename, **kwargs):
    root_numpy.array2root(table.as_array(), filename, **kwargs)


# register I/O
registry.register_reader('root', Table, table_from_root)
registry.register_writer('root', Table, table_to_root)
registry.register_identifier('root', Table, identify_factory('.root'))
