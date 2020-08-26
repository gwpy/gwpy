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

"""Read events from ROOT trees into Tables
"""

from ...io import registry
from ...io.utils import (file_path, identify_factory)
from .. import (Table, EventTable)
from .utils import (read_with_columns, read_with_selection)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@read_with_columns
@read_with_selection
def table_from_root(source, treename=None, **kwargs):
    """Read a Table from a ROOT tree
    """
    import uproot

    with uproot.open(file_path(source)) as trees:
        # find single tree (if only one tree present)
        if treename is None:
            names = [name.decode("utf-8") for name in trees]
            if len(names) == 1:
                treename = names[0]
            elif not trees:
                raise ValueError("No trees found in %s" % source)
            else:
                raise ValueError(
                    "Multiple trees found in {}, please select one via the "
                    "`treename` keyword argument, e.g. `treename='events'`. "
                    "Available trees are: '{}'.".format(
                        source,
                        "', '".join(names),
                    ),
                )

        return Table(trees[treename].arrays(namedecode="utf-8"), **kwargs)


def table_to_root(table, filename, treename="tree",
                  overwrite=False, append=False, **kwargs):
    """Write a Table to a ROOT file
    """
    import uproot

    createkw = {k: kwargs.pop(k) for k in {"compression", } if k in kwargs}
    create_func = uproot.recreate if overwrite else uproot.create

    if append is True:
        raise NotImplementedError(
            "uproot currently doesn't support appending to existing files",
        )

    tree = uproot.newtree(dict(table.dtype.descr), **kwargs)

    with create_func(filename, **createkw) as outf:
        outf[treename] = tree
        outf[treename].extend(dict(table.columns))


# register I/O
for table_class in (Table, EventTable):
    registry.register_reader('root', table_class, table_from_root)
    registry.register_writer('root', table_class, table_to_root)
    registry.register_identifier('root', table_class,
                                 identify_factory('.root'))
