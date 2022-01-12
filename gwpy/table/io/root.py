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


def _get_treename(source, trees):
    """Return the name of the only tree in the trees

    Raises
    ------
    ValueError
        if multiple trees are found
    """
    if not trees:  # nothing to read?
        raise ValueError("No trees found in %s" % source)

    # get list of all names
    names = [
        name.decode("utf-8") if isinstance(name, bytes) else name
        for name in trees
    ]
    if len(names) == 1:
        return names[0]
    raise ValueError(
        "Multiple trees found in {}, please select one via the "
        "`treename` keyword argument, e.g. `treename='events'`. "
        "Available trees are: '{}'.".format(
            source,
            "', '".join(names),
        ),
    )


@read_with_columns
@read_with_selection
def table_from_root(source, treename=None, **kwargs):
    """Read a Table from a ROOT tree
    """
    import uproot

    path = file_path(source)
    with uproot.open(path) as trees:
        # find tree name
        if treename is None:
            treename = _get_treename(path, trees)

        # read branches from tree
        try:
            return Table(trees[treename].arrays(library="np"), **kwargs)
        except TypeError:  # uproot < 4
            return Table(trees[treename].arrays(namedecode="utf-8"), **kwargs)


def table_to_root(
        table,
        filename,
        treename="tree",
        overwrite=False,
        append=False,
        **kwargs,
):
    """Write a Table to a ROOT file
    """
    import uproot

    createkw = {
        k: kwargs.pop(k) for k in {
            "initial_directory_bytes",
            "uuid_function",
        } if k in kwargs
    }
    if append:
        create_func = uproot.update
    elif overwrite:
        create_func = uproot.recreate
    else:
        create_func = uproot.create

    with create_func(filename, **createkw) as outf:
        tree = outf.mktree(
            treename,
            dict(table.dtype.descr),
            **kwargs,
        )
        tree.extend(dict(table.columns))


# register I/O
for table_class in (Table, EventTable):
    registry.register_reader('root', table_class, table_from_root)
    registry.register_writer('root', table_class, table_to_root)
    registry.register_identifier('root', table_class,
                                 identify_factory('.root'))
