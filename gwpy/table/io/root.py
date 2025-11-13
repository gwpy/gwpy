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

"""Read events from ROOT trees into Tables."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ...io.root import identify_root
from ...io.utils import file_path
from .. import (
    EventTable,
    Table,
)
from .utils import read_with_columns_and_where

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

    import uproot

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def get_treename(
    rootdir: uproot.ReadOnlyDirectory,
) -> str:
    """Return the name of the only tree in this root directory.

    Parameters
    ----------
    rootdir : `uproot.reading.ReadOnlyDirectory`
        The ROOT object to inspect.

    Raises
    ------
    ValueError
        If multiple trees are found.
    """
    source = rootdir.file_path

    if not rootdir:  # nothing to read?
        msg = "No trees found in '{source}'"
        raise ValueError(msg)

    # find one and only one tree
    try:
        tree, = rootdir.iterkeys(
            recursive=True,
            cycle=False,
        )
    except ValueError as exc:
        msg = (
            f"Multiple trees found in {source}, please select one via the "
            "`treename` keyword argument, e.g. `treename='events'`. "
            "Available trees are: '{', '.join(names)}'."
        )
        raise ValueError(msg) from exc
    return tree


@read_with_columns_and_where
def table_from_root(
    source: str | Path | IO,
    treename: str | None = None,
    **kwargs,
) -> Table:
    """Read a Table from a ROOT file.

    Requires: :doc:`uproot <uproot:index>`

    Parameters
    ----------
    source : `str`, `pathlib.Path`
        The file path or object from which to read. See `uproot.open`
        for details on acceptable inputs.

    treename : `str`, optional
        The name of the ``TTree`` to read.
        Required unless ``source`` contains exactly one ``TTree``.

    columns : `list` of `str`, optional
        List of column names to read.

    where : `str`, `list` of `str`, optional
        One or more column filters with which to downselect the
        returned table rows as they as read, e.g. ``'snr > 5'``,
        similar to a SQL ``WHERE`` statement.
        Multiple conditions should be connected by ' && ' or ' and ',
        or given as a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``.

    kwargs
        All other keyword arguments are either passed to
        `uproot.open` or to the `~astropy.table.Table` constructor.

    Raises
    ------
    ValueError
        If ``treename=None`` is given and multiple trees exist in
        the ``source``.

    KeyError
        If ``treename`` is given but no tree is found with that name.

    See Also
    --------
    uproot.open
        For details of how ROOT files are parsed and what keyword
        arguments should be supported.

    astropy.table.Table
        For details of the keyword arguments supported when creating tables.
    """
    import uproot

    # handle uproot.open keywords
    createkw = {
        # valid for uproot 5.5.1
        k: kwargs.pop(k) for k in (
            "object_cache",
            "array_cache",
            "custom_classes",
            "decompression_executor",
            "interpretation_executor",
            "handler",
            "timeout",
            "max_num_elements"
            "num_workers"
            "use_threads"
            "num_fallback_workers"
            "begin_chunk_size"
            "minimal_ttree_metadata",
        ) if k in kwargs
    }

    path = file_path(source)
    with uproot.open(path, **createkw) as rootdir:
        # find tree name
        if treename is None:
            treename = get_treename(rootdir)

        # read branches from tree
        return Table(rootdir[treename].arrays(library="np"), **kwargs)


def table_to_root(  # noqa: D417
    table: Table,
    filename: str | Path | IO,
    treename: str = "tree",
    *,
    overwrite: bool = False,
    append: bool =False,
    **kwargs,
) -> None:
    """Write a Table to a ROOT file.

    Requires: :doc:`uproot <uproot:index>`

    Parameters
    ----------
    filename : `str`, `pathlib.Path`
        Filename or object to write to.

    treename : `str`
        Name of the ROOT ``TTree`` to create for this table.

    overwrite : `bool`
        If `True` over-write an existing file of the same name.
        Default is `False`.

    append : `bool`
        If `True` append a new ``TTree`` to an existing file.
        Default is `False`.

    kwargs
        All other keyword arguments are passed to the relevant
        `uproot` functions for creating/updating a file and
        writing a ``TTree``.

    Raises
    ------
    FileExistsError
        If ``overwrite=False, append=False`` is given and the
        target filename already exists.

    OSError
        If ``append=True`` is given and the target filename does not
        already exist.

    See Also
    --------
    uproot.create
        For details of how new files are created and what keyword
        arguments should be supported.
        This is called if ``overwrite=False, append=False`` is given.

    uproot.update
        For details of how existing files are updated and what keyword
        arguments should be supported.
        This is called if ``append=True`` is given.

    uproot.recreate
        For details of how existing files are over-written and what keyword
        arguments should be supported.
        This is called if ``overwrite=True, append=False`` is given.

    uproot.writing.writable.WritableDirectory.mktree
        For deatils of how ``TTree`` objects are created and what keyword
        arguments should be supported.
    """
    import uproot

    # handle file creation/update options
    createkw = {
        k: kwargs.pop(k) for k in (
            "initial_directory_bytes",
            "initial_streamers_bytes",
            "uuid_function",
            "compression",
        ) if k in kwargs
    }
    if overwrite:
        create_func = uproot.recreate
    elif append:
        create_func = uproot.update
    else:
        create_func = uproot.create

    # create file
    with create_func(filename, **createkw) as outf:
        # create the tree
        tree = outf.mktree(
            treename,
            dict(table.dtype.descr),
            **kwargs,
        )
        # add data to it
        tree.extend(dict(table.columns))


# register I/O
for klass in (Table, EventTable):
    klass.read.registry.register_identifier("root", klass, identify_root)
    klass.read.registry.register_reader("root", klass, table_from_root)
    klass.write.registry.register_writer("root", klass, table_to_root)
