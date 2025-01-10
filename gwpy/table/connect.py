# Copyright (C) Louisiana State University (2017)
#               Cardiff University (2017-)
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

"""Unified I/O read/write for Table objects.
"""

from __future__ import annotations

import typing

from astropy.io.registry.compat import default_registry as registry
from astropy.table import vstack

from ..io.registry import (
    UnifiedRead,
    UnifiedWrite,
)

if typing.TYPE_CHECKING:
    from .table import EventTable

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class EventTableRead(UnifiedRead):
    """Read data into an `EventTable`.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    args
        Other positional arguments will be passed directly to the
        underlying reader method for the given format.

    format : `str`, optional
        The format of the given source files; if not given, an attempt
        will be made to automatically identify the format.

    columns : `list` of `str`, optional
        The list of column names to read.

    selection : `str`, or `list` of `str`, optional
        One or more column filters with which to downselect the
        returned table rows as they as read, e.g. ``'snr > 5'``;
        multiple selections should be connected by ' && ', or given as
        a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
        ``['snr > 5', 'frequency < 1000']``

    parallel : `int`
        Number of threads to use for parallel reading of multiple files.

    verbose : `bool`
        Print a progress bar showing read status, default: `False`.

    .. note::

       Keyword arguments other than those listed here may be required
       depending on the `format`

    Returns
    -------
    table : `EventTable`

    Raises
    ------
    astropy.io.registry.IORegistryError
        If the `format` cannot be automatically identified.
    IndexError
        If ``source`` is an empty list.

    Notes
    -----"""
    def __init__(self, instance, cls):
        super().__init__(
            instance,
            cls,
            registry=registry,
        )

    def __call__(
        self,
        *args,
        **kwargs,
    ) -> EventTable:
        return super().__call__(
            vstack,
            *args,
            **kwargs,
        )


class EventTableWrite(UnifiedWrite):
    """Write this table to a file

    Parameters
    ----------
    target: `str`
        Filename for output data file.

    *args
        Other positional arguments will be passed directly to the
        underlying writer method for the given format.

    format : `str`
        Format for output data; if not given, an attempt will be made
        to automatically identify the format based on the `target`
        filename.

    **kwargs
        Other keyword arguments will be passed directly to the
        underlying writer method for the given format.

    Raises
    ------
    astropy.io.registry.IORegistryError
        If the `format` cannot be automatically identified.

    Notes
    -----"""
    def __init__(self, instance, cls):
        super().__init__(
            instance,
            cls,
            registry=registry,
        )
