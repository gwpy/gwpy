# Copyright (C) Cardiff University (2025-)
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

"""Unified I/O read/write for `Channel` and `ChannelList` objects."""

from __future__ import annotations

import typing

from ..io.registry import (
    UnifiedRead,
    UnifiedWrite,
)

if typing.TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path
    from typing import IO

    from . import ChannelList


class ChannelListRead(UnifiedRead):
    """Read a `ChannelList` from a file.

    Parameters
    ----------
    source : `str`, `pathlib.Path`, `file`.
        File or path to read.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    kwargs
        Other keyword arguments depend on the format.

    Returns
    -------
    channels : `ChannelList`
        The list of channels read from the source.

    Notes
    -----"""

    def __call__(
        self,
        source: str | Path | IO | Iterable[str | Path | IO],
        **kwargs,
    ) -> ChannelList:
        """Read a `ChannelList`."""
        def combiner(listoflists: list[ChannelList]) -> ChannelList:
            """Combine the `ChannelList`` from each file into a single object."""
            return self._cls(c for clist in listoflists for c in clist)

        return super().__call__(
            combiner,
            source,
            **kwargs,
        )


class ChannelListWrite(UnifiedWrite):
    """Write this `ChannelList` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format.

    Parameters
    ----------
    target : `str`
        Output filename.

    Notes
    -----"""
