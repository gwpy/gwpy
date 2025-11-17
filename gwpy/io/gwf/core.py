# Copyright (c) 2024-2025 Cardiff University
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

"""Core I/O utilities for GWF files."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ...segments import SegmentList
from ..cache import read_cache
from .backend import get_backend_function

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from typing import (
        IO,
        Literal,
    )

    from ...detector import Channel
    from ...io.utils import (
        FileLike,
        FileSystemPath,
    )
    from ...types import Series

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "channel_exists",
    "data_segments",
    "get_channel_names",
    "get_channel_type",
    "identify_gwf",
    "iter_channel_names",
    "num_channels",
]

# first 4 bytes of any valid GWF file (see LIGO-T970130 §4.3.1)
GWF_SIGNATURE = b"IGWD"

FRDATA_TYPES: tuple[str, ...] = (
    "ADC",
    "Proc",
    "Sim",
)


# -- i/o -----------------------------

def identify_gwf(
    origin: Literal["read", "write"],
    filepath: FileSystemPath | None,
    fileobj: FileLike | None,
    *args,
    **kwargs,
) -> bool:
    """Identify a filename or file object as GWF.

    This function is overloaded in that it will also identify a cache file
    as 'gwf' if the first entry in the cache contains a GWF file extension
    """
    # try and read file descriptor
    if fileobj is not None:
        loc = fileobj.tell()
        fileobj.seek(0)
        try:
            signature = fileobj.read(len(GWF_SIGNATURE))
        finally:
            fileobj.seek(loc)
        return signature == GWF_SIGNATURE

    # otherwise read file extension
    if filepath is not None:
        filepath = os.fspath(filepath)
        if filepath.endswith(".gwf"):
            return True
        if filepath.endswith((".lcf", ".cache")):
            try:
                cache = read_cache(filepath)
            except OSError:
                return False
            else:
                return os.fspath(cache[0]).endswith(".gwf")

    return False


# -- utilities -----------------------

def _from_backend(func, backend, *args, **kwargs):
    """Import a function for the given backend and execute it."""
    impl = get_backend_function(func, backend=backend)
    return impl(*args, **kwargs)


def _iter_toc(
    gwf: str | Path | IO,
    type: str | None = None,
    backend: str | None = None,
):
    """Import the ``_iter_toc`` implementation from the backend and run."""
    return _from_backend("_iter_toc", backend, gwf, type=type)


def _count_toc(
    gwf: str | Path | IO,
    type: str | None = None,
    backend: str | None = None,
):
    """Import the ``_count_toc`` implementation from the backend and run."""
    return _from_backend("_count_toc", backend, gwf, type=type)


def num_channels(
    gwf: str | Path | IO,
    type: str | None = None,
    backend: str | None = None,
) -> int:
    """Find the total number of channels in this GWF file.

    Requires a GWF backend library.

    Parameters
    ----------
    gwf : `str`
        GWF file to read.

    type : `str`
        The `Fr` structure type to match (one of 'adc', 'sim', or 'proc').
        Default is to match all types.

    backend : `str`, optional
        The GWF backend to use. Default is 'any'.

    Returns
    -------
    n : `int`
        The total number of channels found in the table of contents.
    """
    return _count_toc(gwf, type=type, backend=backend)


def get_channel_type(
    channel: str | Channel,
    gwf: str | Path | IO,
    backend: str | None = None,
) -> Literal["adc", "sim", "proc"]:
    """Find the channel type in a given GWF file.

    Requires a GWF backend library.

    Parameters
    ----------
    channel : `str`, `~gwpy.detector.Channel`
        Name of data channel to find.

    gwf : `str`
        GWF file to read.

    backend : `str`, optional
        The GWF backend to use. Default is 'any'.

    Returns
    -------
    ctype : `str`
        the type of the channel ('adc', 'sim', or 'proc')

    Raises
    ------
    ValueError
        if the channel is not found in the table-of-contents
    """
    channel = str(channel)
    for name, type_ in _iter_toc(gwf, backend=backend):
        if channel == name:
            return type_
    msg = f"'{channel}' not found in table-of-contents for {gwf}"
    raise ValueError(msg)


def channel_exists(
    gwf: str | Path | IO,
    channel: str | Channel,
    backend: str | None = None,
) -> bool:
    """Determine whether a channel exists in a GWF file.

    Requires a GWF backend library.

    Parameters
    ----------
    channel : `str`
        Name of data channel to find.

    gwf : `str`
        GWF file to read.

    backend : `str`, optional
        The GWF backend to use. Default is 'any'.

    Returns
    -------
    inframe : `bool`
        whether this channel is included in the table of contents for
        the given GWF file.
    """
    try:
        return _from_backend("_channel_exists", backend, gwf, channel)
    except NotImplementedError:
        # no backend-specific optimisation
        return str(channel) in iter_channel_names(gwf, backend=backend)


def iter_channel_names(
    gwf: str | Path | IO,
    type: str | None = None,
    backend: str | None = None,
) -> Iterator[str]:
    """Iterate over the names of channels found in a GWF file.

    Requires a GWF backend library.

    Parameters
    ----------
    gwf : `str`
        GWF file to read.

    type : `str`
        The `Fr` structure type to match (one of 'adc', 'sim', or 'proc').
        Default is to match all types.

    backend : `str`, optional
        The GWF backend to use. Default is 'any'.

    Returns
    -------
    channels : `generator`
        an iterator that will loop over the names of channels as read from
        the table of contents of the given GWF file
    """
    for name, _ in _iter_toc(gwf, type=type, backend=backend):
        yield name


def get_channel_names(
    gwf: str | Path | IO,
    type: str | None = None,
    backend: str | None = None,
) -> list[str]:
    """Return a list of all channel names found in a GWF file.

    This method just returns

    >>> list(iter_channel_names(gwf))

    Requires a GWF backend library.

    Parameters
    ----------
    gwf : `str`
        GWF file to read.

    type : `str`
        The `Fr` structure type to match (one of 'adc', 'sim', or 'proc').
        Default is to match all types.

    backend : `str`, optional
        The GWF backend to use. Default is 'any'.

    Returns
    -------
    channels : `list` of `str`
        a `list` of channel names as read from the table of contents of
        the given GWF file
    """
    return list(iter_channel_names(gwf, type=type, backend=backend))


def data_segments(
    paths: list[str],
    channel: str,
    warn: bool = True,
    backend: str | None = None,
) -> SegmentList:
    """Returns the segments containing data for a channel.

    Requires a GWF backend library.

    A frame is considered to contain data if a valid FrData structure
    (of any type) exists for the channel in that frame.  No checks
    are directly made against the underlying FrVect structures.

    Parameters
    ----------
    paths : `list` of `str`
        A list of GWF file paths.

    channel : `str`
        Name of data channel to find.

    warn : `bool`, optional
        If `True`, emit a `UserWarning` when a channel is not found
        in a frame, otherwise silently ignore.

    backend : `str`, optional
        The GWF backend to use. Default is 'any'.

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        the list of segments containing data
    """
    _channel_segments = get_backend_function(
        "_channel_segments",
        backend=backend,
    )
    segments = SegmentList()
    if isinstance(paths, str):
        paths = [paths]
    for path in paths:
        segments.extend(_channel_segments(path, channel, warn=warn))
    return segments.coalesce()


def _series_name(series: Series) -> str | None:
    """Returns the 'name' of a `Series` that should be written to GWF.

    This is basically `series.name or str(series.channel) or ""`.

    Parameters
    ----------
    series : `gwpy.types.Series`
        The input series that will be written.

    Returns
    -------
    name : `str`
        The name to use when storing this series.
    """
    return (
        series.name
        or str(series.channel or "")
        or None
    )
