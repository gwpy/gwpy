# Copyright (C) Cardiff University (2024-)
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

"""GWF I/O utilities using LALFrame.
"""

from __future__ import annotations

import typing
import warnings
from collections.abc import Iterator

import lalframe

from ...segments import Segment
from ...time import LIGOTimeGPS
from ..utils import file_path
from .core import FRDATA_TYPES

if typing.TYPE_CHECKING:
    from pathlib import Path
    from typing import (
        IO,
        Literal,
    )


def open_gwf(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    mode: str = "r",
) -> lalframe.FrameUFrFile:
    """Open a GWF file using LALFrame.

    Parameters
    ----------
    gwf : `str`, `pathlib.Path`, `file`, or open frameCPP stream
        The path to read from, or write to. Already open GWF streams
        are returned unmodified, if the type matches the mode.

    mode : `str`, optional
        The mode with which to open the file, either `r` or `w`.

    Returns
    -------
    frfile : `lalframe.FrameUFrFile`
        The opened file object.
    """
    if isinstance(gwf, lalframe.FrameUFrFile):
        return gwf

    path = file_path(gwf)
    try:
        return lalframe.FrameUFrFileOpen(path, mode)
    except RuntimeError as exc:
        if str(exc) == "I/O error":
            raise OSError(f"failed to open '{path}'") from exc
        raise


@typing.overload
def _iter_toc(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    type: str | None,
    count: Literal[True],
) -> Iterator[int]:
    ...


@typing.overload
def _iter_toc(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    type: str | None,
    count: Literal[False],
) -> Iterator[tuple[str, str]]:
    ...


def _iter_toc(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    type: str | None = None,
    count: bool = False,
) -> Iterator[tuple[str, str] | int]:
    """Yield the names and types of channels listed in the TOC for a GWF file.

    Parameters
    ----------
    gwf : `str`, `pathlib.Path`, `file`, `lalframe.FrameUFrFile`
        Path of GWF file, or open file stream, to read.

    Yields
    ------
    name : `str`
        The name of a channel
    type : `str`
        The ``FrVect`` type of the channel, one of ``"sim"``, ``"proc"``,
        or ``"adc"``.
    """
    frfile = open_gwf(gwf)
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    if type is None:
        types = FRDATA_TYPES
    else:
        types = (type,)
    for typename in map(str.lower, types):
        typet = typename.title()
        num = getattr(lalframe, f"FrameUFrTOCQuery{typet}N")(frtoc)
        if count:
            yield num
            continue
        get = getattr(lalframe, f"FrameUFrTOCQuery{typet}Name")
        for i in range(num):
            name = get(frtoc, i)
            yield name, typename


def _count_toc(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    type: str | None = None,
) -> int:
    """Yield the names and types of channels listed in the TOC for a GWF file.

    Parameters
    ----------
    gwf : `str`, `pathlib.Path`, `file`, `lalframe.FrameUFrFile`
        Path of GWF file, or open file stream, to read.

    Yields
    ------
    name : `str`
        The name of a channel
    type : `str`
        The ``FrVect`` type of the channel, one of ``"sim"``, ``"proc"``,
        or ``"adc"``.
    """
    return sum(_iter_toc(gwf, type=type, count=True))


def _channel_exists(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    name: str,
) -> bool:
    """Return `True` if a channel (name) exists in a file.
    """
    frfile = open_gwf(gwf)
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    nframes = lalframe.FrameUFrTOCQueryNFrame(frtoc)
    for i in range(nframes):
        try:
            lalframe.FrameUFrChanRead(frfile, name, i)
        except RuntimeError as exc:
            if str(exc) == "Wrong name":
                continue
            raise
        else:
            return True
    return False


def _channel_segments(
    gwf: str | Path | IO | lalframe.FrameUFrFile,
    name: str,
    warn: bool = True,
) -> Iterator[Segment]:
    """Yields the segments containing data for ``name`` in this GWF path.

    Parameters
    ----------
    gwf : `str`, `pathlib.Path`, `file`, `lalframe.FrameUFrFile`
        Path of GWF file, or open file stream, to read.

    name : `str`
        The name of the channel to read.

    warn : `bool`
        If `True` emit a `UserWarning` when ``name`` is not found in
        a frame in the file.

    Yields
    ------
    gwpy.segments.Segment
        A GPS time segment for which data are available in the file.
    """
    frfile = open_gwf(gwf)
    frtoc = lalframe.FrameUFrTOCRead(frfile)

    # get segments for frames
    nframes = lalframe.FrameUFrTOCQueryNFrame(frtoc)
    for i in range(nframes):
        try:
            chan = lalframe.FrameUFrChanRead(frfile, name, i)
        except RuntimeError as exc:
            if str(exc) == "Wrong name" and warn:
                warnings.warn(
                    f"'{name}' not found in frame {i} of {gwf}",
                )
                continue
            raise
        gps = sum(map(
            LIGOTimeGPS,
            lalframe.FrameUFrTOCQueryGTimeModf(frtoc, i),
        ))
        dur = lalframe.FrameUFrTOCQueryDt(frtoc, i)
        offset = lalframe.FrameUFrChanQueryTimeOffset(chan)
        yield Segment(gps + offset, gps + dur + offset)
