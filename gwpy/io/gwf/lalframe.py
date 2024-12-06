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

import warnings
from collections.abc import Iterator

import lalframe

from ...segments import Segment
from ...time import LIGOTimeGPS


def open_gwf(
    path: str,
    mode: str = "r",
) -> lalframe.FrameUFrFile:
    """Open a GWF file using LALFrame.

    Parameters
    ----------
    path : `str`
        The path to read.

    mode : `str`
        The mode with which to open the file, either `r` or `w`.

    Returns
    -------
    frfile : `lalframe.FrameUFrFile`
        The opened file object.
    """
    if isinstance(path, lalframe.FrameUFrFile):
        return path
    try:
        return lalframe.FrameUFrFileOpen(path, mode)
    except RuntimeError as exc:
        if str(exc) == "I/O error":
            raise OSError(f"failed to open '{path}'") from exc
        raise


def _iter_channels(path: str) -> Iterator[tuple[str, str]]:
    """Yield the names and types of channels listed in the TOC for a GWF file.

    Parameters
    ----------
    path : `str`
        The path to read.

    Yields
    ------
    name : `str`
        The name of a channel
    type : `str`
        The ``FrVect`` type of the channel, one of ``"sim"``, ``"proc"``,
        or ``"adc"``.
    """
    frfile = open_gwf(path)
    frtoc = lalframe.FrameUFrTOCRead(frfile)
    for typename in ("Sim", "Proc", "Adc"):
        typen = typename.lower()
        count = getattr(lalframe, f"FrameUFrTOCQuery{typename}N")(frtoc)
        get = getattr(lalframe, f"FrameUFrTOCQuery{typename}Name")
        for i in range(count):
            name = get(frtoc, i)
            yield name, typen


def _channel_segments(
    path: str,
    name: str,
    warn: bool = True,
) -> Iterator[Segment]:
    """Yields the segments containing data for ``name`` in this GWF path.

    Parameters
    ----------
    path : `str`
        The path to read.

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
    frfile = open_gwf(path)
    frtoc = lalframe.FrameUFrTOCRead(frfile)

    # get segments for frames
    nframes = lalframe.FrameUFrTOCQueryNFrame(frtoc)
    for i in range(nframes):
        try:
            chan = lalframe.FrameUFrChanRead(frfile, name, i)
        except RuntimeError as exc:
            if str(exc) == "Wrong name" and warn:
                warnings.warn(
                    f"'{name}' not found in frame {i} of {path}",
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
