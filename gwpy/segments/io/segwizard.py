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

"""Read SegmentLists from seg-wizard format ASCII files."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from ...io.registry import default_registry
from ...io.utils import with_open
from ...time import LIGOTimeGPS
from .. import (
    Segment,
    SegmentList,
)

if TYPE_CHECKING:
    from typing import (
        IO,
        TextIO,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

_FLOAT_PAT = r"([\d.+-eE]+)"
# simple two-column (gpsstart, gpsend)
TWO_COL_REGEX = re.compile(
    rf"\A\s*{_FLOAT_PAT}\s+{_FLOAT_PAT}\s*\Z",
)
# three column (gpsstart, gpsend, duration)
THREE_COL_REGEX = re.compile(
    rf"\A\s*{_FLOAT_PAT}\s+{_FLOAT_PAT}\s+{_FLOAT_PAT}\s*\Z",
)
# four column (index, gpsstart, gpsend, duration)
FOUR_COL_REGEX = re.compile(
    rf"\A\s*([\d]+)\s+{_FLOAT_PAT}\s+{_FLOAT_PAT}\s+{_FLOAT_PAT}\s*\Z",
)


# -- identify ------------------------

def _is_segwizard(
    origin: str,
    filepath: str,
    fileobj: IO,
    *args,
    **kwargs,
):
    """Return `True` if the given file looks like a segwizard file.

    When introspecting a file contents, this method only recognises the
    four-column seg/start/stop/duration format written by
    `igwn_segments.utils.tosegwizard`.
    """
    if fileobj is not None:
        pos = fileobj.seek(0)
        try:
            header = fileobj.readline()
        finally:
            fileobj.seek(pos)
        cols = header.strip().split()
        return cols == [b"#", b"seg", b"start", b"stop", b"duration"]
    if filepath is not None:
        return filepath.endswith((".dat", ".txt"))
    return False


# -- read ----------------------------

@with_open
def from_segwizard(
    source: IO,
    gpstype: type = LIGOTimeGPS,
    strict: bool = True,
) -> SegmentList:
    """Read segments from a segwizard format file into a `SegmentList`.

    Parameters
    ----------
    source : `file`, `str`, `pathlib.Path`
        An open file, or file path, from which to read.

    gpstype : `type`, optional
        The numeric type to which to cast times (from `str`) when reading.

    strict : `bool`, optional
        Check that recorded duration matches ``end-start`` for all segments;
        only used when reading from a 3+-column file.

    Returns
    -------
    segments : `~gwpy.segments.SegmentList`
        The list of segments as parsed from the file.

    Notes
    -----
    This method is adapted from original code written by Kipp Cannon and
    distributed under GPLv3.
    """
    # read file object
    out = SegmentList()
    fmt_pat = None
    for line in source:
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if line.startswith(("#", ";")):  # comment
            continue
        # determine line format
        if fmt_pat is None:
            fmt_pat = _line_format(line)
        # parse line
        tokens, = fmt_pat.findall(line)
        out.append(_format_segment(
            tokens[-3:],
            gpstype=gpstype,
            strict=strict,
        ))
    return out


def _line_format(
    line: str,
) -> re.Pattern:
    """Determine the column format pattern for a line in an ASCII segment file."""
    for pat in (
        FOUR_COL_REGEX,
        THREE_COL_REGEX,
        TWO_COL_REGEX,
    ):
        if pat.match(line):
            return pat
    msg = f"unable to parse segment from line '{line}'"
    raise ValueError(msg)


def _format_segment(
    tokens: list[str],
    strict: bool = True,
    gpstype: type = LIGOTimeGPS,
) -> Segment:
    """Format a list of tokens parsed from an ASCII file into a segment."""
    try:
        start, end, dur = tokens
    except ValueError:  # two-columns
        return Segment(*map(gpstype, tokens))
    seg = Segment(gpstype(start), gpstype(end))
    if strict and abs(seg) != float(dur):
        msg = f"segment {seg} has incorrect duration {dur}"
        raise ValueError(msg)
    return seg


# -- write ---------------------------

@with_open(mode="w", pos=1)
def to_segwizard(
    segs: SegmentList,
    target: TextIO,
    header: bool = True,
    coltype: type = LIGOTimeGPS,
):
    """Write the given `SegmentList` to a file in SegWizard format.

    Parameters
    ----------
    segs : :class:`~gwpy.segments.SegmentList`
        The list of segments to write.

    target : `io.TextIOBase`
        A file open for writing in text mode.

    header : `bool`, optional
        Print a column header into the file, default: `True`.

    coltype : `type`, optional
        The numerical type in which to cast times before printing.

    Notes
    -----
    This method is adapted from original code written by Kipp Cannon and
    distributed under GPLv3.
    """
    # write file object
    if header:
        print("# seg\tstart\tstop\tduration", file=target)
    for i, seg in enumerate(segs):
        a = coltype(seg[0])
        b = coltype(seg[1])
        c = float(b - a)
        print(
            "\t".join(map(str, (i, a, b, c))),
            file=target,
        )


# -- register ------------------------

default_registry.register_reader("segwizard", SegmentList, from_segwizard)
default_registry.register_writer("segwizard", SegmentList, to_segwizard)
default_registry.register_identifier(
    "segwizard",
    SegmentList,
    _is_segwizard,
)
