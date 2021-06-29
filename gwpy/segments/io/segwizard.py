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

"""Read SegmentLists from seg-wizard format ASCII files
"""

import re

from .. import (Segment, SegmentList)
from ...io import registry
from ...io.utils import (
    identify_factory,
    with_open,
)
from ...time import LIGOTimeGPS

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

_FLOAT_PAT = r'([\d.+-eE]+)'
# simple two-column (gpsstart, gpsend)
TWO_COL_REGEX = re.compile(
    r'\A\s*{float}\s+{float}\s*\Z'.format(float=_FLOAT_PAT))
# three column (gpsstart, gpsend, duration)
THREE_COL_REGEX = re.compile(
    r'\A\s*{float}\s+{float}\s+{float}\s*\Z'.format(float=_FLOAT_PAT))
# four column (index, gpsstart, gpsend, duration)
FOUR_COL_REGEX = re.compile(
    r'\A\s*([\d]+)\s+{float}\s+{float}\s+{float}\s*\Z'.format(
        float=_FLOAT_PAT))


# -- read ---------------------------------------------------------------------

@with_open
def from_segwizard(source, gpstype=LIGOTimeGPS, strict=True):
    """Read segments from a segwizard format file into a `SegmentList`

    Parameters
    ----------
    source : `file`, `str`
        An open file, or file path, from which to read

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
        if line.startswith(('#', ';')):  # comment
            continue
        # determine line format
        if fmt_pat is None:
            fmt_pat = _line_format(line)
        # parse line
        tokens, = fmt_pat.findall(line)
        out.append(_format_segment(tokens[-3:], gpstype=gpstype,
                                   strict=strict))
    return out


def _line_format(line):
    """Determine the column format pattern for a line in an ASCII segment file.
    """
    for pat in (FOUR_COL_REGEX, THREE_COL_REGEX, TWO_COL_REGEX):
        if pat.match(line):
            return pat
    raise ValueError("unable to parse segment from line {!r}".format(line))


def _format_segment(tokens, strict=True, gpstype=LIGOTimeGPS):
    """Format a list of tokens parsed from an ASCII file into a segment.
    """
    try:
        start, end, dur = tokens
    except ValueError:  # two-columns
        return Segment(*map(gpstype, tokens))
    seg = Segment(gpstype(start), gpstype(end))
    if strict and not float(abs(seg)) == float(dur):
        raise ValueError(
            "segment {0!r} has incorrect duration {1!r}".format(seg, dur),
        )
    return seg


# -- write --------------------------------------------------------------------

@with_open(mode="w", pos=1)
def to_segwizard(segs, target, header=True, coltype=LIGOTimeGPS):
    """Write the given `SegmentList` to a file in SegWizard format.

    Parameters
    ----------
    segs : :class:`~gwpy.segments.SegmentList`
        The list of segments to write.

    target : `file`, `str`
        An open file, or file path, to which to write.

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
        print('# seg\tstart\tstop\tduration', file=target)
    for i, seg in enumerate(segs):
        a = coltype(seg[0])
        b = coltype(seg[1])
        c = float(b - a)
        print(
            '\t'.join(map(str, (i, a, b, c))),
            file=target,
        )


# -- register -----------------------------------------------------------------

registry.register_reader('segwizard', SegmentList, from_segwizard)
registry.register_writer('segwizard', SegmentList, to_segwizard)
registry.register_identifier('segwizard', SegmentList,
                             identify_factory('txt', 'dat'))
