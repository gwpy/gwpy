# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

import warnings

from six import string_types

from .. import (Segment, SegmentList, DataQualityFlag)
from ...io import registry
from ...io.utils import identify_factory
from ...io.cache import file_list
from ...time import LIGOTimeGPS

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ---------------------------------------------------------------------

def from_segwizard(source, coalesce=True, gpstype=LIGOTimeGPS, strict=True,
                   nproc=1):
    """Read segments from a segwizard format file into a `SegmentList`
    """
    from glue import segmentsUtils

    if nproc != 1:
        return SegmentList.read(source, coalesce=coalesce, gpstype=gpstype,
                                strict=strict, nproc=nproc, format='cache')

    # format list of files and read in serial
    files = file_list(source)
    segs = SegmentList()
    for file_ in files:
        with open(file_, 'r') as fobj:
            raw = segmentsUtils.fromsegwizard(fobj, coltype=gpstype,
                                              strict=strict)
            segs.extend(SegmentList(map(Segment, raw)))
    if coalesce:
        segs.coalesce()
    return segs


# -- write --------------------------------------------------------------------

def to_segwizard(segs, fobj, header=True, coltype=int):
    """Write the given `SegmentList` to the file object fobj

    Parameters
    ----------
    segs : :class:`~gwpy.segments.segments.SegmentList`
        segmentlist to print
    fobj : `file`, `str`
        open file object, or file path, to write to
    header : `bool`, optional
        print header into the file, default: `True`
    coltype : `type`, optional
        numerical type in which to cast times before printing

    See Also
    --------
    glue.segmentsUtils
        for definition of the segwizard format, and the to/from functions
        used in this GWpy module
    """
    from glue import segmentsUtils

    if isinstance(fobj, string_types):
        close = True
        fobj = open(fobj, 'w')
    else:
        close = False
    segmentsUtils.tosegwizard(fobj, segs, header=header, coltype=coltype)
    if close:
        fobj.close()

# -- register -----------------------------------------------------------------

registry.register_reader('segwizard', SegmentList, from_segwizard)
registry.register_writer('segwizard', SegmentList, to_segwizard)
registry.register_identifier('segwizard', SegmentList,
                             identify_factory('txt', 'dat'))
