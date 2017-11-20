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


# DEPRECATED - remove prior to 1.0 release
def flag_from_segwizard(filename, flag=None, coalesce=True, gpstype=float,
                        strict=True, nproc=1):
    # pylint: disable=missing-docstring,too-many-arguments
    warnings.warn("Reading DataQualityFlags from ASCII files has been "
                  "deprecated, and will be removed prior to the 1.0 "
                  "release of GWpy. Please move to using a structured "
                  "file-format, e.g. HDF5 or JSON", DeprecationWarning)
    if isinstance(flag, DataQualityFlag):
        out = flag
    else:
        out = DataQualityFlag(str(flag))
    out.active = from_segwizard(filename, coalesce=coalesce, gpstype=gpstype,
                                strict=strict, nproc=nproc)
    out.known = out.active
    return out


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


# DEPRECATED - remove prior to 1.0 release
def flag_to_segwizard(flag, fobj, header=True, coltype=int):
    """Write the given `DataQualityFlag` to the file object fobj

    Parameters
    ----------
    flag : :class:`~gwpy.segments.flag.DataQualityFlag`
        data quality flag to print
    fobj : `file`, `str`
        open file object, or file path, to write to
    header : `bool`, optional
        print header into the file, default: `True`
    coltype : `type`, optional
        numerical type in which to cast times before printing

    Raises
    ------
    ValueError
        if the `~DataQualityFlag.known` segments for `flag` do not mirror
        the `~DataQualityFlag.active` segments. In other words, if recording
        only the `active` segments would discard information.

    Notes
    -----
    In this format, only the
    :attr:`~gwpy.segments.flag.DataQualityFlag.active` segments are
    printed

    See Also
    --------
    glue.segmentsUtils
        for definition of the segwizard format, and the to/from functions
        used in this GWpy module
    """
    warnings.warn("Writing DataQualityFlags to ASCII files has been "
                  "deprecated, and will be removed prior to the 1.0 "
                  "release of GWpy. Please move to using a structured "
                  "file-format, e.g. HDF5 or JSON", DeprecationWarning)
    if flag.known and flag.known != flag.active:
        raise ValueError("This DataQualityFlag has known segments that do not "
                         "simply match the active ones, meaning the SegWizard "
                         "format cannot preserve these data completely. "
                         "Consider using HDF5 or LIGO_LW XML, otherwise call "
                         "the write() method of the active SegmentList "
                         "directly to write just those segments.")
    to_segwizard(flag.active, fobj, header=header, coltype=coltype)


# -- identify -----------------------------------------------------------------

identify_segwizard = identify_factory('txt', 'dat')  # pylint: disable=invalid-name

# -- register -----------------------------------------------------------------

registry.register_reader('segwizard', SegmentList, from_segwizard)
registry.register_writer('segwizard', SegmentList, to_segwizard)
registry.register_identifier('segwizard', SegmentList, identify_segwizard)

# DEPRECATED - remove prior to 1.0 release
registry.register_reader('segwizard', DataQualityFlag, flag_from_segwizard)
registry.register_writer('segwizard', DataQualityFlag, flag_to_segwizard)
registry.register_identifier('segwizard', DataQualityFlag, identify_segwizard)
