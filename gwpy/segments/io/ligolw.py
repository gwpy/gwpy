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

"""Read/write segment XML in LIGO_LW format into DataQualityFlags
"""

import operator

from six.moves import reduce

from astropy.io import registry as io_registry

from ...time import LIGOTimeGPS
from ...io.ligolw import (is_xml, build_content_handler, read_ligolw,
                          write_tables, patch_ligotimegps)
from ...segments import (DataQualityFlag, DataQualityDict)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def segment_content_handler():
    """Build a `~xml.sax.handlers.ContentHandler` to read segment XML tables
    """
    from glue.ligolw.lsctables import (SegmentTable, SegmentDefTable,
                                       SegmentSumTable)
    from glue.ligolw.ligolw import PartialLIGOLWContentHandler

    def _filter(name, attrs):
        return reduce(
            operator.or_,
            [table_.CheckProperties(name, attrs) for
             table_ in (SegmentTable, SegmentDefTable, SegmentSumTable)])

    return build_content_handler(PartialLIGOLWContentHandler, _filter)


# -- read ---------------------------------------------------------------------

def read_ligolw_dict(source, flags=None, gpstype=LIGOTimeGPS, coalesce=False):
    """Read segments for the given flag from the LIGO_LW XML file.

    Parameters
    ----------
    source : `file`, `str`, :class:`~glue.ligolw.ligolw.Document`, `list`
        one (or more) open files or file paths, or LIGO_LW `Document` objects

    flags : `list`, `None`, optional
        list of flags to read or `None` to read all into a single
        `DataQualityFlag`.

    gpstype : `type`, `callable`, optional
        class to use for GPS times in returned objects, can be a function
        to convert GPS time to something else, default is
        `~gwpy.time.LIGOTimeGPS`

    coalesce : `bool`, optional
        if `True`, coalesce all parsed `DataQualityFlag` objects before
        returning, default: `False`

    Returns
    -------
    flagdict : `DataQualityDict`
        a new `DataQualityDict` of `DataQualityFlag` entries with ``active``
        and ``known`` segments seeded from the XML tables in the given
        file ``fp``.
    """
    from glue.ligolw.lsctables import (SegmentTable, SegmentDefTable,
                                       SegmentSumTable)

    # read file(s)
    xmldoc = read_ligolw(source, contenthandler=segment_content_handler())

    # extract tables
    tables = [table_.get_table(xmldoc) for
              table_ in (SegmentDefTable, SegmentSumTable, SegmentTable)]

    # parse tables
    with patch_ligotimegps():
        out = DataQualityDict.from_ligolw_tables(*tables, names=flags,
                                                 gpstype=gpstype)

    # coalesce
    if coalesce:
        for flag in out:
            out[flag].coalesce()

    return out


def read_ligolw_flag(source, flag=None, **kwargs):
    """Read a single `DataQualityFlag` from a LIGO_LW XML file
    """
    return read_ligolw_dict(source, flags=flag, **kwargs).values()[0]


# -- write --------------------------------------------------------------------

def write_ligolw(flags, target, **kwargs):
    """Write this `DataQualityFlag` to the given LIGO_LW Document

    Parameters
    ----------
    flags : `DataQualityFlag`, `DataQualityDict`
        `gwpy.segments` object to write

    target : `str`, `file`, :class:`~glue.ligolw.ligolw.Document`
        the file or document to write into

    **kwargs
        keyword arguments to use when writing

    See also
    --------
    gwpy.io.ligolw.write_ligolw_tables
        for details of acceptable keyword arguments
    """
    if isinstance(flags, DataQualityFlag):
        flags = DataQualityDict({flags.name: flags})

    return write_tables(target, flags.to_ligolw_tables(), **kwargs)


# -- register -----------------------------------------------------------------

# register methods for DataQualityDict
io_registry.register_reader('ligolw', DataQualityFlag, read_ligolw_flag)
io_registry.register_writer('ligolw', DataQualityFlag, write_ligolw)
io_registry.register_identifier('ligolw', DataQualityFlag, is_xml)

# register methods for DataQualityDict
io_registry.register_reader('ligolw', DataQualityDict, read_ligolw_dict)
io_registry.register_writer('ligolw', DataQualityDict, write_ligolw)
io_registry.register_identifier('ligolw', DataQualityDict, is_xml)
