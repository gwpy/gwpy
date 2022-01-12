# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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
from functools import reduce

from astropy.io import registry as io_registry

from ...io.ligolw import (is_ligolw, build_content_handler, read_ligolw,
                          write_tables, patch_ligotimegps)
from ...segments import (DataQualityFlag, DataQualityDict)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def segment_content_handler():
    """Build a `~xml.sax.handlers.ContentHandler` to read segment XML tables
    """
    from ligo.lw.lsctables import (SegmentTable, SegmentDefTable,
                                   SegmentSumTable)
    from ligo.lw.ligolw import PartialLIGOLWContentHandler

    def _filter(name, attrs):
        return reduce(
            operator.or_,
            [table_.CheckProperties(name, attrs) for
             table_ in (SegmentTable, SegmentDefTable, SegmentSumTable)])

    return build_content_handler(PartialLIGOLWContentHandler, _filter)


# -- read ---------------------------------------------------------------------

def read_ligolw_dict(source, names=None, coalesce=False, **kwargs):
    """Read segments for the given flag from the LIGO_LW XML file.

    Parameters
    ----------
    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        one (or more) open files or file paths, or LIGO_LW `Document` objects

    names : `list`, `None`, optional
        list of names to read or `None` to read all into a single
        `DataQualityFlag`.

    coalesce : `bool`, optional
        if `True`, coalesce all parsed `DataQualityFlag` objects before
        returning, default: `False`

    **kwargs
        other keywords are passed to :meth:`DataQualityDict.from_ligolw_tables`

    Returns
    -------
    flagdict : `DataQualityDict`
        a new `DataQualityDict` of `DataQualityFlag` entries with ``active``
        and ``known`` segments seeded from the XML tables in the given
        file ``fp``.
    """
    xmldoc = read_ligolw(source, contenthandler=segment_content_handler())

    # parse tables
    with patch_ligotimegps(type(xmldoc.childNodes[0]).__module__):
        out = DataQualityDict.from_ligolw_tables(
            *xmldoc.childNodes,
            names=names,
            **kwargs
        )

    # coalesce
    if coalesce:
        for flag in out:
            out[flag].coalesce()

    return out


def read_ligolw_flag(source, name=None, **kwargs):
    """Read a single `DataQualityFlag` from a LIGO_LW XML file
    """
    name = [name] if name is not None else None
    return list(read_ligolw_dict(source, names=name, **kwargs).values())[0]


# -- write --------------------------------------------------------------------

def write_ligolw(flags, target, attrs=None, **kwargs):
    """Write this `DataQualityFlag` to the given LIGO_LW Document

    Parameters
    ----------
    flags : `DataQualityFlag`, `DataQualityDict`
        `gwpy.segments` object to write

    target : `str`, `file`, :class:`~ligo.lw.ligolw.Document`
        the file or document to write into

    attrs : `dict`, optional
        extra attributes to write into segment tables

    **kwargs
        keyword arguments to use when writing

    See also
    --------
    gwpy.io.ligolw.write_ligolw_tables
        for details of acceptable keyword arguments
    """
    if isinstance(flags, DataQualityFlag):
        flags = DataQualityDict({flags.name: flags})
    return write_tables(
        target,
        flags.to_ligolw_tables(**attrs or dict()),
        **kwargs
    )


# -- register -----------------------------------------------------------------

# register methods for DataQualityDict
io_registry.register_reader('ligolw', DataQualityFlag, read_ligolw_flag)
io_registry.register_writer('ligolw', DataQualityFlag, write_ligolw)
io_registry.register_identifier('ligolw', DataQualityFlag, is_ligolw)

# register methods for DataQualityDict
io_registry.register_reader('ligolw', DataQualityDict, read_ligolw_dict)
io_registry.register_writer('ligolw', DataQualityDict, write_ligolw)
io_registry.register_identifier('ligolw', DataQualityDict, is_ligolw)
