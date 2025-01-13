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

"""Read/write segment XML in LIGO_LW format into DataQualityFlags.
"""

from __future__ import annotations

import operator
import typing
from functools import reduce

from ...io.ligolw import (
    build_content_handler,
    is_ligolw,
    patch_ligotimegps,
    read_ligolw,
    read_table,
    write_tables,
)
from ...io.registry import (
    register_identifier,
    register_reader,
    register_writer,
)
from ...segments import (
    DataQualityDict,
    DataQualityFlag,
)

if typing.TYPE_CHECKING:
    from typing import (
        IO,
        TypeAlias,
    )

    from ligo.lw.ligolw import (
        Document,
        PartialContentHandler,
    )

    LigolwInput: TypeAlias = IO | str | Document | list[str]

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def segment_content_handler() -> PartialContentHandler:
    """Build a `~xml.sax.handlers.ContentHandler` to read segment XML tables
    """
    from ligo.lw.ligolw import PartialLIGOLWContentHandler
    from ligo.lw.lsctables import (
        SegmentDefTable,
        SegmentSumTable,
        SegmentTable,
    )

    def _filter(name, attrs):
        return reduce(
            operator.or_,
            [table_.CheckProperties(name, attrs) for
             table_ in (SegmentTable, SegmentDefTable, SegmentSumTable)])

    return build_content_handler(PartialLIGOLWContentHandler, _filter)


# -- read ---------------------------------------------------------------------

def read_ligolw_dict(
    source: LigolwInput,
    names: list[str] | None = None,
    coalesce: bool = False,
    **kwargs,
) -> DataQualityDict:
    """Read segments for the given flag from the LIGO_LW XML file.

    Parameters
    ----------
    source : `file`, `str`, :class:`~ligo.lw.ligolw.Document`, `list`
        One (or more) open files or file paths, or LIGO_LW `Document` objects.

    names : `list`, `None`, optional
        List of names to read or `None` to read all into a single
        `DataQualityFlag`.

    coalesce : `bool`, optional
        If `True`, coalesce all parsed `DataQualityFlag` objects before
        returning, default: `False`

    kwargs
        Other keywords are passed to `DataQualityDict.from_ligolw_tables`.

    Returns
    -------
    flagdict : `DataQualityDict`
        A new `DataQualityDict` of `DataQualityFlag` entries with ``active``
        and ``known`` segments seeded from the XML tables in the given
        file ``fp``.
    """
    xmldoc = read_ligolw(source, contenthandler=segment_content_handler())
    segdef = read_table(xmldoc, "segment_definer")
    segsum = read_table(xmldoc, "segment_summary")
    seg = read_table(xmldoc, "segment")

    # parse tables
    with patch_ligotimegps(type(xmldoc.childNodes[0]).__module__):
        out = DataQualityDict.from_ligolw_tables(
            segdef,
            segsum,
            seg,
            names=names,
            **kwargs,
        )

    # coalesce
    if coalesce:
        for flag in out:
            out[flag].coalesce()

    return out


def read_ligolw_flag(
    source: LigolwInput,
    name: str | None = None,
    **kwargs,
) -> DataQualityFlag:
    """Read a single `DataQualityFlag` from a LIGO_LW XML file
    """
    return list(read_ligolw_dict(
        source,
        names=[name] if name is not None else None,
        **kwargs,
    ).values())[0]


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
register_reader("ligolw", DataQualityFlag, read_ligolw_flag)
register_writer("ligolw", DataQualityFlag, write_ligolw)
register_identifier("ligolw", DataQualityFlag, is_ligolw)

# register methods for DataQualityDict
register_reader("ligolw", DataQualityDict, read_ligolw_dict)
register_writer("ligolw", DataQualityDict, write_ligolw)
register_identifier("ligolw", DataQualityDict, is_ligolw)
