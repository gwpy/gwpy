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

import datetime

from six import string_types

from ...time import (LIGOTimeGPS, to_gps)
from ...io import registry
from ...io.ligolw import (identify_ligolw, write_tables)
from ...io.cache import (file_list, FILE_LIKE)
from ...segments import (Segment, DataQualityFlag, DataQualityDict)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ---------------------------------------------------------------------

def read_ligolw_dict(f, flags=None, gpstype=LIGOTimeGPS, coalesce=False,
                     contenthandler=None, nproc=1):
    """Read segments for the given flag from the LIGO_LW XML file.

    Parameters
    ----------
    fp : `str`
        path of XML file to read.
    flags : `list`, `None`, optional
        list of flags to read or `None` to read all into a single
        `DataQualityFlag`.

    Returns
    -------
    flagdict : :class:`~gwpy.segments.flag.DataQualityDict`
        a new `DataQualityDict` of `DataQualityFlag` entries with ``active``
        and ``known`` segments seeded from the XML tables in the given
        file ``fp``.
    """
    if nproc != 1:
        return DataQualityDict.read(f, flags, coalesce=coalesce,
                                    gpstype=gpstype,
                                    contenthandler=contenthandler,
                                    format='cache', nproc=nproc)

    from glue.ligolw import lsctables
    from glue.ligolw.ligolw import (Document, LIGOLWContentHandler)
    from glue.ligolw.utils.ligolw_add import ligolw_add

    if contenthandler is None:
        contenthandler = LIGOLWContentHandler

    lsctables.use_in(contenthandler)

    # generate Document and populate
    xmldoc = Document()
    files = [fp.name if isinstance(fp, FILE_LIKE) else fp
             for fp in file_list(f)]
    ligolw_add(xmldoc, files, non_lsc_tables_ok=True,
               contenthandler=contenthandler)

    # read segment definers and generate DataQualityFlag object
    seg_def_table = lsctables.SegmentDefTable.get_table(xmldoc)

    # find flags
    if isinstance(flags, string_types):
        flags = flags.split(',')
    out = DataQualityDict()
    id_ = dict()
    if flags is not None and len(flags) == 1 and flags[0] is None:
        out[None] = DataQualityFlag()
        id_[None] = []
    for row in seg_def_table:
        ifos = row.get_ifos()
        name = row.name
        if ifos and name:
            name = ':'.join([''.join(row.get_ifos()), row.name])
            if row.version is not None:
                name += ':%d' % row.version
        else:
            name = None
        if flags is None or name in flags:
            out[name] = DataQualityFlag(name)
            try:
                id_[name].append(row.segment_def_id)
            except (AttributeError, KeyError):
                id_[name] = [row.segment_def_id]
    if flags is None and not len(out.keys()):
        raise RuntimeError("No segment definitions found in file.")
    elif flags is not None and len(out.keys()) != len(flags):
        for flag in flags:
            if flag not in out:
                raise ValueError("No segment definition found for flag=%r "
                                 "in file." % flag)
    # read segment summary table as 'known'
    seg_sum_table = lsctables.SegmentSumTable.get_table(xmldoc)
    for row in seg_sum_table:
        for flag in out:
            if not id_[flag] or row.segment_def_id in id_[flag]:
                try:
                    s = row.get()
                except AttributeError:
                    s = row.start_time, row.end_time
                out[flag].known.append(Segment(gpstype(s[0]), gpstype(s[1])))
    for dqf in out:
        if coalesce:
            out[dqf].coalesce()
    # read segment table as 'active'
    seg_table = lsctables.SegmentTable.get_table(xmldoc)
    for row in seg_table:
        for flag in out:
            if not id_[flag] or row.segment_def_id in id_[flag]:
                try:
                    s = row.get()
                except AttributeError:
                    s = row.start_time, row.end_time
                out[flag].active.append(Segment(gpstype(s[0]), gpstype(s[1])))
    for dqf in out:
        if coalesce:
            out[dqf].coalesce()
    return out


def read_ligolw_flag(fp, flag=None, **kwargs):
    """Read a single `DataQualityFlag` from a LIGO_LW XML file
    """
    return read_ligolw_dict(fp, flags=flag, **kwargs).values()[0]


# -- write --------------------------------------------------------------------

def dqdict_to_ligolw_tables(dqdict):
    """Convert a `DataQualityDict` into LIGO_LW segment tables

    Parameters
    ----------
    dqdict : `~gwpy.segments.DataQualityDict`
        the dict of flags to write

    Returns
    -------
    segdeftab : :class:`~glue.ligolw.lsctables.SegmentDefTable`
        the ``segment_definer`` table defining each flag
    segsumtab : :class:`~glue.ligolw.lsctables.SegmentSumTable`
        the ``segment_summary`` table containing the known segments
    segtab : :class:`~glue.ligolw.lsctables.SegmentTable`
        the ``segment`` table containing the active segments
    """
    from glue.ligolw import lsctables

    segdeftab = lsctables.New(lsctables.SegmentDefTable)
    segsumtab = lsctables.New(lsctables.SegmentSumTable)
    segtab = lsctables.New(lsctables.SegmentTable)

    # write flags to tables
    for flag in dqdict.values():
        # segment definer
        segdef = segdeftab.RowType()
        for col in segdeftab.columnnames:  # default all columns to None
            setattr(segdef, col, None)
        segdef.set_ifos([flag.ifo])
        segdef.name = flag.tag
        segdef.version = flag.version
        segdef.comment = flag.description
        segdef.insertion_time = to_gps(datetime.datetime.now()).gpsSeconds
        segdef.segment_def_id = lsctables.SegmentDefTable.get_next_id()
        segdeftab.append(segdef)

        # write segment summary (known segments)
        for vseg in flag.known:
            segsum = segsumtab.RowType()
            for col in segsumtab.columnnames:  # default all columns to None
                setattr(segsum, col, None)
            segsum.segment_def_id = segdef.segment_def_id
            segsum.set(map(LIGOTimeGPS, map(float, vseg)))
            segsum.comment = None
            segsum.segment_sum_id = lsctables.SegmentSumTable.get_next_id()
            segsumtab.append(segsum)

        # write segment table (active segments)
        for aseg in flag.active:
            seg = segtab.RowType()
            for col in segtab.columnnames:  # default all columns to None
                setattr(seg, col, None)
            seg.segment_def_id = segdef.segment_def_id
            seg.set(map(LIGOTimeGPS, map(float, aseg)))
            seg.segment_id = lsctables.SegmentTable.get_next_id()
            segtab.append(seg)

    return segdeftab, segsumtab, segtab


def write_ligolw(flags, f, **kwargs):
    """Write this `DataQualityFlag` to the given LIGO_LW Document

    Parameters
    ----------
    flags : `DataQualityFlag`, `DataQualityDict`
        `gwpy.segments` object to wriet

    f : `str`, `file`, :class:`~glue.ligolw.ligolw.Document`
        the file or document to write into

    **kwargs
        keyword arguments to use when writing

    See also
    --------
    gwpy.io.ligolw.write_ligolw_tables
        for details of acceptabled keyword arguments
    """
    if isinstance(flags, DataQualityFlag):
        flags = {flags.name: flags}

    llwtables = dqdict_to_ligolw_tables(flags)
    return write_tables(f, llwtables, **kwargs)


# -- register -----------------------------------------------------------------

# register methods for DataQualityDict
registry.register_reader('ligolw', DataQualityFlag, read_ligolw_flag)
registry.register_writer('ligolw', DataQualityFlag, write_ligolw)
registry.register_identifier('ligolw', DataQualityFlag, identify_ligolw)

# register methods for DataQualityDict
registry.register_reader('ligolw', DataQualityDict, read_ligolw_dict)
registry.register_writer('ligolw', DataQualityDict, write_ligolw)
registry.register_identifier('ligolw', DataQualityDict, identify_ligolw)
