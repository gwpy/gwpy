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

from glue.lal import LIGOTimeGPS
from glue.ligolw.ligolw import (Document, LIGO_LW)
from glue.ligolw.utils.ligolw_add import ligolw_add

from astropy.time import Time
from astropy.io import registry

from ... import version
from ...io.ligolw import (identify_ligolw_file, GWpyContentHandler)
from ...io.cache import file_list
from ...segments import (Segment, DataQualityFlag, DataQualityDict)
from ...table import lsctables

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def read_flag_dict(f, flags=None, gpstype=LIGOTimeGPS, coalesce=True,
                   contenthandler=GWpyContentHandler, nproc=1):
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
        and ``valid`` segments seeded from the XML tables in the given
        file ``fp``.
    """
    if nproc != 1:
        return DataQualityDict.read(f, flags, coalesce=coalesce,
                                    gpstype=gpstype,
                                    contenthandler=contenthandler,
                                    format='cache', nproc=nproc)

    # generate Document and populate
    xmldoc = Document()
    files = [fp.name if isinstance(fp, file) else fp for fp in file_list(f)]
    ligolw_add(xmldoc, files, non_lsc_tables_ok=True,
               contenthandler=contenthandler)

    # read segment definers and generate DataQualityFlag object
    seg_def_table = lsctables.SegmentDefTable.get_table(xmldoc)

    # find flags
    if isinstance(flags, (unicode, str)):
        flags = flags.split(',')
    out = DataQualityDict()
    id_ = dict()
    if flags is not None and len(flags) == 1 and flags[0] is None:
        out[None] = DataQualityFlag()
        id_[None] = None
    for row in seg_def_table:
        name = ':'.join([''.join(row.get_ifos()), row.name])
        if row.version:
            name += ':%d' % row.version
        if flags is None:
            out[name] = DataQualityFlag()
            id_[name] = row.segment_def_id
        elif name in flags:
            out[name] = DataQualityFlag(name)
            id_[name] = row.segment_def_id
    if flags is None and not len(out.keys()):
        raise RuntimeError("No segment definitions found in file.")
    elif flags is not None and len(out.keys()) != len(flags):
        for flag in flags:
            if flag not in out:
                raise ValueError("No segment definition found for flag='%s' "
                                 " in file.")
    # read segment summary table as 'valid'
    seg_sum_table = lsctables.SegmentSumTable.get_table(xmldoc)
    for row in seg_sum_table:
        for flag in out:
            if id_[flag] is None or row.segment_def_id == id_[flag]:
                try:
                    s = row.get()
                except AttributeError:
                    s = row.start_time, row.end_time
                out[flag].valid.append(Segment(gpstype(s[0]), gpstype(s[1])))
    for dqf in out:
        if coalesce or dqf is None:
            out[dqf].coalesce()
    # read segment table as 'active'
    seg_table = lsctables.SegmentTable.get_table(xmldoc)
    for row in seg_table:
        for flag in out:
            if id_[flag] is None or row.segment_def_id == id_[flag]:
                try:
                    s = row.get()
                except AttributeError:
                    s = row.start_time, row.end_time
                out[flag].active.append(Segment(gpstype(s[0]), gpstype(s[1])))
    for dqf in out:
        if coalesce or dqf is None:
            out[dqf].coalesce()
    return out


def read_flag(fp, flag=None, **kwargs):
    """Read a single `DataQualityFlag` from a LIGO_LW XML file
    """
    return read_flag_dict(fp, flags=[flag], **kwargs)[flag]


def write_ligolw(flag, fobj, **kwargs):
    """Write this `DataQualityFlag` to XML in LIGO_LW format
    """
    if isinstance(fobj, Document):
        return write_to_xmldoc(flag, fobj, **kwargs)
    elif isinstance(fobj, (str, unicode)):
        fobj = open(fobj, 'w')
        close = True
    else:
        close = False
    xmldoc = Document()
    xmldoc.appendChild(LIGO_LW())
    # TODO: add process information
    write_to_xmldoc(flag, xmldoc)
    xmldoc.write(fobj)
    if close:
        fobj.close()


def write_to_xmldoc(flags, xmldoc, process_id=None):
    """Write this `DataQualityFlag` to the given LIGO_LW Document
    """
    if isinstance(flags, DataQualityFlag):
        flags = {str(flags): flags}

    # write SegmentDefTable
    try:
        segdeftab = lsctables.SegmentDefTable.get_table(xmldoc)
    except ValueError:
        segdeftab = lsctables.New(lsctables.SegmentDefTable,
                                  columns=['ifos', 'name', 'version',
                                           'comment', 'insertion_time',
                                           'segment_def_id', 'process_id'])
        xmldoc.childNodes[-1].appendChild(segdeftab)

    for flag in flags:
        segdef = lsctables.SegmentDef()
        segdef.set_ifos([flag.ifo])
        segdef.name = flag.tag
        segdef.version = flag.version
        segdef.comment = flag.comment
        segdef.insertion_time = int(Time(datetime.datetime.now(),
                                         scale='utc').gps)
        segdef.segment_def_id = lsctables.SegmentDefTable.get_next_id()
        segdef.process_id = process_id
        segdeftab.append(segdef)

    # write SegmentSumTable
    try:
        segsumtab = lsctables.SegmentSumTable.get_table(xmldoc)
    except ValueError:
        segsumtab = lsctables.New(lsctables.SegmentSumTable,
                                  columns=['segment_def_id', 'start_time',
                                           'start_time_ns', 'end_time',
                                           'end_time_ns', 'comment',
                                           'segment_sum_id', 'process_id'])
        xmldoc.childNodes[-1].appendChild(segsumtab)
    for flag in flags.iterkeys():
        for vseg in flag.valid:
            segsum = lsctables.SegmentSum()
            segsum.segment_def_id = segdef.segment_def_id
            segsum.set(map(LIGOTimeGPS, map(float, vseg)))
            segsum.comment = None
            segsum.segment_sum_id = lsctables.SegmentSumTable.get_next_id()
            segsum.process_id = process_id
            segsumtab.append(segsum)

    # write SegmentTable
    try:
        segtab = lsctables.SegmentTable.get_table(xmldoc)
    except ValueError:
        segtab = lsctables.New(lsctables.SegmentTable,
                               columns=['process_id', 'segment_id',
                                        'segment_def_id', 'start_time',
                                        'start_time_ns', 'end_time',
                                        'end_time_ns'])
        xmldoc.childNodes[-1].appendChild(segtab)
    for flag in flags.iterkeys():
        for aseg in flag.active:
            seg = lsctables.Segment()
            seg.segment_def_id = segdef.segment_def_id
            seg.set(map(LIGOTimeGPS, map(float, aseg)))
            seg.segment_id = lsctables.SegmentTable.get_next_id()
            seg.process_id = process_id
            segtab.append(seg)
    return xmldoc


registry.register_reader('ligolw', DataQualityFlag, read_flag)
registry.register_writer('ligolw', DataQualityFlag, write_ligolw)
registry.register_identifier('ligolw', DataQualityFlag, identify_ligolw_file)

registry.register_reader('ligolw', DataQualityDict, read_flag_dict)
registry.register_identifier('ligolw', DataQualityDict, identify_ligolw_file)
