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

"""Read segment XML into DataQualityFlags
"""

import re
import datetime

from glue.lal import LIGOTimeGPS
from glue.ligolw import (ligolw, table as ligolw_table, utils as ligolw_utils,
                         lsctables)

from astropy.time import Time
from astropy.io import registry

from ... import version
from ...segments import (Segment, SegmentList, DataQualityFlag)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_ligolw_segments(file, flag=None):
    """Read segments for the given flag from the LIGO_LW XML file
    """
    if isinstance(file, basestring):
        f = open(file, 'r')
    else:
        f = file
    xmldoc,_ = ligolw_utils.load_fileobj(f)
    seg_def_table = ligolw_table.get_table(xmldoc,
                                           lsctables.SegmentDefTable.tableName)
    flags = []
    dqflag = None
    for row in seg_def_table:
        name = ':'.join(["".join(row.get_ifos()), row.name])
        if row.version:
            name += ':%d' % row.version
        if flag is None or name == flag:
            dqflag = DataQualityFlag(name)
            id_ = row.segment_def_id
            break
    if not dqflag:
        raise ValueError("No segment definition found for flag='%s' in "
                         "file '%s'" % (flag, f.name))
    dqflag.valid = SegmentList()
    seg_sum_table = ligolw_table.get_table(xmldoc,
                                           lsctables.SegmentSumTable.tableName)
    for row in seg_sum_table:
        if row.segment_def_id == id_:
            try:
                dqflag.valid.append(row.get())
            except AttributeError:
                dqflag.valid.append(Segment(row.start_time, row.end_time))
    dqflag.active = SegmentList()
    seg_table = ligolw_table.get_table(xmldoc, lsctables.SegmentTable.tableName)
    for row in seg_table:
        if row.segment_def_id == id_:
            try:
                dqflag.active.append(row.get())
            except AttributeError:
                dqflag.active.append(Segment(row.start_time, row.end_time))
    if isinstance(file, basestring):
        f.close()
    return dqflag


_re_xml = re.compile(r'(xml|xml.gz)\Z')

def identify_ligolw(*args, **kwargs):
    filename = args[1]
    if isinstance(filename, file):
        filename = filename.name
    if _re_xml.search(filename):
        return True
    else:
        return False


def write_ligolw(flag, fobj, **kwargs):
    """Write this `DataQualityFlag` to XML in LIGO_LW format
    """
    if isinstance(fobj, ligolw.Document):
        return write_to_xmldoc(flag, fobj, **kwargs)
    elif isinstance(fobj, basestring):
        fobj = open(fobj, 'w')
        close = True
    else:
        close = False
    xmldoc = ligolw.Document()
    xmldoc.appendChild(ligolw.LIGO_LW())
    # TODO: add process information
    write_to_xmldoc(flag, xmldoc)
    xmldoc.write(fobj)
    if close:
        fobj.close()


def write_to_xmldoc(flag, xmldoc, process_id=None):
    """Write this `DataQualityFlag` to the given LIGO_LW Document
    """
    # write SegmentDefTable
    try:
        segdeftab = ligolw_table.get_table(xmldoc,
                                    lsctables.SegmentDefTable.tableName)
    except ValueError:
        segdeftab = lsctables.New(lsctables.SegmentDefTable,
                                  columns=['ifos', 'name', 'version',
                                           'comment', 'insertion_time',
                                           'segment_def_id', 'process_id'])
        xmldoc.childNodes[-1].appendChild(segdeftab)
    segdef = lsctables.SegmentDef()
    segdef.set_ifos([flag.ifo])
    segdef.name = flag.name
    segdef.version = flag.version
    segdef.comment = flag.comment
    segdef.insertion_time = int(Time(datetime.datetime.now(),
                                    scale='utc').gps)
    segdef.segment_def_id = lsctables.SegmentDefTable.get_next_id()
    segdef.process_id = process_id
    segdeftab.append(segdef)

    # write SegmentSumTable
    try:
        segsumtab = ligolw_table.get_table(xmldoc,
                                    lsctables.SegmentSumTable.tableName)
    except ValueError:
        segsumtab = lsctables.New(lsctables.SegmentSumTable,
                                  columns=['segment_def_id', 'start_time',
                                           'start_time_ns', 'end_time',
                                           'end_time_ns', 'comment',
                                           'segment_sum_id', 'process_id'])
        xmldoc.childNodes[-1].appendChild(segsumtab)
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
        segtab = ligolw_table.get_table(xmldoc,
                                        lsctables.SegmentTable.tableName)
    except ValueError:
        segtab = lsctables.New(lsctables.SegmentTable,
                               columns=['process_id', 'segment_id',
                                        'segment_def_id', 'start_time',
                                        'start_time_ns', 'end_time',
                                        'end_time_ns'])
        xmldoc.childNodes[-1].appendChild(segtab)
    for aseg in flag.active:
        seg = lsctables.Segment()
        seg.segment_def_id = segdef.segment_def_id
        seg.set(map(LIGOTimeGPS, map(float, aseg)))
        seg.segment_id = lsctables.SegmentTable.get_next_id()
        seg.process_id = process_id
        segtab.append(seg)

    return xmldoc

registry.register_reader("ligolw", DataQualityFlag, read_ligolw_segments,
                         force=True)
registry.register_writer("ligolw", DataQualityFlag, write_ligolw,
                         force=True)
registry.register_identifier("ligolw", DataQualityFlag, identify_ligolw)
