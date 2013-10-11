# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Read segment XML into DataQualityFlags
"""

import re

from glue.ligolw import (table as ligolw_table, utils as ligolw_utils,
                         lsctables)

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

registry.register_reader("ligolw", DataQualityFlag, read_ligolw_segments,
                         force=True)
registry.register_identifier("ligolw", DataQualityFlag, identify_ligolw)
