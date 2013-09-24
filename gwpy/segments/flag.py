# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines the DataQualityFlag object, representing a
named set of segments used in defining GW instrument state or
data quality information.
"""

import math
import operator
import numpy
import re
import warnings

from astropy.io import registry as io_registry

from .segments import Segment, SegmentList

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ["DataQualityFlag"]


class DataQualityFlag(object):
    """A representation of a named set of segments.

    Each DataQualityFlag has a name, and two sets of segments:

        - '`valid`' segments: times during which this flag was well defined
        - '`active`' segments: times during which this flag was active
    """
    def __init__(self, name=None, active=[], valid=[]):
        """Define a new DataQualityFlag, with a name, a set of active
        segments, and a set of valid segments

        Parameters
        ----------

        name : str, optional
            The name of this data quality flag. This should be of the form
            IFO:DQ-FLAG_NAME:VERSION, e.g. 'H1:DMT-SCIENCE:1'
        active : `~segments.SegmentList`, optional
            A list of active segments for this flag
        valid : `~segments.SegmentList`, optional
            A list of valid segments for this flag
        """
        self.ifo, self.name, self.version = parse_flag_name(name)
        self.valid = valid
        self.active = active

    @property
    def active(self):
        """The set of segments during which this DataQualityFlag was
        active
        """
        if hasattr(self, "valid"):
            return (self._active & self.valid).coalesce()
        else:
            return self._active
    @active.setter
    def active(self, segmentlist):
        self._active = SegmentList(map(Segment, segmentlist)).coalesce()
        if not self.valid and len(segmentlist):
            self.valid = [segmentlist.extent()]

    @property
    def valid(self):
        """The set of segments during which this DataQualityFlag was
        valid, and its state was well defined.
        """
        return self._valid
    @valid.setter
    def valid(self, segmentlist):
        self._valid = SegmentList(map(Segment, segmentlist)).coalesce()

    def efficiency(self, trigtable):
        raise NotImplementedError("Class-level efficiency calculation has "
                                  "not been implemented yet.")

    @classmethod
    def query(cls, flag, gpsstart, gpsend,
              url="https://segdb.ligo.caltech.edu"):
        """Query the segment database URL as give, returning segments
        during which the given flag was defined and active.

        Parameters
        ----------

        flag : str
            The name of the flag for which to query
        gpsstart : [ `float` | `LIGOTimeGPS` | `~gwpy.time.Time` ]
            GPS start time of the query
        gpsend : [ `float` | `LIGOTimeGPS` | `~gwpy.time.Time` ]
            GPS end time of the query
        url : str
            URL of the segment database, defaults to segdb.ligo.caltech.edu

        Returns
        -------

        A new DataQualityFlag, with the `valid` and `active` lists
        filled appropriately.
        """
        from glue.segmentdb import (segmentdb_utils as segdb_utils,
                                    query_engine as segdb_engine)
        ifo, name, version = parse_flag_name(flag)
        if not version:
            version = '*'
        gpsstart = int(float(gpsstart))
        gpsend = int(math.ceil(float(gpsend)))
        connection = segdb_utils.setup_database(url)
        engine = segdb_engine.LdbdQueryEngine(connection)

        seg_def = segdb_utils.expand_version_number(engine,
                                                    (ifo, name, version,
                                                     gpsstart, gpsend, 0, 0))
        segs = segdb_utils.query_segments(engine, 'segment', seg_def)
        seg_sum = segdb_utils.query_segments(engine, 'segment_summary', seg_def)
        # build output
        return cls(flag, valid=reduce(operator.or_, seg_sum).coalesce(),
                   active=reduce(operator.or_, segs).coalesce())

    read = classmethod(io_registry.read)

_re_inv = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<name>[^/]+):(?P<version>\d+)\Z")
_re_in = re.compile(r"\A(?P<ifo>[A-Z]\d):(?P<name>[^/]+)\Z")
_re_nv = re.compile(r"\A(?P<name>[^/]+):(?P<ver>\d+)\Z")

def parse_flag_name(name):
    """Internal method to parse a `string` name into constituent
    `ifo, `name` and `version` components.

    Returns
    -------

    A tuple of (ifo, name, version) component parts
    """
    if name is None:
        return None, None, None
    if _re_inv.match(name):
        match = _re_inv.match(name).groupdict()
        return match['ifo'], match['name'], int(match['version'])
    elif _re_in.match(name):
        match = _re_in.match(name).groupdict()
        return match['ifo'], match['name'], None
    elif _re_nv.match(name):
        match = _re_nv.match(name).groupdict()
        return None, match['name'], int(match['version'])
    warnings.warn("No flag name structure detected in '%s', flags should be "
                  "named as 'IFO:DQ-FLAG_NAME:VERSION'" % name)
    return None, name, None
