# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines the DataQualityFlag object, representing a
named set of segments used in defining GW instrument state or
data quality information.
"""

import operator
import numpy
import re
import warnings
from math import (floor, ceil)

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
    EntryClass = Segment
    ListClass = SegmentList
    __slots__ = ('_active', '_valid', 'ifo', 'name', 'version',
                 'category', 'comment')
    def __init__(self, name=None, active=[], valid=[], category=None,
                 comment=None):
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
        self.category = category
        self.comment = comment

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
        self._active = self.ListClass(map(self.EntryClass,
                                          segmentlist)).coalesce()
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
        self._valid = self.ListClass(map(self.EntryClass,
                                         segmentlist)).coalesce()

    def efficiency(self, trigtable):
        raise NotImplementedError("Class-level efficiency calculation has "
                                  "not been implemented yet.")

    def __getitem__(self, item):
        if isinstance(item, int) or isinstance(item, slice):
            return self.active[item]
        else:
            new = self.__class__()
            new.ifo = self.ifo
            new.name = self.name
            new.version = self.version
            new.active = self.active[item]
            new.valid = self.valid
            return new

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
        gpsend = int(ceil(float(gpsend)))
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

    def round(self):
        """Expand this `DataQualityFlag` to integer segments, with each
        segment protracted to the nearest integer boundaries
        """
        new = self.copy()
        new.active = self.ListClass([self.EntryClass(int(floor(s[0])),
                                                     int(ceil(s[1]))) for
                                     s in new.active])
        new.valid = self.ListClass([self.EntryClass(int(floor(s[0])),
                                                    int(ceil(s[1]))) for
                                    s in new.valid])
        return new.coalesce()

    def coalesce(self):
        """Run the coalesce method over both the active and valid
        SegmentLists
        """
        self.active = self.active.coalesce()
        self.valid = self.valid.coalesce()
        return self

    def to_ligolw(self):
        """Return a formatted LIGO_LW document describing the full
        content of this `DataQualityFlag`.

        Returns
        -------
        xmldoc : :class`~glue.ligolw.ligolw.Document`
            a formatted LIGO_LW XML document with the following tables:

                - :class:`~glue.ligolw.lsctables.VetoDefTable` -
                  defining each of the valid segments
                - :class:`~glue.ligolw.lsctables.SegmentDefTable` -
                  defining the segments for this flag
                - :class:`~glue.ligolw.lsctables.SegmentSumTable` -
                  summarising the valid segments, this should be a
                  one-to-one mapping with the `VetoDefTable`
                - :class:`~glue.ligolw.lsctables.SegmentTable` -
                  listing each of the active segments
        """
        raise NotImplementedError()

    def to_ligolw_veto_definer_table(self):
        """Return a formatted LIGO_LW veto-definer table describing
        the valid segments for this `DataQualityFlag`

        Returns
        -------
        lsctable : :class:`~glue.ligolw.lsctable.VetoDefTable`
            XML table defining each of the valid segments for this flag
        """
        raise NotImplementedError()


    def to_ligolw_segment_table(self):
        """Return a formatted LIGO_LW segment table describing
        the active segments for this `DataQualityFlag`

        Returns
        -------
        lsctable : :class:`~glue.ligolw.lsctable.SegmentTable`
            XML table listing the active segments for this flag
        """
        raise NotImplementedError()

    def __repr__(self):
        indent = " " * len("<%s(" % self.__class__.__name__)
        valid = str(self.valid).replace("\n",
                                         "\n%s      " % indent)
        active = str(self.active).replace("\n",
                                           "\n%s       " % indent)
        return ("<{1}(valid={2},\n{0}active={3},\n{0}ifo={4},\n{0}name={5},\n"
                "{0}version={6},\n{0}comment={7})>".format(
                    indent, self.__class__.__name__, valid, active,
                    repr(self.ifo), repr(self.name), repr(self.version),
                    repr(self.comment)))

    def copy(self):
        new = self.__class__()
        new.ifo = self.ifo
        new.name = self.name
        new.version = self.version
        new.comment = self.comment
        new.valid = self.ListClass([self.EntryClass(s[0], s[1]) for
                                    s in self.valid])
        new.active = self.ListClass([self.EntryClass(s[0], s[1]) for
                                     s in self.active])
        return new

class DataQualityList(list):
    EntryClass = DataQualityFlag
    def __init__(self, *entries):
        list.__init__(*entries)

    def to_ligolw(self):
        raise NotImplementedError()


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
