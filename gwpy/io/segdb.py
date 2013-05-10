# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Thie module provides methods to query the LIGO-Virgo
'segment databases'.

Up to and including the year 2013, these databases used the IBM DB2
architecture. In the future, a new architecture will be used, but an
API has yet to be released. When it has been, GWpy will be updated
accordingly.
"""

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

try:
    from glue.segmentdb import (query_engine, segmentdb_utils)
except ImportError:
    GLUE_AVAIL = False
else:
    def query(gpsstart, gpsend, flags, url=DEFAULT_HOST):
        """Query the given database url for active segments for each
        of a list of flags in the given [gpsstart, gpsend) interval
        """
        pass
