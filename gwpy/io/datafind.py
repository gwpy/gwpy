# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""I/O wrapper to the glue.GWDataFindClient module
"""

import os
from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from glue import GWDataFindClient


def query(site, frametype, gpsstart, gpsend, urltype='file', on_gaps='error',
          server=os.getenv('LIGO_DATAFIND_SERVER', None)):
    """Query the LIGO_DATAFIND_SERVER for frame paths matching the
    given site, frametype, and [start, end) interval.
    """
    if server:
        host, port = server.split(':', 1)
    else:
        host = port = None
    conn = GWDataFindClient.GWDataFindHTTPConnection(host=host, port=port)
    cache = conn.find_frame_urls(site, frametype, gpsstart, gpsend,
                                       urltype=urltype, on_gaps=on_gaps)
    conn.close()
    return cache
