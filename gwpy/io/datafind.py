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
