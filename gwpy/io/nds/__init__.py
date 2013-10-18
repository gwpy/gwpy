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

"""Wrapper to the nds2-client package, providing network access
to LIGO data.
"""

import nds2
import os
import sys
from math import (floor, ceil)

from ... import (version, detector)
from ...detector import Channel
from ...time import Time
from ...timeseries import (TimeSeries, TimeSeriesList)

from .kerberos import *

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


try:
    from collections import OrderedDict
except ImportError:
    from astropy.utils import OrderedDict
finally:
    DEFAULT_HOSTS = OrderedDict([
                    (None,('ldas-pcdev4.ligo.caltech.edu', 31200)),
                    (None,('nds.ligo.caltech.edu', 31200)),
                    (detector.LHO_4k.prefix,('nds.ligo-wa.caltech.edu', 31200)),
                    (detector.LLO_4k.prefix,('nds.ligo-la.caltech.edu', 31200)),
                    (detector.CIT_40.prefix,('nds40.ligo.caltech.edu', 31200))])


class NDSOutputContext(object):
    def __init__(self, stdout=sys.stdout, stderr=sys.stderr):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class NDSWarning(UserWarning):
    pass


_HOST_RESOLTION_ORDER = ['nds.ligo.caltech.edu'] + DEFAULT_HOSTS.values()
def host_resolution_order(ifo):
    hosts = []
    if ifo in DEFAULT_HOSTS:
        hosts.append(DEFAULT_HOSTS[ifo])
    for difo,hp in DEFAULT_HOSTS.iteritems():
        if difo != ifo:
            hosts.append(hp)
    return hosts
