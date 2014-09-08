# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""This module provides a discovery mechanism for LIGO_LW XML trigger
files written on the LIGO Data Grid according to the conventions in
LIGO-T1300468.
"""

import glob
import os.path
import re

from glue.lal import (Cache, CacheEntry)

from ... import version
from ...segments import Segment
from ...utils import gprint

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

TRIGFIND_BASE_PATH = "/home/detchar/triggers"
re_dash = re.compile('-')


def find_trigger_urls(channel, etg, gpsstart, gpsend, verbose=False):
    """Find the paths of trigger files that represent the given
    observatory, channel, and ETG (event trigger generator) for a given
    GPS [start, end) segment.
    """
    if etg.lower() == 'omicron':
        etg = '?micron'

    # construct search
    span = Segment(gpsstart, gpsend)
    ifo, channel = channel.split(':', 1)
    trigtype = "%s_%s" % (channel, etg.lower())
    epoch = '*'
    searchbase = os.path.join(TRIGFIND_BASE_PATH, epoch, ifo, trigtype)
    gpsdirs = range(int(str(gpsstart)[:5]), int(str(gpsend)[:5])+1)
    trigform = ('%s-%s_%s-%s-*.xml*'
                % (ifo, re_dash.sub('_', channel), etg.lower(), '[0-9]'*10))

    # perform and cache results
    out = Cache()
    for gpsdir in gpsdirs:
        gpssearchpath = os.path.join(searchbase, str(gpsdir), trigform)
        if verbose:
            gprint("Searching %s..." % os.path.split(gpssearchpath)[0],
                   end =' ')
        gpscache = Cache(map(CacheEntry.from_T050017,
                             glob.glob(os.path.join(searchbase, str(gpsdir),
                                                    trigform))))
        out.extend(gpscache.sieve(segment=span))
        if verbose:
            gprint("%d found" % len(gpscache.sieve(segment=span)))
    out.sort(key=lambda e: e.path)

    return out
