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

"""Wrapper to the nds2-client package, providing network access
to LIGO data.
"""

import sys

import nds2

from .. import version
from .kerberos import *

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

try:
    from collections import OrderedDict
except ImportError:
    from ordereddict import OrderedDict

DEFAULT_HOSTS = OrderedDict([
    (None, ('ldas-pcdev4.ligo.caltech.edu', 31200)),
    (None, ('nds.ligo.caltech.edu', 31200)),
    ('H1', ('nds.ligo-wa.caltech.edu', 31200)),
    ('H0', ('nds.ligo-wa.caltech.edu', 31200)),
    ('L1', ('nds.ligo-la.caltech.edu', 31200)),
    ('L0', ('nds.ligo-la.caltech.edu', 31200)),
    ('C1', ('nds40.ligo.caltech.edu', 31200)),
    ('C0', ('nds40.ligo.caltech.edu', 31200))])

# set type dicts
NDS2_CHANNEL_TYPESTR = {}
for ctype in (nds2.channel.CHANNEL_TYPE_RAW,
              nds2.channel.CHANNEL_TYPE_ONLINE,
              nds2.channel.CHANNEL_TYPE_RDS,
              nds2.channel.CHANNEL_TYPE_STREND,
              nds2.channel.CHANNEL_TYPE_MTREND,
              nds2.channel.CHANNEL_TYPE_STATIC,
              nds2.channel.CHANNEL_TYPE_TEST_POINT):
    NDS2_CHANNEL_TYPESTR[ctype] = nds2.channel_channel_type_to_string(ctype)
NDS2_CHANNEL_TYPE = dict((val, key) for (key, val) in
                         NDS2_CHANNEL_TYPESTR.iteritems())


class NDSOutputContext(object):
    def __init__(self, stdout=sys.stdout, stderr=sys.stderr):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, *args):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr


class NDSWarning(UserWarning):
    pass


def host_resolution_order(ifo):
    hosts = []
    if ifo in DEFAULT_HOSTS:
        hosts.append(DEFAULT_HOSTS[ifo])
    for difo, hp in DEFAULT_HOSTS.iteritems():
        if difo != ifo and hp not in hosts:
            hosts.append(hp)
    return hosts
