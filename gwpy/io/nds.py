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

from __future__ import print_function

import os
import sys
import warnings

import nds2

from ..time import to_gps
from .kerberos import kinit
from ..utils.compat import OrderedDict

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

DEFAULT_HOSTS = OrderedDict([
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
NDS2_CHANNEL_TYPESTR[max(NDS2_CHANNEL_TYPESTR.keys()) * 2] = 'rds'
NDS2_CHANNEL_TYPE = dict((val, key) for (key, val) in
                         NDS2_CHANNEL_TYPESTR.items())
# manually add RDS


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


warnings.simplefilter('always', NDSWarning)


def host_resolution_order(ifo, env='NDSSERVER', epoch='now',
                          lookback=14*86400):
    """Generate a logical ordering of NDS (host, port) tuples for this IFO

    Parameters
    ----------
    ifo : `str`
        prefix for IFO of interest
    env : `str`, optional
        environment variable name to use for server order,
        default ``'NDSSERVER'``. The contents of this variable should
        be a comma-separated list of `host:port` strings, e.g.
        ``'nds1.server.com:80,nds2.server.com:80'``
    epoch : `~gwpy.time.LIGOTimeGPS`, `float`
        GPS epoch of data requested
    lookback : `float`
        duration of spinning-disk cache. This value triggers defaulting to
        the CIT NDS2 server over those at the LIGO sites

    Returns
    -------
    hro : `list` of `2-tuples <tuple>`
        ordered `list` of ``(host, port)`` tuples
    """
    hosts = []
    # if given environment variable exists, it will contain a
    # comma-separated list of host:port strings giving the logical ordering
    if env and os.getenv(env):
        for host in os.getenv(env).split(','):
            try:
                host, port = host.rsplit(':', 1)
            except ValueError:
                port = None
            else:
                port = int(port)
            if (host, port) not in hosts:
                hosts.append((host, port))
    # If that host fails, return the server for this IFO and the backup at CIT
    if to_gps('now') - to_gps(epoch) > lookback:
        ifolist = [None, ifo]
    else:
        ifolist = [ifo, None]
    for difo in ifolist:
        try:
            host, port = DEFAULT_HOSTS[difo]
        except KeyError:
            warnings.warn('No default host found for ifo %r' % ifo)
        else:
            if (host, port) not in hosts:
                hosts.append((host, port))
    return list(hosts)


def auth_connect(host, port=None):
    """Open a connection to the given host and port

    This method will catch exceptions related to kerberos authentication,
    and execute a kinit() for the user before connecting again

    Parameters
    ----------
    host : `str`
        name of server with which to connect
    port : `int`, optional
        connection port
    """
    if port is None:
        def _connect():
            return nds2.connection(host)
    else:
        def _connect():
            return nds2.connection(host, port)
    try:
        connection = _connect()
    except RuntimeError as e:
        if str(e).startswith('Request SASL authentication'):
            print('\nError authenticating against %s' % host,
                  file=sys.stderr)
            kinit()
            connection = _connect()
        else:
            raise
    return connection
