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

from __future__ import (absolute_import, print_function)

import enum
import operator
import os
import re
import sys
import warnings
from functools import wraps

import numpy

import nds2

from ..time import to_gps
from .kerberos import kinit
from ..utils.compat import OrderedDict

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

NDS1_HOSTNAME = re.compile('[a-z]1nds[0-9]\Z')

DEFAULT_HOSTS = OrderedDict([
    (None, ('nds.ligo.caltech.edu', 31200)),
    ('H1', ('nds.ligo-wa.caltech.edu', 31200)),
    ('H0', ('nds.ligo-wa.caltech.edu', 31200)),
    ('L1', ('nds.ligo-la.caltech.edu', 31200)),
    ('L0', ('nds.ligo-la.caltech.edu', 31200)),
    ('C1', ('nds40.ligo.caltech.edu', 31200)),
    ('C0', ('nds40.ligo.caltech.edu', 31200))])

# -- enums --------------------------------------------------------------------

class Nds2Enum(enum.Enum):
    """`~enum.Enum` providing `any` property with logical OR of members
    """
    @classmethod
    def any(cls):
        return reduce(operator.or_, (x.value for x in cls))


NDS2_TYPE_NAME = {
    0: 'unknown',
    1: 'online',
    2: 'raw',
    4: 'reduced',
    8: 's-trend',
    16: 'm-trend',
    32: 'test-pt',
    64: 'static',
}

class Nds2ChannelType(Nds2Enum):
    """`~enum.Enum` of NDS2 channel types
    """
    @property
    def name(self):
        return NDS2_TYPE_NAME[self.value]

    @classmethod
    def names(cls):
        return [x.name for x in cls]

    @classmethod
    def find(cls, name):
        try:
            return cls._member_map_[name]
        except KeyError as e:
            for ctype in cls._member_map_.values():
                if ctype.name == name:
                    return ctype
            raise e

    UNKNOWN = 0
    ONLINE = 1
    RAW = 2
    RDS = 4
    STREND = 8
    MTREND = 16
    TEST_POINT = 32
    STATIC = 64


NUMPY_DTYPE = {
    1: numpy.int16,
    2: numpy.int32,
    4: numpy.int64,
    8: numpy.float32,
    16: numpy.float64,
    32: numpy.complex64,
    64: numpy.uint32,
}


class Nds2DataType(Nds2Enum):
    """`~enum.Enum` of NDS2 data types
    """
    @property
    def numpy_dtype(self):
        return NUMPY_DTYPE[self.value]

    @classmethod
    def find(cls, dtype):
        try:
            return cls._member_map_[dtype]
        except KeyError as e:
            dtype = numpy.dtype(dtype).type
            for ndstype in cls._member_map_.values():
                if ndstype.numpy_dtype is dtype:
                    return ndstype
            raise e

    UNKNOWN = 0
    INT16 = 1
    INT32 = 2
    INT64 = 4
    FLOAT32 = 8
    FLOAT64 = 16
    COMPLEX32 = 32
    UINT32 = 64


# -- warning suppression ------------------------------------------------------

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


# -- connection utilities -----------------------------------------------------

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
    # set default port for NDS1 connections (required, I think)
    if port is None and NDS1_HOSTNAME.match(host):
        port = 8088

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


def open_connection(func):
    """Decorate a function to create a `nds2.connection` if required
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if kwargs.get('connection', None) is None:
            try:
                host = kwargs.pop('host')
            except KeyError:
                raise TypeError("one of `connection` or `host` is required "
                                "to query NDS2 server")
            kwargs['connection'] = auth_connect(host, kwargs.pop('port', None))
        return func(*args, **kwargs)
    return wrapped_func


def parse_nds2_enums(func):
    """Decorate a function to translate a type string into an integer
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        for kw, enum_ in (('type', Nds2ChannelType),
                          ('dtype', Nds2DataType)):
            kwargs.setdefault(kw, enum_.any())
            if not isinstance(kwargs[kw], int):
                kwargs[kw] = enum_.find(kwargs[kw]).value
        return func(*args, **kwargs)
    return wrapped_func


# -- query methods ------------------------------------------------------------

@open_connection
@parse_nds2_enums
def find_channels(channels, connection=None, host=None, port=None,
                  sample_rate=None, type=Nds2ChannelType.any(),
                  dtype=Nds2DataType.any(), unique=False, epoch=None):
    """Query an NDS2 server for channel information

    Parameters
    ----------
    channels : `list` of `str`
        list of channel names to query, each can include bash-style globs

    connection : `nds2.connection`, optional
        open NDS2 connection to use for query

    host : `str`, optional
        name of NDS2 server to query, required if ``connection`` is not
        given

    port : `int`, optional
        port number on host to use for NDS2 connection

    sample_rate : `int`, `float`, `tuple`, optional
        a single number, representing a specific sample rate to match,
        or a tuple representing a ``(low, high)` interval to match

    type : `int`, optional
        the NDS2 channel type to match

    dtype : `int`, optional
        the NDS2 data type to match

    unique : `bool`, optional, default: `False`
        require one (and only one) match per channel

    Returns
    -------
    channels : `list` of `nds2.channel`
        list of NDS2 channel objects

    See also
    --------
    nds2.connection.find_channels
        for documentation on the underlying query method
    """
    # set epoch
    if isinstance(epoch, tuple):
        connection.set_epoch(*epoch)
    elif epoch is not None:
        connection.set_epoch(epoch)

    # query for channels
    out = []
    if isinstance(sample_rate, (int, float)):
        sample_rate = (sample_rate, sample_rate)
    elif sample_rate is None:
        sample_rate = tuple()
    qargs = (type, dtype) + sample_rate

    for name in map(str, channels):
        found = connection.find_channels(name, *qargs)
        if unique and len(found) != 1:
            raise ValueError("unique NDS2 channel match not found for %r"
                             % name)
        out.extend(found)

    return out
