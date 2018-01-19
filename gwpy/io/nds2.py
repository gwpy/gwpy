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

# pylint: disable=wrong-import-order
import enum
import operator
import os
import re
import sys
import warnings
from collections import OrderedDict
from functools import wraps

from six.moves import reduce

import numpy

try:
    import nds2
except ImportError:
    HAS_NDS2 = False
else:
    HAS_NDS2 = True

from ..time import to_gps
from .kerberos import kinit

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

NDS1_HOSTNAME = re.compile(r'[a-z]1nds[0-9]\Z')

DEFAULT_HOSTS = OrderedDict([
    (None, ('nds.ligo.caltech.edu', 31200)),
    ('H1', ('nds.ligo-wa.caltech.edu', 31200)),
    ('H0', ('nds.ligo-wa.caltech.edu', 31200)),
    ('L1', ('nds.ligo-la.caltech.edu', 31200)),
    ('L0', ('nds.ligo-la.caltech.edu', 31200)),
    ('C1', ('nds40.ligo.caltech.edu', 31200)),
    ('C0', ('nds40.ligo.caltech.edu', 31200))])


# -- enums --------------------------------------------------------------------

class Nds2Enum(enum.Enum):  # pylint:  disable=too-few-public-methods
    """`~enum.Enum` providing `any` property with logical OR of members
    """
    @classmethod
    def any(cls):
        """The logical OR of all members in this enum
        """
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
        """The NDS2 string name for this channel type
        """
        return NDS2_TYPE_NAME[self.value]

    @classmethod
    def names(cls):
        """The list of all recognised channel type names
        """
        return [x.name for x in cls]

    @classmethod
    def find(cls, name):
        """Returns the NDS2 channel type corresponding to the given name
        """
        try:
            return cls._member_map_[name]
        except KeyError:
            for ctype in cls._member_map_.values():
                if ctype.name == name:
                    return ctype
            raise ValueError('%s is not a valid %s' % (name, cls.__name__))

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
    8: numpy.float32,  # pylint: disable=no-member
    16: numpy.float64,  # pylint: disable=no-member
    32: numpy.complex64,  # pylint: disable=no-member
    64: numpy.uint32,  # pylint: disable=no-member
}


class Nds2DataType(Nds2Enum):
    """`~enum.Enum` of NDS2 data types
    """
    @property
    def numpy_dtype(self):
        """The `numpy` type corresponding to this NDS2 type"""
        return NUMPY_DTYPE[self.value]

    @classmethod
    def find(cls, dtype):
        """Returns the NDS2 type corresponding to the given python type
        """
        try:
            return cls._member_map_[dtype]
        except KeyError as exc:
            dtype = numpy.dtype(dtype).type
            for ndstype in cls._member_map_.values():
                if ndstype.value and ndstype.numpy_dtype is dtype:
                    return ndstype
            raise exc

    UNKNOWN = 0
    INT16 = 1
    INT32 = 2
    INT64 = 4
    FLOAT32 = 8
    FLOAT64 = 16
    COMPLEX32 = 32
    UINT32 = 64


# -- warning suppression ------------------------------------------------------

class NDSWarning(UserWarning):
    """Warning about communicating with the Network Data Server
    """
    pass


warnings.simplefilter('always', NDSWarning)


# -- query utilities ----------------------------------------------------------

def _get_nds2_name(channel):
    """Returns the NDS2-formatted name for a channel

    Understands how to format NDS name strings from
    `gwpy.detector.Channel` and `nds2.channel` objects
    """
    if hasattr(channel, 'ndsname'):  # gwpy.detector.Channel
        return channel.ndsname
    if hasattr(channel, 'channel_type'):  # nds2.channel
        return '%s,%s' % (channel.name,
                          channel.channel_type_to_string(channel.channel_type))
    return str(channel)


def _get_nds2_names(channels):
    """Maps `_get_nds2_name` for a list of input channels
    """
    return map(_get_nds2_name, channels)


# -- connection utilities -----------------------------------------------------

def parse_nds_env(env='NDSSERVER'):
    """Parse the NDSSERVER environment variable into a list of hosts

    Parameters
    ----------
    env : `str`, optional
        environment variable name to use for server order,
        default ``'NDSSERVER'``. The contents of this variable should
        be a comma-separated list of `host:port` strings, e.g.
        ``'nds1.server.com:80,nds2.server.com:80'``

    Returns
    -------
    hostiter : `list` of `tuple`
        a list of (unique) ``(str, int)`` tuples for each host:port
        pair
    """
    hosts = []
    for host in os.getenv(env).split(','):
        try:
            host, port = host.rsplit(':', 1)
        except ValueError:
            port = None
        else:
            port = int(port)
        if (host, port) not in hosts:
            hosts.append((host, port))
    return hosts


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
        hosts = parse_nds_env(env)
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


def connect(host, port=None):
    """Open an `nds2.connection` to a given host and port

    Parameters
    ----------
    host : `str`
        name of server with which to connect

    port : `int`, optional
        connection port

    Returns
    -------
    connection : `nds2.connection`
        a new open connection to the given NDS host
    """
    if port is None:
        return nds2.connection(host)
    return nds2.connection(host, port)


def auth_connect(host, port=None):
    """Open an `nds2.connection` handling simple authentication errors

    This method will catch exceptions related to kerberos authentication,
    and execute a kinit() for the user before attempting to connect again.

    Parameters
    ----------
    host : `str`
        name of server with which to connect

    port : `int`, optional
        connection port

    Returns
    -------
    connection : `nds2.connection`
        a new open connection to the given NDS host
    """
    if not HAS_NDS2:
        raise ImportError("No module named nds2")

    # set default port for NDS1 connections (required, I think)
    if port is None and NDS1_HOSTNAME.match(host):
        port = 8088

    try:
        return connect(host, port)
    except RuntimeError as exc:
        if 'Request SASL authentication' in str(exc):
            print('\nError authenticating against %s' % host,
                  file=sys.stderr)
            kinit()
            return connect(host, port)
        else:
            raise


def open_connection(func):
    """Decorate a function to create a `nds2.connection` if required
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):  # pylint: disable=missing-docstring
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
    def wrapped_func(*args, **kwargs):  # pylint: disable=missing-docstring
        for kwd, enum_ in (('type', Nds2ChannelType),
                           ('dtype', Nds2DataType)):
            if kwargs.get(kwd, None) is None:
                kwargs[kwd] = enum_.any()
            elif not isinstance(kwargs[kwd], int):
                kwargs[kwd] = enum_.find(kwargs[kwd]).value
        return func(*args, **kwargs)
    return wrapped_func


# -- query methods ------------------------------------------------------------

@open_connection
@parse_nds2_enums
def find_channels(channels, connection=None, host=None, port=None,
                  sample_rate=None, type=Nds2ChannelType.any(),
                  dtype=Nds2DataType.any(), unique=False, epoch=None):
    # pylint: disable=unused-argument,redefined-builtin
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

    # format sample_rate as tuple for find_channels call
    if isinstance(sample_rate, (int, float)):
        sample_rate = (sample_rate, sample_rate)
    elif sample_rate is None:
        sample_rate = tuple()

    # query for channels
    out = []
    for name in _get_nds2_names(channels):
        try:
            name, ctype = name.rsplit(',', 1)
        except ValueError:
            ctype = type
        else:
            ctype = Nds2ChannelType.find(ctype).value
        found = connection.find_channels(name, ctype, dtype, *sample_rate)
        if unique and len(found) != 1:
            raise ValueError("unique NDS2 channel match not found for %r"
                             % name)
        out.extend(found)
    return out


@open_connection
def get_availability(channels, start, end,
                     connection=None, host=None, port=None):
    # pylint: disable=unused-argument
    """Query an NDS2 server for data availability

    Parameters
    ----------
    channels : `list` of `str`
        list of channel names to query, each name should be of the form
        ``name,type``, e.g. ``L1:GDS-CALIB_STRAIN,reduced`` in order to
        match results

    start : `int`
        GPS start time of query

    end : `int`
        GPS end time of query

    connection : `nds2.connection`, optional
        open NDS2 connection to use for query

    host : `str`, optional
        name of NDS2 server to query, required if ``connection`` is not
        given

    port : `int`, optional
        port number on host to use for NDS2 connection

    Returns
    -------
    segdict : `~gwpy.segments.SegmentListDict`
        dict of ``(name, SegmentList)`` pairs

    See also
    --------
    nds2.connection.get_availability
        for documentation on the underlying query method
    """
    from ..segments import (Segment, SegmentList, SegmentListDict)
    connection.set_epoch(start, end)
    names = _get_nds2_names(channels)
    result = connection.get_availability(names)
    out = SegmentListDict()
    for name, result in zip(_get_nds2_names(channels), result):
        out[name] = SegmentList([Segment(s.gps_start, s.gps_stop) for s in
                                 result.simple_list()])
    return out


def minute_trend_times(start, end):
    """Expand a [start, end) interval for use in querying for minute trends

    NDS2 requires start and end times for minute trends to be a multiple of
    60 (to exactly match the time of a minute-trend sample), so this function
    expands the given ``[start, end)`` interval to the nearest multiples.

    Parameters
    ----------
    start : `int`
        GPS start time of query

    end : `int`
        GPS end time of query

    Returns
    -------
    mstart : `int`
        ``start`` rounded down to nearest multiple of 60
    mend : `int`
        ``end`` rounded up to nearest multiple of 60
    """
    if start % 60:
        start = int(start) // 60 * 60
    if end % 60:
        end = int(end) // 60 * 60 + 60
    return int(start), int(end)
