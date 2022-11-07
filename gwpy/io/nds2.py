# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

import enum
import operator
import os
import re
import warnings
from collections import OrderedDict
from functools import (reduce, wraps)

from ..time import to_gps
from ..utils.enum import NumpyTypeEnum
from .kerberos import kinit

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# regular expression to match LIGO-standard NDS1 hostnames (e.g x1nds0)
NDS1_HOSTNAME = re.compile(r'[a-z]1nds[0-9]\Z')

# map of default hosts for each interferometer prefix
DEFAULT_HOSTS = OrderedDict([
    (None, ('nds.ligo.caltech.edu', 31200)),
    ('H1', ('nds.ligo-wa.caltech.edu', 31200)),
    ('H0', ('nds.ligo-wa.caltech.edu', 31200)),
    ('L1', ('nds.ligo-la.caltech.edu', 31200)),
    ('L0', ('nds.ligo-la.caltech.edu', 31200)),
    ('V1', ('nds.ligo.caltech.edu', 31200)),
    ('C1', ('nds40.ligo.caltech.edu', 31200)),
    ('C0', ('nds40.ligo.caltech.edu', 31200)),
])

# -- NDS2 types ---------------------------------------------------------------

# -- correct as of nds2-client 0.16.6
# channel type
_NDS2_CHANNEL_TYPE = {
    "UNKNOWN": (0, "UNKNOWN"),
    "ONLINE": (1, "online"),
    "RAW": (2, "raw"),
    "RDS": (4, "reduced"),
    "STREND": (8, "s-trend"),
    "MTREND": (16, "m-trend"),
    "TEST_POINT": (32, "test-pt"),
    "STATIC": (64, "static"),
}
# data type
_NDS2_DATA_TYPE = {
    "UNKNOWN": (0, "UNKNOWN"),
    "INT16": (1, "int_2"),
    "INT32": (2, "int_4"),
    "INT64": (4, "int_8"),
    "FLOAT32": (8, "real_2"),
    "FLOAT64": (16, "real_4"),
    "COMPLEX32": (32, "complex_8"),
    "UINT32": (64, "uint_4"),
}

# try and override dicts with actual from nds2
try:
    import nds2
except ModuleNotFoundError:  # nds2 not found, that's ok
    pass
else:
    def _nds2_attr_dict(prefix, to_string):
        """Regenerate the relevant nds2 type dict from `nds2.channel` itself
        """
        chan = nds2.channel
        for name in filter(
            operator.methodcaller('startswith', prefix),
            chan.__dict__,
        ):
            attr = getattr(chan, name)
            yield name[len(prefix):], (attr, to_string(attr))

    _NDS2_CHANNEL_TYPE = dict(_nds2_attr_dict(
        "CHANNEL_TYPE_",
        nds2.channel.channel_type_to_string,
    ))
    _NDS2_DATA_TYPE = dict(_nds2_attr_dict(
        "DATA_TYPE_",
        nds2.channel.data_type_to_string,
    ))


# -- enums --------------------------------------------------------------------

class _Nds2Enum(enum.IntFlag):
    """Base class for NDS2 enums
    """
    def __new__(cls, value, nds2name=None):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.nds2name = nds2name  # will be None for bitwise operations
        return obj

    @classmethod
    def any(cls):
        """The logical OR of all members in this enum
        """
        return reduce(operator.or_, cls).value

    @classmethod
    def nds2names(cls):
        """The list of all recognised NDS2 names for this type
        """
        return [x.nds2name for x in cls]

    @classmethod
    def names(cls):
        """DEPRECATED: see ``.nds2names()``
        """
        warnings.warn(
            f"{cls.__name__}.names has been renamed {cls.__name__}.nds2names",
            DeprecationWarning,
        )
        return cls.nds2names()

    @classmethod
    def find(cls, name):
        """Returns the NDS2 type corresponding to the given name
        """
        name = str(name)
        # if given a number, use it
        if name.isdigit():
            return cls(int(name))
        # otherwise we might have been given a registered name
        try:
            return cls[name.upper()]
        except KeyError:
            # otherwise otherwise check the NDS2 names
            for item in cls:
                if name.lower() == item.nds2name.lower():
                    return item
        # bail out
        raise ValueError(
            f"'{name}' is not a valid {cls.__name__}",
        )


Nds2ChannelType = _Nds2Enum(
    "Nds2ChannelType",
    _NDS2_CHANNEL_TYPE,
)


class _Nds2DataType(NumpyTypeEnum, _Nds2Enum):
    pass


Nds2DataType = _Nds2DataType(
    "Nds2DataType",
    _NDS2_DATA_TYPE,
)


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
        return ",".join((
            channel.name,
            channel.channel_type_to_string(channel.channel_type),
        ))
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
    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
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
            # unknown default NDS2 host for detector, if we don't have
            # hosts already defined (either by NDSSERVER or similar)
            # we should warn the user
            if not hosts:
                warnings.warn(f"No default host found for ifo '{ifo}'")
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
    import nds2
    # pylint: disable=no-member

    # set default port for NDS1 connections (required, I think)
    if port is None and NDS1_HOSTNAME.match(host):
        port = 8088

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
    try:
        return connect(host, port)
    except RuntimeError as exc:
        if 'Request SASL authentication' not in str(exc):
            raise
    warnings.warn(f"Error authenticating against {host}:{port}", NDSWarning)
    kinit()
    return connect(host, port)


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


def reset_epoch(func):
    """Wrap a function to reset the epoch when finished

    This is useful for functions that wish to use `connection.set_epoch`.
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):  # pylint: disable=missing-docstring
        connection = kwargs.get('connection', None)
        epoch = connection.current_epoch() if connection else None
        try:
            return func(*args, **kwargs)
        finally:
            if epoch is not None:
                connection.set_epoch(epoch.gps_start, epoch.gps_stop)
    return wrapped_func


# -- query methods ------------------------------------------------------------

@open_connection
@reset_epoch
@parse_nds2_enums
def find_channels(channels, connection=None, host=None, port=None,
                  sample_rate=None, type=Nds2ChannelType.any(),
                  dtype=Nds2DataType.any(), unique=False, epoch='ALL'):
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

    epoch : `str`, `tuple` of `int`, optional
        the NDS epoch to restrict to, either the name of a known epoch,
        or a 2-tuple of GPS ``[start, stop)`` times

    Returns
    -------
    channels : `list` of `nds2.channel`
        list of NDS2 channel objects

    See also
    --------
    nds2.connection.find_channels
        for documentation on the underlying query method

    Examples
    --------
    >>> from gwpy.io.nds2 import find_channels
    >>> find_channels(['G1:DER_DATA_H'], host='nds.ligo.caltech.edu')
    [<G1:DER_DATA_H (16384Hz, RDS, FLOAT64)>]
    """
    # set epoch
    if not isinstance(epoch, tuple):
        epoch = (epoch or 'ALL',)
    connection.set_epoch(*epoch)

    # format sample_rate as tuple for find_channels call
    if isinstance(sample_rate, (int, float)):
        sample_rate = (sample_rate, sample_rate)
    elif sample_rate is None:
        sample_rate = tuple()

    # query for channels
    out = []
    for name in _get_nds2_names(channels):
        out.extend(_find_channel(connection, name, type, dtype, sample_rate,
                                 unique=unique))
    return out


def _find_channel(connection, name, ctype, dtype, sample_rate, unique=False):
    """Internal method to find a single channel

    Parameters
    ----------
    connection : `nds2.connection`, optional
        open NDS2 connection to use for query

    name : `str`
        the name of the channel to find

    ctype : `int`
        the NDS2 channel type to match

    dtype : `int`
        the NDS2 data type to match

    sample_rate : `tuple`
        a pre-formatted rate tuple (see `find_channels`)

    unique : `bool`, optional, default: `False`
        require one (and only one) match per channel

    Returns
    -------
    channels : `list` of `nds2.channel`
        list of NDS2 channel objects, if `unique=True` is given the list
        is guaranteed to have only one element.

    See also
    --------
    nds2.connection.find_channels
        for documentation on the underlying query method
    """
    # parse channel type from name,
    # e.g. 'L1:GDS-CALIB_STRAIN,reduced' -> 'L1:GDS-CALIB_STRAIN', 'reduced'
    name, ctype = _strip_ctype(name, ctype, connection.get_protocol())

    # query NDS2
    found = connection.find_channels(name, ctype, dtype, *sample_rate)

    # if don't care about defaults, just return now
    if not unique:
        return found

    # if two results, remove 'online' copy (if present)
    #    (if no online channels present, this does nothing)
    if len(found) == 2:
        found = [c for c in found if
                 c.channel_type != Nds2ChannelType.ONLINE.value]

    # if not unique result, panic
    if len(found) != 1:
        raise ValueError(f"unique NDS2 channel match not found for '{name}'")

    return found


def _strip_ctype(name, ctype, protocol=2):
    """Strip the ctype from a channel name for the given nds server version

    This is needed because NDS1 servers store trend channels _including_
    the suffix, but not raw channels, and NDS2 doesn't do this.
    """
    # parse channel type from name (e.g. 'L1:GDS-CALIB_STRAIN,reduced')
    try:
        name, ctypestr = name.rsplit(',', 1)
    except ValueError:
        pass
    else:
        ctype = Nds2ChannelType.find(ctypestr).value
        # NDS1 stores channels with trend suffix, so we put it back:
        if protocol == 1 and ctype in (
                Nds2ChannelType.STREND.value,
                Nds2ChannelType.MTREND.value
        ):
            name += f',{ctypestr}'
    return name, ctype


@open_connection
@reset_epoch
def get_availability(channels, start, end,
                     connection=None, host=None, port=None):
    # pylint: disable=unused-argument
    """Query an NDS2 server for data availability

    Parameters
    ----------
    channels : `list` of `str`
        list of channel names to query; this list is mapped to NDS channel
        names using :func:`find_channels`.

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

    Raises
    ------
    ValueError
        if the given channel name cannot be mapped uniquely to a name
        in the NDS server database.

    See also
    --------
    nds2.connection.get_availability
        for documentation on the underlying query method
    """
    from ..segments import (Segment, SegmentList, SegmentListDict)
    connection.set_epoch(start, end)
    # map user-given real names to NDS names
    names = list(map(
        _get_nds2_name, find_channels(channels, epoch=(start, end),
                                      connection=connection, unique=True),
    ))
    # query for availability
    result = connection.get_availability(names)
    # map to segment types
    out = SegmentListDict()
    for name, result in zip(channels, result):
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
