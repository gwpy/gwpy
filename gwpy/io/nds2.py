# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

from __future__ import annotations

import enum
import operator
import os
import re
import typing
import warnings
from functools import (
    reduce,
    wraps,
)

from ..time import to_gps
from ..utils.enum import NumpyTypeEnum
from .kerberos import kinit

if typing.TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
    )

    import nds2

    from ..detector import Channel
    from ..segments import SegmentListDict
    from ..typing import (
        GpsLike,
        Self,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# regular expression to match LIGO-standard NDS1 hostnames (e.g x1nds0)
NDS1_HOSTNAME = re.compile(r"[a-z]1nds[0-9]\Z")

# map of default hosts for each interferometer prefix
DEFAULT_HOSTS = {
    None: ("nds.ligo.caltech.edu", None),
    "H1": ("nds.ligo-wa.caltech.edu", None),
    "H0": ("nds.ligo-wa.caltech.edu", None),
    "L1": ("nds.ligo-la.caltech.edu", None),
    "L0": ("nds.ligo-la.caltech.edu", None),
    "V1": ("nds.ligo.caltech.edu", None),
    "C1": ("nds40.ligo.caltech.edu", None),
    "C0": ("nds40.ligo.caltech.edu", None),
}
GWOSC_NDS2_HOSTS = [
    ("nds.gwosc.org", None),
]

# minimum and maximum acceptable sample rates
MIN_SAMPLE_RATE = 0.
MAX_SAMPLE_RATE = 1e12

# -- NDS2 types ----------------------

def _nds2_attr_dict(prefix, to_string):
    """Regenerate the relevant nds2 type dict from `nds2.channel` itself."""
    chan = nds2.channel
    for name in filter(
        operator.methodcaller("startswith", prefix),
        chan.__dict__,
    ):
        attr = getattr(chan, name)
        yield name[len(prefix):], (attr, to_string(attr))


_NDS2_CHANNEL_TYPE: dict[str, tuple[int, str]]
_NDS2_DATA_TYPE: dict[str, tuple[int, str]]
try:
    import nds2
except ModuleNotFoundError:
    # nds2 not found, that's ok, use a static copy
    #     correct as of nds2-client 0.16.16
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
else:
    MIN_SAMPLE_RATE = nds2.channel.MIN_SAMPLE_RATE
    MAX_SAMPLE_RATE = nds2.channel.MAX_SAMPLE_RATE

    # channel type
    _NDS2_CHANNEL_TYPE = dict(_nds2_attr_dict(
        "CHANNEL_TYPE_",
        nds2.channel.channel_type_to_string,
    ))
    # data type
    _NDS2_DATA_TYPE = dict(_nds2_attr_dict(
        "DATA_TYPE_",
        nds2.channel.data_type_to_string,
    ))


# -- enums ---------------------------

class _Nds2Enum(enum.IntFlag):
    """Base class for NDS2 enums."""
    def __new__(
        cls,
        value: int,
        nds2name: str,
    ):
        obj = int.__new__(cls, value)
        obj._value_ = value
        obj.nds2name = nds2name  # type: ignore[attr-defined]
        return obj

    @classmethod
    def any(cls) -> Self:
        """The logical OR of all members in this enum."""
        return reduce(operator.or_, cls)

    @classmethod
    def nds2names(cls) -> list[str]:
        """The list of all recognised NDS2 names for this type."""
        return [x.nds2name for x in cls]

    @classmethod
    def find(cls, name: str | int) -> Self:
        """Returns the NDS2 type corresponding to the given name."""
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
        # return unknown
        return cls["UNKNOWN"]


Nds2ChannelType = _Nds2Enum(
    "Nds2ChannelType",  # type: ignore[arg-type]
    _NDS2_CHANNEL_TYPE,  # type: ignore[arg-type]
)
Nds2ChannelType.__doc__ = "NDS2 channel type with descriptive name."


class _Nds2DataType(NumpyTypeEnum, _Nds2Enum):
    """NDS2 data type with descriptive name."""
    @classmethod
    def find(cls, name):
        try:
            return super().find(name)
        except ValueError:
            # return unknown
            return cls(0)


Nds2DataType = _Nds2DataType(
    "Nds2DataType",  # type: ignore[arg-type]
    _NDS2_DATA_TYPE,  # type: ignore[arg-type]
)


# -- warning suppression -------------

class NDSWarning(UserWarning):
    """Warning about communicating with the Network Data Server."""
    pass


# always display these warnings
warnings.simplefilter("always", NDSWarning)


# -- query utilities -----------------

def _get_nds2_name(
    channel: str | Channel | nds2.channel,
) -> str:
    """Returns the NDS2-formatted name for a channel.

    Understands how to format NDS name strings from
    `gwpy.detector.Channel` and `nds2.channel` objects
    """
    from ..detector import Channel

    if isinstance(channel, Channel):
        return channel.ndsname

    import nds2

    if isinstance(channel, nds2.channel):
        return ",".join((
            channel.name,
            channel.channel_type_to_string(channel.channel_type),
        ))
    return str(channel)


def _get_nds2_names(
    channels: Iterable[str | Channel | nds2.channel],
) -> Iterator[str]:
    """Maps `_get_nds2_name` for a list of input channels."""
    return map(_get_nds2_name, channels)


# -- connection utilities ------------

def parse_nds_env(
    env: str = "NDSSERVER",
) -> list[tuple[str, int | None]]:
    """Parse the NDSSERVER environment variable into a list of hosts.

    Parameters
    ----------
    env : `str`, optional
        Shell environment variable name to use for server order.
        Default is ``"NDSSERVER"``.
        The contents of this variable should be a comma-separated
        list of ``host:port`` strings, e.g.
        ``"nds1.server.com:80,nds2.server.com:80"``.

    Returns
    -------
    hostiter : `list` of `tuple`
        a list of (unique) ``(str, int)`` tuples for each host:port
        pair
    """
    hosts = []
    for host in os.environ[env].split(","):
        try:
            host, portstr = host.rsplit(":", 1)
        except ValueError:
            port = None
        else:
            port = int(portstr)
        if (host, port) not in hosts:
            hosts.append((host, port))
    return hosts


def host_resolution_order(
    ifo: str,
    env: str | None = "NDSSERVER",
    epoch: GpsLike = "now",
    lookback: float = 14*86400,
    include_gwosc : bool = True,
) -> list[tuple[str, int | None]]:
    """Generate a logical ordering of NDS (host, port) tuples for this IFO.

    Parameters
    ----------
    ifo : `str`
        Prefix for IFO of interest.

    env : `str`, optional
        Shell environment variable name to use for server order.
        Default is ``"NDSSERVER"``.
        The contents of this variable should be a comma-separated
        list of ``host:port`` strings, e.g.
        ``"nds1.server.com:80,nds2.server.com:80"``.
        Pass ``env=None`` to disable parsing any environment variable.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch of data requested.

    lookback : `float`, optional
        Duration of spinning-disk cache. This value triggers defaulting to
        the CIT NDS2 server over those at the LIGO sites.
        Default is two weeks.

    include_gwosc : `bool`, optional
        If `True` include the |GWOSC|_ NDS2 instance in the list.

    Returns
    -------
    hro : `list` of `2-tuples <tuple>`
        Drdered `list` of ``(host, port)`` tuples.

    Examples
    --------
    With no environment settings:

    >>> host_resolution_order('H1')
    [('nds.ligo-wa.caltech.edu', None),
     ('nds.ligo.caltech.edu', None),
     ('nds.gwosc.org', None),
    ]
    >>> host_resolution_order('H1', include_gwosc=False))
    [('nds.ligo-wa.caltech.edu', None), ('nds.ligo.caltech.edu', None)]
    """
    hosts = []

    # if given environment variable exists, it will contain a
    # comma-separated list of host:port strings giving the logical ordering
    if env and os.getenv(env):
        hosts = parse_nds_env(env)

    # add our ordered list based on the IFO and the lookback time
    # (at LIGO, the LIGO-Caltech data centre has more data on disk than
    #  the observatory data centres do, so prefer Caltech for older data)
    if to_gps("now") - to_gps(epoch) > lookback:
        ifolist = [None, ifo]
    else:
        ifolist = [ifo, None]

    for difo in ifolist:
        try:
            ifodefault = DEFAULT_HOSTS[difo]
        except KeyError:
            # unknown default NDS2 host for detector, if we don't have
            # hosts already defined (either by NDSSERVER or similar)
            # we should warn the user
            if not hosts:
                warnings.warn(
                    f"no default host found for ifo '{ifo}'",
                )
        else:
            if ifodefault not in hosts:
                hosts.append(ifodefault)

    # append the GWOSC NDS2 instance(s) to the list, just in case.
    if include_gwosc:
        hosts.extend(GWOSC_NDS2_HOSTS)

    return hosts


def connect(
    host: str,
    port: int | None = None,
) -> nds2.connection:
    """Open an `nds2.connection` to a given host and port.

    Parameters
    ----------
    host : `str`
        Name of server with which to connect.

    port : `int`, optional
        Connection port.

    Returns
    -------
    connection : `nds2.connection`
        A new open connection to the given NDS host.
    """
    import nds2

    # set default port for NDS1 connections (required, I think)
    if port is None and NDS1_HOSTNAME.match(host):
        port = 8088

    if port is None:
        return nds2.connection(host)
    return nds2.connection(host, port)


def auth_connect(
    host: str,
    port: int | None = None,
) -> nds2.connection:
    """Open an `nds2.connection` handling simple authentication errors.

    This method will catch exceptions related to kerberos authentication,
    and execute a kinit() for the user before attempting to connect again.

    Parameters
    ----------
    host : `str`
        Name of server with which to connect.

    port : `int`, optional
        Connection port.

    Returns
    -------
    connection : `nds2.connection`
        A new open connection to the given NDS host.

    See also
    --------
    gwpy.io.nds2.connect
        For details of opening connections.
    """
    try:
        return connect(host, port)
    except RuntimeError as exc:
        if "Request SASL authentication" not in str(exc):
            raise
    warnings.warn(
        f"error authenticating against {host}:{port}, "
        "attempting Kerberos kinit()",
        NDSWarning,
    )
    kinit()
    return connect(host, port)


def open_connection(func: Callable) -> Callable:
    """Decorate a function to create a `nds2.connection` if required."""
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if kwargs.get("connection", None) is None:
            try:
                host = kwargs.pop("host")
            except KeyError:
                raise TypeError(
                    "one of `connection` or `host` is required "
                    "to query NDS2 server",
                )
            kwargs["connection"] = auth_connect(
                host,
                kwargs.pop("port", None),
            )
        return func(*args, **kwargs)
    return wrapped_func


def parse_nds2_enums(func: Callable) -> Callable:
    """Decorate a function to translate a type string into an integer."""
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        for kwd, enum_ in (
            ("type", Nds2ChannelType),
            ("dtype", Nds2DataType),
        ):
            if kwargs.get(kwd, None) is None:
                kwargs[kwd] = enum_.any()
            elif not isinstance(kwargs[kwd], int):
                kwargs[kwd] = enum_.find(kwargs[kwd]).value
        return func(*args, **kwargs)
    return wrapped_func


def reset_epoch(func: Callable) -> Callable:
    """Wrap a function to reset the epoch when finished.

    This is useful for functions that wish to use `connection.set_epoch`.
    """
    @wraps(func)
    def wrapped_func(*args, **kwargs):
        if (connection := kwargs.get("connection", None)) is not None:
            epoch = connection.current_epoch()
        else:
            epoch = None
        try:
            return func(*args, **kwargs)
        finally:
            if epoch is not None:
                connection.set_epoch(epoch.gps_start, epoch.gps_stop)
    return wrapped_func


# -- query methods -------------------

@open_connection
@reset_epoch
@parse_nds2_enums
def find_channels(
    channels: list[str],
    connection: nds2.connection | None = None,
    host: str | None = None,
    port: int | None = None,
    sample_rate: float | tuple[float, float] | None = None,
    type: int = Nds2ChannelType.any(),
    dtype: int = Nds2DataType.any(),
    unique: bool = False,
    epoch: str | tuple[int, int] = "ALL",
) -> list[nds2.channel]:
    """Query an NDS2 server for channel information.

    Parameters
    ----------
    channels : `list` of `str`
        List of channel names to query, each can include bash-style globs.

    connection : `nds2.connection`, optional
        Open NDS2 connection to use for query.

    host : `str`, optional
        Name of NDS2 server to query, required if ``connection`` is not
        given.

    port : `int`, optional
        Port number on host to use for NDS2 connection.

    sample_rate : `int`, `float`, `tuple`, optional
        A single number, representing a specific sample rate to match,
        or a tuple representing a ``(low, high)` interval to match.

    type : `int`, optional
        The NDS2 channel type to match.

    dtype : `int`, optional
        The NDS2 data type to match.

    unique : `bool`, optional, default: `False`
        Require one (and only one) match per channel.

    epoch : `str`, `tuple` of `int`, optional
        The NDS epoch to restrict to, either the name of a known epoch,
        or a 2-tuple of GPS ``[start, stop)`` times.

    Returns
    -------
    channels : `list` of `nds2.channel`
        List of NDS2 channel objects.

    See also
    --------
    nds2.connection.find_channels
        For documentation on the underlying query method.

    Examples
    --------
    >>> from gwpy.io.nds2 import find_channels
    >>> find_channels(['G1:DER_DATA_H'], host='nds.ligo.caltech.edu')
    [<G1:DER_DATA_H (16384Hz, RDS, FLOAT64)>]
    """
    # set epoch
    if isinstance(epoch, tuple):
        connection.set_epoch(*epoch)
    else:
        connection.set_epoch(epoch or "ALL")

    # format {min,max}_sample_rate options
    kwargs = {}
    if isinstance(sample_rate, (int, float)):
        kwargs["min_sample_rate"] = sample_rate
        kwargs["max_sample_rate"] = sample_rate
    elif sample_rate is not None:
        kwargs["min_sample_rate"], kwargs["max_sample_rate"] = sample_rate

    # query for channels
    out = []
    for name in _get_nds2_names(channels):
        out.extend(_find_channel(
            connection,
            channel_glob=name,
            channel_type_mask=type,
            data_type_mask=dtype,
            unique=unique,
            **kwargs,
        ))
    return out


def _find_channel(
    connection: nds2.connection,
    channel_glob: str = "*",
    channel_type_mask: int = Nds2ChannelType.any().value,
    data_type_mask: int = Nds2DataType.any().value,
    min_sample_rate: float = MIN_SAMPLE_RATE,
    max_sample_rate: float = MAX_SAMPLE_RATE,
    unique: bool = False,
) -> list[nds2.channel]:
    """Internal method to find a single channel.

    Parameters
    ----------
    connection : `nds2.connection`, optional
        Open NDS2 connection to use for query

    channel_glob : `str`, optional
        The name of the channel to find.

    channel_type_mask : `int`, optional
        The NDS2 channel type to match.

    data_type_mask : `int`, optional
        The NDS2 data type to match.

    min_sample_rate : `float`, optional
        The lowest sample rate to match.

    max_sample_rate : `float`, optional
        The highest sample rate to match.

    unique : `bool`, optional
        Require one (and only one) match per channel.

    Returns
    -------
    channels : `list` of `nds2.channel`
        List of NDS2 channel objects, if `unique=True` is given the list
        is guaranteed to have only one element.

    Raises
    ------
    ValueError
        If ``unique=True`` and more than one channel is found.

    See also
    --------
    nds2.connection.find_channels
        For documentation of all arguments and on the underlying query method.
    """
    # parse channel type from name,
    # e.g. 'L1:GDS-CALIB_STRAIN,reduced' -> 'L1:GDS-CALIB_STRAIN', 'reduced'
    channel_glob, channel_type_mask = _strip_ctype(
        channel_glob,
        channel_type_mask,
        connection.get_protocol(),
    )

    # query NDS2
    found = connection.find_channels(
        channel_glob,
        channel_type_mask,
        data_type_mask,
        min_sample_rate,
        max_sample_rate,
    )

    # if don't care about defaults, just return now
    if not unique:
        return found

    # if two results, remove 'online' copy (if present)
    #    (if no online channels present, this does nothing)
    if len(found) == 2:
        online = Nds2ChannelType.ONLINE.value  # type: ignore[attr-defined]
        found = [c for c in found if c.channel_type != online]

    # if not unique result, panic
    if len(found) != 1:
        raise ValueError(
            f"unique NDS2 channel match not found for '{channel_glob}'",
        )

    return found


def _strip_ctype(
    name: str,
    ctype: int,
    protocol: int = 2,
) -> tuple[str, int]:
    """Strip the ctype from a channel name for the given nds server version.

    This is needed because NDS1 servers store trend channels _including_
    the suffix, but not raw channels, and NDS2 doesn't do this.
    """
    # parse channel type from name (e.g. 'L1:GDS-CALIB_STRAIN,reduced')
    try:
        name, ctypestr = name.rsplit(",", 1)
    except ValueError:
        return name, ctype

    ctype = Nds2ChannelType.find(ctypestr).value

    # NDS1 stores channels with trend suffix, so we put it back:
    if protocol == 1 and ctype in (
        Nds2ChannelType.STREND.value,  # type: ignore[attr-defined]
        Nds2ChannelType.MTREND.value  # type: ignore[attr-defined]
    ):
        name += f",{ctypestr}"

    return name, ctype


@open_connection
@reset_epoch
def get_availability(
    channels: list[str],
    start: int,
    end: int,
    connection: nds2.connection | None = None,
    **kwargs,
) -> SegmentListDict:
    """Query an NDS2 server for data availability.

    Parameters
    ----------
    channels : `list` of `str`
        List of channel names to query; this list is mapped to NDS channel
        names using :func:`find_channels`..

    start : `int`
        GPS start time of query.

    end : `int`
        GPS end time of query.

    connection : `nds2.connection`, optional
        Open NDS2 connection to use for query.

    host : `str`, optional
        Name of NDS2 server to query, required if ``connection`` is not
        given.

    port : `int`, optional
        Port number on host to use for NDS2 connection.

    Returns
    -------
    segdict : `~gwpy.segments.SegmentListDict`
        Dict of ``(name, SegmentList)`` pairs.

    Raises
    ------
    ValueError
        If the given channel name cannot be mapped uniquely to a name
        in the NDS server database.

    See also
    --------
    nds2.connection.get_availability
        For documentation on the underlying query method.
    """
    from ..segments import (
        Segment,
        SegmentList,
        SegmentListDict,
    )

    connection.set_epoch(start, end)

    # map user-given real names to NDS names
    names = list(map(_get_nds2_name, find_channels(
        channels,
        epoch=(start, end),
        connection=connection,
        unique=True,
    )))

    # query for availability
    result = connection.get_availability(names)

    # map to segment types
    out = SegmentListDict()
    for name, result in zip(channels, result):
        out[name] = SegmentList([
            Segment(s.gps_start, s.gps_stop)
            for s in result.simple_list()
        ])
    return out


def minute_trend_times(start: int, end: int) -> tuple[int, int]:
    """Expand a [start, end) interval for use in querying for minute trends.

    NDS2 requires start and end times for minute trends to be a multiple of
    60 (to exactly match the time of a minute-trend sample), so this function
    expands the given ``[start, end)`` interval to the nearest multiples.

    Parameters
    ----------
    start : `int`
        GPS start time of query.

    end : `int`
        GPS end time of query.

    Returns
    -------
    mstart : `int`
        ``start`` rounded down to nearest multiple of 60.
    mend : `int`
        ``end`` rounded up to nearest multiple of 60.

    Examples
    --------
    >>> minute_trend_times(123, 456)
    (120, 480)
    """
    if start % 60:
        start = int(start) // 60 * 60
    if end % 60:
        end = int(end) // 60 * 60 + 60
    return int(start), int(end)
