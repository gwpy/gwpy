# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""The `Channel` and `ChannelList`."""

from __future__ import annotations

import re
from copy import copy
from math import ceil
from typing import (
    TYPE_CHECKING,
    overload,
)

import numpy
from astropy import units
from urllib3.util import parse_url

from ..io import nds2 as io_nds2
from ..io.registry import UnifiedReadWriteMethod
from ..time import to_gps
from .connect import (
    ChannelListRead,
    ChannelListWrite,
)
from .units import parse_unit

if TYPE_CHECKING:
    import builtins
    from typing import (
        Literal,
        Self,
        SupportsFloat,
    )

    import arrakis
    import nds2
    from astropy.units import (
        Quantity,
        UnitBase,
    )

    from ..segments import SegmentListDict
    from ..time import SupportsToGps
    from ..typing import UnitLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# NOTE: when we depend on numpy>=2.0, numpy.bool_ here can
#       be replaced with numpy.bool.
BOOL_TYPES = {bool, numpy.bool_, numpy.dtype(bool), "bool"}

QUOTE_REGEX = re.compile(r'^[\s\"\']+|[\s\"\']+$')


class Channel:
    """Representation of a gravitational-wave detectory data channel.

    Parameters
    ----------
    name : `str`, `Channel`
        Name of this Channel (or another  Channel itself).
        If a `Channel` is given, all other parameters not set explicitly
        will be copied over.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        Rate at which data are sampled for this channel, simple floats
        must be given in Hz.

    unit : `~astropy.units.Unit`, `str`, optional
        Name of the unit for the data of this channel.

    frequency_range : `tuple` of `float`
        ``[low, high)`` spectral frequency range of interest for this channel.

    safe : `bool`, optional
        If `True` this channel is 'safe' to use as a witness of
        non-gravitational-wave noise in the detector.

    dtype : `numpy.dtype`, optional
        Numeric type of data for this channel.

    frametype : `str`, optional
        LDAS name for frames that contain this channel.

    model : `str`, optional
        Name of the SIMULINK front-end model that produces this channel.

    Notes
    -----
    The `Channel` structure implemented here is designed to match the
    data recorded in the LIGO Channel Information System
    (https://cis.ligo.org) for which a query interface is provided.
    """

    MATCH = re.compile(
        r"((?:(?P<ifo>[A-Z]\d))?|[\w-]+):"  # match IFO prefix
        r"(?:(?P<system>[a-zA-Z0-9]+))?"  # match system
        r"(?:[-_](?P<subsystem>[a-zA-Z0-9]+))?"  # match subsystem
        r"(?:[-_](?P<signal>[a-zA-Z0-9_-]+?))?"  # match signal
        r"(?:[\.-](?P<trend>[a-z]+))?"  # match trend type
        r"(?:,(?P<type>([a-z]-)?[a-z]+))?$",  # match channel type
    )

    def __init__(
        self,
        name: str | Self,
        **params,
    ) -> None:
        """Create a new `Channel`.

        Parameters
        ----------
        name : `str`, `Channel`
            The name of the new channel, or an existing channel to copy.

        params
            ``(key, value)`` pairs for attributes of new channel.
        """
        # init properties
        self._name: str  # set in _init_name()
        self._ifo: str | None = None
        self._system: str | None = None
        self._subsystem: str | None = None
        self._signal: str | None = None
        self._trend: str | None = None
        self._sample_rate: Quantity | None = None
        self._unit: UnitBase | None = None
        self._frequency_range: Quantity | None = None
        self._safe: bool | None = None
        self._type: str | None = None
        self._dtype: numpy.dtype | None = None
        self._frametype: str | None = None
        self._model: str | None = None
        self._url: str | None = None

        # copy existing Channel
        if isinstance(name, Channel):
            self._init_from_channel(name)

        # parse name into component parts
        else:
            self._init_name(name)

        # set metadata
        for key, value in params.items():
            setattr(self, key, value)

    def _init_from_channel(self, other: Self) -> None:
        """Copy attributes from ``other`` to this channel."""
        for key, value in vars(other).items():
            setattr(self, key, copy(value))

    def _init_name(self, name: str) -> None:
        """Initialise the name of this `Channel`."""
        try:
            parts = self.parse_channel_name(name)
        except (TypeError, ValueError):  # failed to parse
            self.name = str(name)
        else:
            self.name = str(name).split(",")[0]
            for key, val in parts.items():
                try:
                    setattr(self, key, val)
                except AttributeError:
                    setattr(self, f"_{key}", val)

    # -- properties -----------------------------

    @property
    def name(self) -> str:
        """Name of this channel.

        This should follow the naming convention, with the following
        format: 'IFO:SYSTEM-SUBSYSTEM_SIGNAL'
        """
        return self._name

    @name.setter
    def name(self, name: str) -> None:
        self._name = str(name)

    @property
    def sample_rate(self) -> Quantity | None:
        """Rate of samples (Hertz) for this channel."""
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, rate: SupportsFloat | Quantity | None) -> None:
        if rate is None:
            self._sample_rate = None
        elif isinstance(rate, units.Quantity):
            self._sample_rate = rate
        else:
            # pylint: disable=no-member
            self._sample_rate = units.Quantity(float(rate), unit=units.Hertz)

    @sample_rate.deleter
    def sample_rate(self) -> None:
        self._sample_rate = None

    @property
    def unit(self) -> UnitBase | None:
        """Data unit for this channel."""
        return self._unit

    @unit.setter
    def unit(self, u: UnitLike) -> None:
        if u is None:
            self._unit = None
        else:
            self._unit = parse_unit(u)

    @unit.deleter
    def unit(self) -> None:
        self._unit = None

    @property
    def frequency_range(self) -> Quantity | None:
        """Frequency range of interest (Hertz) for this channel."""
        return self._frequency_range

    @frequency_range.setter
    def frequency_range(self, frange: tuple[float, float] | None) -> None:
        if frange is None:
            self._frequency_range = None
        else:
            low, hi = frange
            self._frequency_range = units.Quantity((low, hi), unit="Hz")

    @frequency_range.deleter
    def frequency_range(self) -> None:
        self._frequency_range = None

    @property
    def safe(self) -> bool | None:
        """Whether this channel is 'safe' to use as a noise witness.

        Any channel that records part or all of a GW signal as it
        interacts with the interferometer is not safe to use as a noise
        witness

        A safe value of `None` simply indicates that the safety of this
        channel has not been determined
        """
        return self._safe

    @safe.setter
    def safe(self, s: bool | None) -> None:
        if s is None:
            self._safe = None
        else:
            self._safe = bool(s)

    @safe.deleter
    def safe(self) -> None:
        self._safe = None

    @property
    def model(self) -> str | None:
        """Name of the SIMULINK front-end model that defines this channel."""
        return self._model

    @model.setter
    def model(self, mdl: str | None) -> None:
        if mdl is None:
            self._model = None
        else:
            self._model = mdl.lower()

    @model.deleter
    def model(self) -> None:
        self._model = None

    @property
    def type(self) -> str | None:
        """DAQ data type for this channel.

        Valid values for this field are restricted to those understood
        by the NDS2 client sofware, namely:

        'm-trend', 'online', 'raw', 'reduced', 's-trend', 'static', 'test-pt'
        """
        return self._type

    @type.setter
    def type(self, type_: str | int | None) -> None:
        if type_ is None:
            self._type = None
        else:
            self._type = io_nds2.Nds2ChannelType.find(type_).nds2name

    @type.deleter
    def type(self) -> None:
        self._type = None

    @property
    def ndstype(self) -> int:
        """NDS type integer for this channel.

        This property is mapped to the `Channel.type` string.
        """
        if self.type is None:
            return 0  # UNKNOWN
        return io_nds2.Nds2ChannelType.find(self.type).value

    @ndstype.setter
    def ndstype(self, type_: str | int | None) -> None:
        self.type = type_

    @ndstype.deleter
    def ndstype(self) -> None:
        del self.type

    @property
    def dtype(self) -> numpy.dtype | None:
        """Numeric type for data in this channel."""
        return self._dtype

    @dtype.setter
    def dtype(self, type_: str | int | builtins.type | None) -> None:
        if type_ is None:
            self._dtype = None
        elif type_ in BOOL_TYPES:  # NDS2 doesn't support bool
            self._dtype = numpy.dtype(bool)
        else:
            self._dtype = io_nds2.Nds2DataType.find(type_).dtype

    @dtype.deleter
    def dtype(self) -> None:
        self._dtype = None

    @property
    def url(self) -> str | None:
        """CIS browser url for this channel."""
        return self._url

    @url.setter
    def url(self, href: str | None) -> None:
        if href is None:
            self._url = None
        else:
            url = parse_url(href)
            if url.scheme not in ("http", "https", "file"):
                msg = f"Invalid URL '{href}'"
                raise ValueError(msg)
            self._url = href

    @url.deleter
    def url(self) -> None:
        self._url = None

    @property
    def frametype(self) -> str | None:
        """LDAS type description for frame files containing this channel."""
        return self._frametype

    @frametype.setter
    def frametype(self, ft: str | None) -> None:
        self._frametype = ft

    @frametype.deleter
    def frametype(self) -> None:
        self._frametype = None

    # -- read-only properties -------------------

    @property
    def ifo(self) -> str | None:
        """Interferometer prefix for this channel."""
        return self._ifo

    @property
    def system(self) -> str | None:
        """Instrumental system for this channel."""
        return self._system

    @property
    def subsystem(self) -> str | None:
        """Instrumental sub-system for this channel."""
        return self._subsystem

    @property
    def signal(self) -> str | None:
        """Instrumental signal for this channel."""
        return self._signal

    @property
    def trend(self) -> str | None:
        """Trend type for this channel."""
        return self._trend

    @property
    def texname(self) -> str:
        """Name of this channel in LaTeX printable format."""
        return str(self).replace("_", r"\_")

    @property
    def ndsname(self) -> str:
        """Name of this channel as stored in the NDS database."""
        if self.type not in {None, "raw", "reduced", "online"}:
            return f"{self.name},{self.type}"
        return self.name

    # -- classmethods ---------------------------

    @classmethod
    def query(
        cls,
        name: str,
        *,
        kerberos: bool | None = None,
        **kwargs,
    ) -> Self:
        """Query the LIGO Channel Information System for ``name``.

        Parameters
        ----------
        name : `str`
            Name of channel for which to query.

        kerberos : `bool`, optional
            If `True` use an existing Kerberos ticket as the authentication credential.
            Default (`None`) is to check for credentials and request username
            and password if none are found.

        kwargs
            Other keyword arguments are passed directly to
            :func:`ciecplib.get`.

        Returns
        -------
        channel : `Channel`
            A new `Channel` containing all of the attributes set from
            its entry in the CIS.
        """
        channellist = ChannelList.query(name, kerberos=kerberos, **kwargs)
        if not channellist:
            msg = f"No channels found matching '{name}'"
            raise ValueError(msg)
        if len(channellist) > 1:
            msg = (
                f"{len(channellist)} channels found matching '{name}', "
                "please refine search, or use `ChannelList.query` to "
                "return all results"
            )
            raise ValueError(msg)
        return channellist[0]

    @classmethod
    def query_nds2(
        cls,
        name: str,
        host: str | None = None,
        port: int | None = None,
        connection: nds2.connection | None = None,
        type: str | int | None = None,  # noqa: A002
    ) -> Self:
        """Query an NDS server for channel information.

        Parameters
        ----------
        name : `str`
            Name of channel for which to query.

        host : `str`, optional
            Name of NDS2 server.

        port : `int`, optional
            Port number for NDS2 connection.

        connection : `nds2.connection`
            Open connection to use for query.

        type : `str`, `int`
            NDS2 channel type with which to restrict query

        Returns
        -------
        channel : `Channel`
            Channel with metadata retrieved from NDS2 server.

        Raises
        ------
        ValueError
            If multiple channels are found for a given name.

        Notes
        -----
        .. warning::

            One of ``host`` or ``connection`` is required.
        """
        return ChannelList.query_nds2(
            [name],
            host=host,
            port=port,
            connection=connection,
            type=type,
            unique=True,
        )[0]

    @classmethod
    def from_nds2(
        cls,
        nds2channel: nds2.channel,
    ) -> Self:
        """Generate a new channel using an existing nds2.channel object."""
        return cls(
            nds2channel.name,
            sample_rate=nds2channel.sample_rate,
            unit=nds2channel.signal_units or None,
            dtype=nds2channel.data_type,
            type=nds2channel.channel_type,
        )

    @classmethod
    def from_arrakis(
        cls,
        arrakischannel: arrakis.Channel,
    ) -> Self:
        """Generate a new channel using an existing `arrakis.Channel`.

        Parameters
        ----------
        arrakischannel : `arrakis.Channel`
            The input channel from Arrakis to parse.

        Returns
        -------
        channel : `gwpy.detector.Channel`
            A new `Channel`.
        """
        return cls(
            arrakischannel.name,
            sample_rate=arrakischannel.sample_rate,
            dtype=arrakischannel.data_type,
        )

    # -- methods --------------------------------

    @classmethod
    def parse_channel_name(
        cls,
        name: str,
        *,
        strict: bool = True,
    ) -> dict[str, str | None]:
        """Decompose a channel name string into its components.

        This method parses channels acccording to the LIGO Channel Naming
        Convention :dcc:`LIGO-T990033`.

        Parameters
        ----------
        name : `str`
            Name to parse.

        strict : `bool`, optional
            If `True` (default) require exact matching of format,
            with no surrounding text.

        Returns
        -------
        match : `dict`
            `dict` of channel name components with the following keys:

            - `'ifo'`: the letter-number interferometer prefix
            - `'system'`: the top-level system name
            - `'subsystem'`: the second-level sub-system name
            - `'signal'`: the remaining underscore-delimited signal name
            - `'trend'`: the trend type
            - `'ndstype'`: the NDS2 channel suffix

            Any optional keys that aren't found will return a value of `None`

        Raises
        ------
        ValueError
            if the name cannot be parsed with at least an IFO and SYSTEM

        Examples
        --------
        >>> Channel.parse_channel_name("L1:LSC-DARM_IN1_DQ")
        {'ifo': 'L1',
         'ndstype': None,
         'signal': 'IN1_DQ',
         'subsystem': 'DARM',
         'system': 'LSC',
         'trend': None}

        >>> Channel.parse_channel_name(
        ...     "H1:ISI-BS_ST1_SENSCOR_GND_STS_X_BLRMS_100M_300M.rms,m-trend",
        ... )
        {'ifo': 'H1',
         'ndstype': 'm-trend',
         'signal': 'ST1_SENSCOR_GND_STS_X_BLRMS_100M_300M',
         'subsystem': 'BS',
         'system': 'ISI',
         'trend': 'rms'}
        """
        match = cls.MATCH.search(name)
        if match is None or (
            strict
            and (match.start() != 0 or match.end() != len(name))
        ):
            msg = "Cannot parse channel name according to LIGO-T990033"
            raise ValueError(msg)
        return match.groupdict()

    @overload
    def find_frametype(
        self,
        gpstime: SupportsToGps | None = None,
        *,
        frametype_match: str | re.Pattern | None = None,
        return_all: Literal[False] = False,
        allow_tape: bool = True,
        **kwargs,
    ) -> str: ...

    @overload
    def find_frametype(
        self,
        gpstime: SupportsToGps | None = None,
        *,
        frametype_match: str | re.Pattern | None = None,
        return_all: Literal[True] = True,
        allow_tape: bool = True,
        **kwargs,
    ) -> list[str]: ...

    def find_frametype(
        self,
        gpstime: SupportsToGps | None = None,
        *,
        frametype_match: str | re.Pattern | None = None,
        return_all: bool = False,
        allow_tape: bool = True,
        **kwargs,
    ) -> str | list[str]:
        """Find the containing frametype(s) for this `Channel`.

        Parameters
        ----------
        gpstime : `int`, optional
            A reference GPS time at which to search for frame files.
            Default is to search in the latest available data for each
            discoverable dataset.

        frametype_match : `str`
            A regular expression string to use to down-select from the
            list of all available datasets.

        return_all: `bool`, optional
            If `True` return all matched datasets; if `False` (default)
            only the first match is returned

        allow_tape : `bool`, default: `True`
            If `True` (default) include datasets whose files are stored on slow
            magnetic tape.

        kwargs
            Other keyword arguments are passed directly to
            :func:`gwpy.io.datafind.find_frametype`.

        Returns
        -------
        frametype : `str` or `list[str]`
            If ``return_all=False`` a single `str` containing the 'best' dataset name.
            If ``return_all=True`` a `list` of dataset names.
        """
        from ..io import datafind

        return datafind.find_frametype(
            self,
            gpstime=gpstime,
            frametype_match=frametype_match,
            return_all=return_all,
            allow_tape=allow_tape,
            **kwargs,
        )

    def copy(self) -> Self:
        """Return a copy of this `Channel`."""
        return type(self)(self)

    def __str__(self) -> str:
        """Return the name of this `Channel`."""
        return self.name or ""

    def __repr__(self) -> str:
        """Return a printable representation of this `Channel`."""
        repr_ = f'<Channel("{self}"'
        if self.type:
            repr_ += f" [{self.type}]"
        repr_ += f", {self.sample_rate}"
        return repr_ + f") at {hex(id(self))}>"

    def __eq__(self, other: object) -> bool:
        """Return `True` if all attributes of this channel match ``other``."""
        try:
            for attr in ("name", "sample_rate", "unit", "url", "type", "dtype"):
                if getattr(self, attr) != getattr(other, attr):
                    return False
        except (
            AttributeError,  # no such attribute
            TypeError,  # attribute values can't be compared
        ):
            return False
        return True

    def __hash__(self) -> int:
        """Return a hash of this `Channel`."""
        hash_ = 0
        for attr in ("name", "sample_rate", "unit", "url", "type", "dtype"):
            hash_ += hash(getattr(self, attr))
        return hash_


class ChannelList(list):
    """A `list` of `channels <Channel>`, with parsing utilities."""

    # -- properties ------------------

    @property
    def ifos(self) -> set[str]:
        """The `set` of interferometer prefixes used in this `ChannelList`."""
        return {c.ifo for c in self}

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(ChannelListRead)
    write = UnifiedReadWriteMethod(ChannelListWrite)

    # -- methods ---------------------

    @classmethod
    def from_names(cls, *names: str) -> Self:
        """Create a new `ChannelList` from a list of names.

        The list of names can include comma-separated sets of names,
        in which case the return will be a flattened list of all parsed
        channel names.
        """
        new = cls()
        for namestr in names:
            for name in cls._split_names(namestr):
                new.append(Channel(name))
        return new

    @staticmethod
    def _split_names(namestr: str) -> list[str]:
        """Split a comma-separated list of channel names."""
        out = []
        namestr = QUOTE_REGEX.sub("", namestr)
        while True:
            namestr = namestr.strip("' \n")
            if "," not in namestr:
                break
            for nds2type in [*io_nds2.Nds2ChannelType.nds2names(), ""]:
                if nds2type and f",{nds2type}" in namestr:
                    try:
                        channel, ctype, namestr = namestr.split(",", 2)
                    except ValueError:
                        channel, ctype = namestr.split(",")
                        namestr = ""
                    out.append(f"{channel},{ctype}")
                    break
                if nds2type == "" and "," in namestr:
                    channel, namestr = namestr.split(",", 1)
                    out.append(channel)
                    break
        if namestr:
            out.append(namestr)
        return out

    def find(self, name: str) -> int:
        """Find the `Channel` with a specific name in this `ChannelList`.

        Parameters
        ----------
        name : `str`
            name of the `Channel` to find

        Returns
        -------
        index : `int`
            the position of the first `Channel` in this `ChannelList`
            whose `~Channel.name` matches the search key.

        Raises
        ------
        ValueError
            if no matching `Channel` is found.
        """
        for i, chan in enumerate(self):
            if name == chan.name:
                return i
        raise ValueError(name)

    def sieve(
        self,
        name: str | re.Pattern | None = None,
        sample_rate: float | None = None,
        *,
        sample_range: tuple[float, float] | None = None,
        exact_match: bool = False,
        **others,
    ) -> Self:
        """Find all channels in this list matching the specified criteria.

        Parameters
        ----------
        name : `str`, or regular expression
            Any part of the channel name against which to match
            (or full name if `exact_match=False` is given).

        sample_rate : `float`
            Rate (number of samples per second) to match exactly.

        sample_range : 2-`tuple`
            `[low, high]` closed interval or rates to match within.

        exact_match : `bool`
            If `True` return channels matching ``name`` exactly.
            If `False` (default) allow partial matches.

        others:
            Other ``(key, value)`` attribute pairs to match.

        Returns
        -------
        new : `ChannelList`
            A new `ChannelList` containing the matching channels.
        """
        # format name regex
        if isinstance(name, re.Pattern):
            flags = name.flags
            name = str(name.pattern)
        else:
            flags = 0
        if name is not None:
            if exact_match:
                name = name if name.startswith(r"\A") else fr"\A{name}"
                name = name if name.endswith(r"\Z") else fr"{name}\Z"
            name_regexp = re.compile(name, flags=flags)

        def _match(channel: Channel) -> bool:
            if name is not None and name_regexp.search(channel.name) is None:
                return False
            if sample_rate is not None and channel.sample_rate != sample_rate:
                return False
            if sample_range is not None and (
                channel.sample_rate is None
                or sample_range[0] > channel.sample_rate.value
                or sample_range[1] <= channel.sample_rate.value
            ):
                return False
            for key, val in others.items():
                if val is not None and getattr(channel, key, None) != val:
                    return False
            return True

        return type(self)(filter(_match, self))

    @classmethod
    def query(
        cls,
        name: str,
        *,
        kerberos: bool | None = None,
        **kwargs,
    ) -> Self:
        """Query the LIGO Channel Information System a `ChannelList`.

        Parameters
        ----------
        name : `str`
            name of channel, or part of it.

        kerberos : `bool`, optional
            If `True` use an existing Kerberos ticket as the authentication credential.
            Default (`None`) is to check for credentials and request username
            and password if none are found.

        kwargs
            Other keyword arguments are passed directly to
            :func:`ciecplib.get`.

        Returns
        -------
        channels : `ChannelList`
            A new list containing all `Channels <Channel>` found.
        """
        from .io import cis
        return cls(cis.query(
            name,
            kerberos=kerberos,
            **kwargs,
        ))

    @classmethod
    def query_nds2(
        cls,
        names: list[str],
        *,
        host: str | None = None,
        port: int | None = None,
        connection: nds2.connection | None = None,
        type: str | int | None = io_nds2.NDS2_CHANNEL_TYPE_ANY,  # noqa: A002
        unique: bool = False,
    ) -> Self:
        """Query an NDS server for channel information.

        Parameters
        ----------
        names : `str`
            Names of requested channels.

        host : `str`, optional
            Name of NDS2 server.

        port : `int`, optional
            Port number for NDS2 connection.

        connection : `nds2.connection`, optional
            Open connection to use for query.

        type : `str`, `int`, optional
            NDS2 channel type with which to restrict query

        unique : `bool`, optional
            If `True` require a unique query result for each name given.
            Default is `False`.

        Returns
        -------
        channellist : `ChannelList`
            list of `Channels <Channel>` with metadata retrieved from
            NDS2 server

        Raises
        ------
        ValueError
            If multiple channels are found for a given name and `unique=True`
            is given.

        Notes
        -----
        .. warning::

            One of ``host`` or ``connection`` is required.
        """
        ndschannels = io_nds2.find_channels(
            names,
            host=host,
            port=port,
            connection=connection,
            type=type,
            unique=unique,
        )
        return cls(map(Channel.from_nds2, ndschannels))

    @classmethod
    def query_nds2_availability(
        cls,
        channels: list[str | Channel],
        start: SupportsToGps,
        end: SupportsToGps,
        ctype: int | str = io_nds2.Nds2ChannelType.any().value,
        connection: nds2.connection | None = None,
        host: str | None = None,
        port: int | None = None,
    ) -> SegmentListDict:
        """Query for when data are available for these channels in NDS2.

        Parameters
        ----------
        channels : `list` of `str` or `Channel`
            List of `Channel` or `str` for which to search.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of search.
            Any input parseable by `~gwpy.time.to_gps` is fine.

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of search.
            Any input parseable by `~gwpy.time.to_gps` is fine.

        ctype : `int`, `str`
            The NDS2 channel type name or enum ID against which to restrict
            results. Default is ``127`` which means 'all'.

        host : `str`, optional
            Name of NDS2 server to use.

        port : `int`, optional
            Port number for NDS2 connection.

        connection : `nds2.connection`, optional
            Open connection to use for query.

        Returns
        -------
        segdict : `~gwpy.segments.SegmentListDict`
            dict of ``(name, SegmentList)`` pairs
        """
        start = int(to_gps(start))
        end = ceil(to_gps(end))
        with io_nds2._connection(connection=connection, host=host, port=port) as conn:
            chans = io_nds2.find_channels(
                channels,
                unique=True,
                epoch=(start, end),
                type=ctype,
                connection=conn,
            )
            availability = io_nds2.get_availability(
                chans,
                start,
                end,
                connection=conn,
            )
        return type(availability)(zip(
            channels,
            availability.values(),
            strict=True,
        ))
