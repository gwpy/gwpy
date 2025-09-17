# Copyright (c) 2017 Louisiana State University
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

"""Mock objects for GWpy tests."""

from __future__ import annotations

import inspect
from typing import TYPE_CHECKING
from unittest import mock

from ..detector import Channel
from ..time import LIGOTimeGPS

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import TypeAlias

    import nds2
    from astropy.units import UnitBase
    from numpy.typing import NDArray

    from ..timeseries import TimeSeries

    SegmentLike: TypeAlias = tuple[float, float]
    SegmentListLike: TypeAlias = list[tuple[float, float]]

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- NDS2 ----------------------------

def nds2_buffer(
    channel: str,
    data: list | NDArray,
    epoch: LIGOTimeGPS | float,
    sample_rate: float,
    unit: str | UnitBase,
    name: str | None = None,
    slope: float = 1,
    offset: float = 0,
    channel_type: int = 2,
    data_type: int = 8,
) -> nds2.buffer:
    """Create a mocked `nds2.buffer`."""
    import nds2
    epoch = LIGOTimeGPS(epoch)
    ndsbuffer = mock.create_autospec(nds2.buffer)
    ndsbuffer.length = len(data)
    ndsbuffer.channel = nds2_channel(
        channel,
        sample_rate,
        unit,
        channel_type=channel_type,
        data_type=data_type,
    )
    ndsbuffer.name = name or ndsbuffer.channel.name
    ndsbuffer.sample_rate = sample_rate
    ndsbuffer.gps_seconds = epoch.gpsSeconds
    ndsbuffer.gps_nanoseconds = epoch.gpsNanoSeconds
    ndsbuffer.signal_slope = slope
    ndsbuffer.signal_offset = offset
    ndsbuffer.data = data
    return ndsbuffer


def nds2_buffer_from_timeseries(
    ts: TimeSeries,
) -> nds2.buffer:
    """Create a mocked `nds2.buffer` from a :class:`TimeSeries`."""
    return nds2_buffer(
        ts.name or "",
        ts.value,
        ts.x0.value,
        ts.sample_rate.value,
        str(ts.unit),
    )


def nds2_channel(
    name: str,
    sample_rate: float,
    unit: str | UnitBase,
    channel_type: int = 2,
    data_type: int = 8,
) -> nds2.channel:
    """Create a mocked `nds2.channel`."""
    import nds2
    channel = mock.create_autospec(nds2.channel)
    channel.name = name
    channel.sample_rate = sample_rate
    channel.signal_units = str(unit)
    channel.channel_type = channel_type
    channel.channel_type_to_string = nds2.channel.channel_type_to_string
    channel.data_type = data_type
    for attr, value in inspect.getmembers(
        nds2.channel,
        predicate=lambda x: isinstance(x, int),
    ):
        setattr(channel, attr, value)
    return channel


def nds2_connection(
    host: str = "nds.test.gwpy",
    port: int = 31200,
    buffers: Iterable[nds2.buffer] = [],
    protocol: int = 2,
) -> nds2.connection:
    """Create a mock an `nds2.connection` that returns the given buffers."""
    import nds2
    NdsConnection = mock.create_autospec(nds2.connection)  # noqa: N806
    NdsConnection.get_parameter.return_value = False
    NdsConnection.get_host.return_value = host
    NdsConnection.get_port.return_value = int(port)
    NdsConnection.get_protocol.return_value = int(protocol)

    # store buffers internally
    NdsConnection._buffers = list(buffers)  # noqa: SLF001

    def iterate(
        start: float,  # noqa: ARG001
        end: float,  # noqa: ARG001
        names: list[str],
    ) -> list[list[nds2.buffer]]:
        if not NdsConnection._buffers:  # noqa: SLF001
            return []
        return [[
            b for b in NdsConnection._buffers  # noqa: SLF001
            if Channel.from_nds2(b.channel).ndsname in names
        ]]

    NdsConnection.iterate = mock.Mock(side_effect=iterate)

    def find_channels(
        channel_glob: str = "*",
        channel_type_mask: int = nds2.channel.DEFAULT_CHANNEL_MASK,
        data_type_mask: int = nds2.channel.DEFAULT_DATA_MASK,
        min_sample_rate: float = nds2.channel.MIN_SAMPLE_RATE,
        max_sample_rate: float = nds2.channel.MAX_SAMPLE_RATE,
    ) -> list[nds2.channel]:
        out = []
        for b in NdsConnection._buffers:  # noqa: SLF001
            chan = b.channel
            if (
                chan.name == channel_glob
                and chan.sample_rate >= min_sample_rate
                and chan.sample_rate <= max_sample_rate
                and chan.channel_type & channel_type_mask
                and chan.data_type & data_type_mask
            ):
                out.append(chan)
        return out

    NdsConnection.find_channels = mock.Mock(side_effect=find_channels)

    def get_availability(
        names: list[str],
    ) -> list[nds2.availability]:
        out = []
        match = set()
        for buff in NdsConnection._buffers:  # noqa: SLF001
            name = "{0.name},{0.type}".format(Channel.from_nds2(buff.channel))
            if name not in names:
                continue
            start = buff.gps_seconds + buff.gps_nanoseconds * 1e-9
            end = start + buff.length / buff.sample_rate
            segs = [(start, end)]
            out.append(nds2_availability(name, segs))
            match.add(name)
        if missing := match.symmetric_difference(names):
            msg = f"bad channel: {missing.pop()}"
            raise RuntimeError(msg)
        return out

    NdsConnection.get_availability = mock.Mock(side_effect=get_availability)

    return NdsConnection


def nds2_availability(
    name: str,
    segments: SegmentListLike,
) -> nds2.availability:
    """Create a mock `nds2.availability` object."""
    import nds2
    segs = list(map(nds2_segment, segments))
    availability = mock.create_autospec(nds2.availability)
    availability.name = name
    availability.data = segs
    availability.simple_list.return_value = segs
    return availability


def nds2_segment(
    segment: SegmentLike,
) -> nds2.segment:
    """Create a mock `nds2.simple_segment`."""
    import nds2
    nds2seg = mock.create_autospec(nds2.segment)
    nds2seg.gps_start = segment[0]
    nds2seg.gps_stop = segment[1]
    return nds2seg
