# Copyright (c) 2015-2017 Louisiana State University
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

"""I/O routines for parsing Omega pipeline scan channel lists."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from ...io.utils import with_open
from .. import (
    Channel,
    ChannelList,
)

if TYPE_CHECKING:
    from typing import IO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

@with_open
def read_omega_scan_config(source: IO) -> ChannelList:
    """Parse an Omega-scan configuration file into a `ChannelList`.

    Parameters
    ----------
    source : `str`
        path of Omega configuration file to parse

    Returns
    -------
    channels : `ChannelList`
        the list of channels (in order) as parsed

    Raises
    ------
    RuntimeError
        if this method finds a line it cannot parse sensibly
    """
    out = ChannelList()
    append = out.append
    section = None
    for line in source:
        line = line.strip()
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            section = line[1:-1]
        elif line.startswith("{"):
            append(parse_omega_channel(source, section))
        else:
            msg = f"Failed to parse Omega config line: '{line}'"
            raise RuntimeError(msg)
    return out


def parse_omega_channel(
    fobj: IO,
    section: str | None = None,
) -> Channel:
    """Parse a `Channel` from an Omega-scan configuration file.

    Parameters
    ----------
    fobj : `file`
        the open file-like object to parse
    section : `str`
        name of section in which this channel should be recorded

    Returns
    -------
    channel : `Channel`
        the channel as parsed from this `file`
    """
    params = {}
    while True:
        line = next(fobj).strip()
        if isinstance(line, bytes):
            line = line.decode("utf-8")
        if not line:  # empty
            continue
        if line == "}":  # end of section
            break
        # parse 'key: value' setting
        key, value = line.split(":", 1)
        params[key.strip()] = omega_param(value)

    # build channel with params
    out = Channel(
        params.get("channelName"),
        sample_rate=params.get("sampleFrequency"),
        frametype=params.get("frameType"),
        frequency_range=params.get("searchFrequencyRange"),
    )
    out.group = section
    out.params = params

    return out


def omega_param(val: str) -> str | float | tuple[float, ...]:
    """Parse a value from an Omega-scan configuration file.

    This method tries to parse matlab-syntax parameters into a `str`,
    `float`, or `tuple`
    """
    val = val.strip()
    if val.startswith(('"', "'")):
        return str(val[1:-1])
    if val.startswith("["):
        return tuple(map(float, val[1:-1].split()))
    return float(val)


# -- write ---------------------------

@with_open(mode="w", pos=1)
def write_omega_scan_config(
    channellist: ChannelList,
    fobj: IO,
    *,
    header: bool = True,
) -> None:
    """Write a `ChannelList` to an Omega-pipeline scan configuration file.

    This method is dumb and assumes the channels are sorted in the right
    order already
    """
    # print header
    if header:
        print("# Q Scan configuration file", file=fobj)
        print("# Generated with GWpy from a ChannelList", file=fobj)
    group = None
    for channel in channellist:
        # print header
        if channel.group != group:
            group = channel.group
            print(f"\n[{group}]", file=fobj)
        print(file=fobj)
        print_omega_channel(channel, file=fobj)


def print_omega_channel(
    channel: Channel,
    file: IO = sys.stdout,
) -> None:
    """Print a `Channel` in Omega-pipeline scan format."""
    print("{", file=file)
    try:
        params = channel.params.copy()
    except AttributeError:
        params = {}
    params.setdefault("channelName", str(channel))
    params.setdefault("alwaysPlotFlag", int(params.pop("important", False)))
    if channel.frametype:
        params.setdefault("frameType", channel.frametype)
    if channel.sample_rate is not None:
        params.setdefault(
            "sampleFrequency",
            channel.sample_rate.to("Hz").value,
        )
    if channel.frequency_range is not None:
        low, high = channel.frequency_range.to("Hz").value
        params.setdefault("searchFrequencyRange", (low, high))
    if "qlow" in params or "qhigh" in params:
        qlow = params.pop("qlow", "sqrt(11)")
        qhigh = params.pop("qhigh", 64)
        params.setdefault("searchQRange", (qlow, qhigh))
    # write params
    for key in ["channelName", "frameType"]:
        if key not in params:
            msg = f"No '{key}' defined for {channel}"
            raise KeyError(msg)
    for key, value in params.items():
        keystr = f"{key}:"
        if isinstance(value, tuple):
            valuestr = f"[{' '.join(map(str, value))}]"
        elif isinstance(value, float) and value.is_integer():
            valuestr = str(int(value))
        elif isinstance(value, str):
            valuestr = repr(value)
        else:
            valuestr = str(value)
        print(f"  {keystr: <30}  {valuestr}", file=file)
    print("}", file=file)


# -- registry ------------------------

ChannelList.read.registry.register_reader(
    "omega-scan",
    ChannelList,
    read_omega_scan_config,
)
ChannelList.write.registry.register_writer(
    "omega-scan",
    ChannelList,
    write_omega_scan_config,
)
