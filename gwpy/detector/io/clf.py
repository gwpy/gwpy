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

"""Read and write LIGO Channel List Files.

The Channel List File (CLF) is a schema of the INI configuration file format
designed to hold lists of interferometer data channels for bulk processing.

Each CLF file should contain one or more `[groups]` each containing, at
least, an option named `channels` with a newline-delimited list of channel
names, e.g.

.. code-block:: ini

    [my channels]
    channels =
        X1:CHANNEL-1
        X1:CHANNEL-2

Each channel line can also optionally contain a sampling rate number, e.g.

.. code-block:: ini

    [my channels]
    channels =
       X1:CHANNEL-1
       X1:CHANNEL-2
       X1:CHANNEL-3 2048

Other recommended options to include with a channel group include

  - `frametype`: the GWF type for files containing data for these channels
  - `flow`: the lower-frequency bound for processing these channels
  - `fhigh`: the upper-frequency bound for processing these channels
  - `qhigh`: the upper bound on sine-Gaussian Q for processing these channels

For example,

.. code-block:: ini

    [ALS]
    flow = 32
    qhigh = 100
    frametype = L1_R
    channels =
        L1:ALS-X_ARM_IN1_DQ 2048
        L1:ALS-Y_ARM_IN1_DQ

"""

from __future__ import annotations

import configparser
import re
from typing import TYPE_CHECKING

from numpy import inf

from ...io.registry import identify_factory
from ...io.utils import (
    file_list,
    with_open,
)
from .. import (
    Channel,
    ChannelList,
)

if TYPE_CHECKING:
    from pathlib import Path
    from typing import IO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

CHANNEL_DEFINITION = re.compile(
    r"(?P<name>[a-zA-Z0-9:_-]+)"
    r"(?:\s+(?P<sample_rate>[0-9.]+))?"
    r"(?:\s+(?P<safe>(safe|unsafe|unsafeabove2kHz|unknown)))?"
    r"(?:\s+(?P<fidelity>(clean|flat|glitchy|unknown)))?",
)


# -- read ----------------------------

def read_channel_list_file(*source: str | Path | IO) -> ChannelList:
    """Read a `~gwpy.detector.ChannelList` from a Channel List File."""
    # read file(s)
    config = configparser.ConfigParser()
    files = file_list(source)
    success_ = config.read(files)
    if failed := set(files) - set(success_):
        msg = f"Failed to read '{failed.pop()}'"
        raise OSError(msg)

    # create channel list
    out = ChannelList()
    out.source = source
    append = out.append

    # loop over all groups and channels
    for group in config.sections():
        params = dict(config.items(group))

        # parse flow and fhigh as 'frange'
        if "flow" in params or "fhigh" in params:
            low = params.pop("flow", 0)
            high = params.pop("fhigh", inf)
            if isinstance(high, str) and high.lower() == "nyquist":
                high = inf
            frange = float(low), float(high)
        else:
            frange = None

        for channel in params.pop("channels").strip("\n").split("\n"):
            append(_read_channel(channel, group, frange, **params))

    return out


def _read_channel(
    value: str,
    group: str,
    frange: tuple[float, float] | None,
    **params,
) -> Channel:
    """Parse a single channel."""
    if not (match := CHANNEL_DEFINITION.match(value)):
        msg = f"Cannot parse '{value}' as channel list entry"
        raise ValueError(msg)

    # remove Nones from match
    inlineattrs = {k: v for k, v in match.groupdict().items() if v is not None}
    inlineattrs.setdefault("fidelity", "clean")

    # parse safe
    safe = inlineattrs.get("safe", "safe").lower() != "unsafe"

    # create channel and copy group params
    channel = Channel(
        inlineattrs.pop("name"),
        frequency_range=frange,
        safe=safe,
        sample_rate=inlineattrs.pop("sample_rate"),
    )
    channel.params = params | inlineattrs
    channel.group = group

    # extract those params for which the Channel has an attribute
    for key in ["frametype"]:
        setattr(channel, key, channel.params.pop(key, None))

    return channel


# -- write ---------------------------

@with_open(mode="w", pos=1)
def write_channel_list_file(
    channels: ChannelList,
    fobj: IO,
) -> None:
    """Write a `~gwpy.detector.ChannelList` to a INI-format channel list file."""
    # create the configparser
    out = configparser.ConfigParser()
    for channel in channels:
        _write_channel_ini(out, channel)

    # write to file
    out.write(fobj)


def _write_channel_ini(
    ini: configparser.ConfigParser,
    channel: Channel,
) -> None:
    """Write a `Channel` to a `~configparser.ConfigParser`."""
    group = channel.group
    if not ini.has_section(group):
        ini.add_section(group)
    for param, value in channel.params.items():
        ini.set(group, param, value)
    if channel.sample_rate is not None:
        entry = f"{channel} {channel.sample_rate.to('Hz').value}"
    else:
        entry = str(channel)
    entry += f" {channel.params.get('safe', 'safe')}"
    entry += f" {channel.params.get('fidelity', 'clean')}"
    try:
        clist = ini.get(group, "channels")
    except configparser.NoOptionError:
        ini.set(group, "channels", f"\n{entry}")
    else:
        ini.set(group, "channels", clist + f"\n{entry}")


# -- Unified I/O registration --------

ChannelList.read.registry.register_reader(
    "ini",
    ChannelList,
    read_channel_list_file,
)
ChannelList.read.registry.register_identifier(
    "ini",
    ChannelList,
    identify_factory(".ini", ".clf"),
)
ChannelList.write.registry.register_writer(
    "ini",
    ChannelList,
    write_channel_list_file,
)
