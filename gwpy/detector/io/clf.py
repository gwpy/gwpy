# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2015-2020)
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

"""
Channel List Files
==================

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

import re
import configparser
from collections import OrderedDict

from numpy import inf

from ...io import registry
from ...io.utils import (
    file_list,
    identify_factory,
    with_open,
)
from .. import (Channel, ChannelList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

CHANNEL_DEFINITION = re.compile(
    r"(?P<name>[a-zA-Z0-9:_-]+)"
    r"(?:\s+(?P<sample_rate>[0-9.]+))?"
    r"(?:\s+(?P<safe>(safe|unsafe|unsafeabove2kHz|unknown)))?"
    r"(?:\s+(?P<fidelity>(clean|flat|glitchy|unknown)))?"
)


def read_channel_list_file(*source):
    """Read a `~gwpy.detector.ChannelList` from a Channel List File
    """
    # read file(s)
    config = configparser.ConfigParser(dict_type=OrderedDict)
    source = file_list(source)
    success_ = config.read(*source)
    if len(success_) != len(source):
        raise IOError("Failed to read one or more CLF files")
    # create channel list
    out = ChannelList()
    out.source = source
    append = out.append
    # loop over all groups and channels
    for group in config.sections():
        params = OrderedDict(config.items(group))
        channels = params.pop('channels').strip('\n').split('\n')
        if 'flow' in params or 'fhigh' in params:
            low = params.pop('flow', 0)
            high = params.pop('fhigh', inf)
            if isinstance(high, str) and high.lower() == 'nyquist':
                high = inf
            frange = float(low), float(high)
        else:
            frange = None
        for channel in channels:
            try:
                match = CHANNEL_DEFINITION.match(channel).groupdict()
            except AttributeError as exc:
                exc.args = (f"Cannot parse {channel!r} as channel list entry",)
                raise
            # remove Nones from match
            match = dict((k, v) for k, v in match.items() if v is not None)
            match.setdefault('safe', 'safe')
            match.setdefault('fidelity', 'clean')
            # create channel and copy group params
            safe = match.get('safe', 'safe').lower() != 'unsafe'
            channel = Channel(match.pop('name'), frequency_range=frange,
                              safe=safe, sample_rate=match.pop('sample_rate'))
            channel.params = params.copy()
            channel.params.update(match)
            channel.group = group
            # extract those params for which the Channel has an attribute
            for key in ['frametype']:
                setattr(channel, key, channel.params.pop(key, None))
            append(channel)
    return out


@with_open(mode="w", pos=1)
def write_channel_list_file(channels, fobj):
    """Write a `~gwpy.detector.ChannelList` to a INI-format channel list file
    """
    out = configparser.ConfigParser(dict_type=OrderedDict)
    for channel in channels:
        group = channel.group
        if not out.has_section(group):
            out.add_section(group)
        for param, value in channel.params.items():
            out.set(group, param, value)
        if channel.sample_rate is not None:
            entry = f"{channel} {channel.sample_rate.to('Hz').value}"
        else:
            entry = str(channel)
        entry += f" {channel.params.get('safe', 'safe')}"
        entry += f" {channel.params.get('fidelity', 'clean')}"
        try:
            clist = out.get(group, 'channels')
        except configparser.NoOptionError:
            out.set(group, 'channels', f'\n{entry}')
        else:
            out.set(group, 'channels', clist + f'\n{entry}')

    out.write(fobj)


registry.register_reader('ini', ChannelList, read_channel_list_file)
registry.register_identifier('ini', ChannelList,
                             identify_factory('.ini', '.clf'))
registry.register_writer('ini', ChannelList, write_channel_list_file)
