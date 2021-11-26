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

"""I/O routines for parsing Omega pipeline scan channel lists
"""

import sys
import os
from collections import OrderedDict

from astropy.io import registry

from ...io.utils import with_open
from .. import (Channel, ChannelList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OMEGA_LOCATION = os.getenv('OMEGA_LOCATION', None)
WPIPELINE = OMEGA_LOCATION and os.path.join(OMEGA_LOCATION, 'bin', 'wpipeline')


# -- read ---------------------------------------------------------------------

@with_open
def read_omega_scan_config(source):
    """Parse an Omega-scan configuration file into a `ChannelList`

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
    for line in map(str.strip, source):
        if not line or line.startswith('#'):
            continue
        if line.startswith('['):
            section = line[1:-1]
        elif line.startswith('{'):
            append(parse_omega_channel(source, section))
        else:
            raise RuntimeError(
                f"Failed to parse Omega config line: {line!r}",
            )
    return out


def parse_omega_channel(fobj, section=None):
    """Parse a `Channel` from an Omega-scan configuration file

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
    params = OrderedDict()
    while True:
        line = next(fobj)
        if line == '}\n':
            break
        key, value = line.split(':', 1)
        params[key.strip().rstrip()] = omega_param(value)
    out = Channel(params.get('channelName'),
                  sample_rate=params.get('sampleFrequency'),
                  frametype=params.get('frameType'),
                  frequency_range=params.get('searchFrequencyRange'))
    out.group = section
    out.params = params
    return out


def omega_param(val):
    """Parse a value from an Omega-scan configuration file

    This method tries to parse matlab-syntax parameters into a `str`,
    `float`, or `tuple`
    """
    val = val.strip().rstrip()
    if val.startswith(('"', "'")):
        return str(val[1:-1])
    if val.startswith('['):
        return tuple(map(float, val[1:-1].split()))
    return float(val)


# -- write --------------------------------------------------------------------

@with_open(mode="w", pos=1)
def write_omega_scan_config(channellist, fobj, header=True):
    """Write a `ChannelList` to an Omega-pipeline scan configuration file

    This method is dumb and assumes the channels are sorted in the right
    order already
    """
    # print header
    if header:
        print('# Q Scan configuration file', file=fobj)
        print('# Generated with GWpy from a ChannelList', file=fobj)
    group = None
    for channel in channellist:
        # print header
        if channel.group != group:
            group = channel.group
            print(f'\n[{group}]', file=fobj)
        print("", file=fobj)
        print_omega_channel(channel, file=fobj)


# pylint: disable=redefined-builtin
def print_omega_channel(channel, file=sys.stdout):
    """Print a `Channel` in Omega-pipeline scan format
    """
    print('{', file=file)
    try:
        params = channel.params.copy()
    except AttributeError:
        params = OrderedDict()
    params.setdefault('channelName', str(channel))
    params.setdefault('alwaysPlotFlag', int(params.pop('important', False)))
    if channel.frametype:
        params.setdefault('frameType', channel.frametype)
    if channel.sample_rate is not None:
        params.setdefault('sampleFrequency',
                          channel.sample_rate.to('Hz').value)
    if channel.frequency_range is not None:
        low, high = channel.frequency_range.to('Hz').value
        params.setdefault('searchFrequencyRange', (low, high))
    if 'qlow' in params or 'qhigh' in params:
        qlow = params.pop('qlow', 'sqrt(11)')
        qhigh = params.pop('qhigh', 64)
        params.setdefault('searchQRange', (qlow, qhigh))
    # write params
    for key in ['channelName', 'frameType']:
        if key not in params:
            raise KeyError(f"No {key!r} defined for {channel}")
    for key, value in params.items():
        key = f'{key}:'
        if isinstance(value, tuple):
            value = f"[{' '.join(map(str, value))}]"
        elif isinstance(value, float) and value.is_integer():
            value = int(value)
        elif isinstance(value, str):
            value = repr(value)
        print(f'  {key: <30}  {value}', file=file)
    print('}', file=file)


# -- registry -----------------------------------------------------------------

registry.register_reader('omega-scan', ChannelList, read_omega_scan_config)
registry.register_writer('omega-scan', ChannelList, write_omega_scan_config)
