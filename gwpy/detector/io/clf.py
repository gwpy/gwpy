# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2015-)
#
# This file is part of the GWpy python package.
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
try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from numpy import inf

from ...io import registry
from ...io.utils import identify_factory
from ...utils.compat import OrderedDict
from .. import (Channel, ChannelList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

CHANNEL_DEFINITION = re.compile(
    "(?P<name>[a-zA-Z0-9:_-]+)"
    "(?P<important>\*)?"
    "(?:\s+(?P<sample_rate>[0-9.]+))?"
    "(?:\s+(?P<safe>(safe|unsafe)))?"
)


def read_channel_list_file(*source):
    """Read a `~gwpy.detector.ChannelList` from a Channel List File
    """
    # read file(s)
    config = configparser.ConfigParser(dict_type=OrderedDict)
    source = [f.name for f in source if isinstance(f, file) or f]
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
            lo = params.pop('flow', 0)
            hi = params.pop('fhigh', inf)
            if isinstance(hi, str) and hi.lower() == 'nyquist':
                hi = inf
            frange = float(lo), float(hi)
        else:
            frange = None
        for channel in channels:
            try:
                match = CHANNEL_DEFINITION.match(channel).groupdict()
            except AttributeError as e:
                e.args = ('Cannot parse %r as channel list entry' % channel,)
                raise
            # create channel and copy group params
            if match['safe']:
                match['safe'] = match.pop('safe', 'safe').lower() != 'unsafe'
            important = bool(match.pop('important', False))
            channel = Channel(match.pop('name'), frequency_range=frange,
                              **match)
            channel.params = params.copy()
            channel.params['important'] = important
            channel.group = group
            # extract those params for which the Channel has an attribute
            for key in ['frametype']:
                setattr(channel, key, channel.params.pop(key, None))
            append(channel)
    return out


def write_channel_list_file(channels, fobj):
    """Write a `~gwpy.detector.ChannelList` to a INI-format channel list file
    """
    out = configparser.ConfigParser(dict_type=OrderedDict)
    for channel in channels:
        group = channel.group
        if not out.has_section(group):
            out.add_section(group)
        for param, value in channel.params.iteritems():
            out.set(group, param, value)
        if channel.sample_rate:
            entry = '%s %s' % (str(channel),
                               str(channel.sample_rate.to('Hz').value))
        else:
            entry = str(channel)
        try:
            cl = out.get(group, 'channels')
        except configparser.NoOptionError:
            out.set(group, 'channels', '\n%s' % entry)
        else:
            out.set(group, 'channels', cl + '\n%s' % entry)
    if isinstance(fobj, file):
        close = False
    else:
        fobj = open(fobj, 'w')
        close = True
    out.write(fobj)
    if close:
        fobj.close()


registry.register_reader('ini', ChannelList, read_channel_list_file)
registry.register_identifier('ini', ChannelList,
                             identify_factory('.ini', '.clf'))
registry.register_writer('ini', ChannelList, write_channel_list_file)
