# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Mock objects for GWpy tests
"""

import inspect
from unittest import mock

from ..detector import Channel
from ..time import LIGOTimeGPS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- NDS2 ---------------------------------------------------------------------

def nds2_buffer(channel, data, epoch, sample_rate, unit,
                name=None, slope=1, offset=0):
    import nds2
    epoch = LIGOTimeGPS(epoch)
    ndsbuffer = mock.create_autospec(nds2.buffer)
    ndsbuffer.length = len(data)
    ndsbuffer.channel = nds2_channel(channel, sample_rate, unit)
    ndsbuffer.name = name or ndsbuffer.channel.name
    ndsbuffer.sample_rate = sample_rate
    ndsbuffer.gps_seconds = epoch.gpsSeconds
    ndsbuffer.gps_nanoseconds = epoch.gpsNanoSeconds
    ndsbuffer.signal_slope = slope
    ndsbuffer.signal_offset = offset
    ndsbuffer.data = data
    return ndsbuffer


def nds2_buffer_from_timeseries(ts):
    return nds2_buffer(ts.name, ts.value, ts.t0.value,
                       ts.sample_rate.value, str(ts.unit))


def nds2_channel(name, sample_rate, unit):
    import nds2
    channel = mock.create_autospec(nds2.channel)
    channel.name = name
    channel.sample_rate = sample_rate
    channel.signal_units = unit
    channel.channel_type = 2
    channel.channel_type_to_string = nds2.channel.channel_type_to_string
    channel.data_type = 8
    for attr, value in inspect.getmembers(
            nds2.channel, predicate=lambda x: isinstance(x, int)):
        setattr(channel, attr, value)
    return channel


def nds2_connection(host='nds.test.gwpy', port=31200, buffers=[], protocol=2):
    import nds2
    NdsConnection = mock.create_autospec(nds2.connection)
    try:
        NdsConnection.get_parameter.return_value = False
    except AttributeError:
        # nds2-client < 0.12 doesn't have {get,set}_parameter
        pass
    NdsConnection.get_host.return_value = host
    NdsConnection.get_port.return_value = int(port)
    NdsConnection.get_protocol.return_value = int(protocol)

    def iterate(start, end, names):
        if not buffers:
            return []
        return [[b for b in buffers if
                 Channel.from_nds2(b.channel).ndsname in names]]

    NdsConnection.iterate = iterate

    def find_channels(name, ctype, dtype, *sample_rate):
        return [b.channel for b in buffers if b.channel.name == name]

    NdsConnection.find_channels = find_channels

    def get_availability(names):
        out = []
        for buff in buffers:
            name = '{0.name},{0.type}'.format(Channel.from_nds2(buff.channel))
            if name not in names:
                segs = []
            else:
                start = buff.gps_seconds + buff.gps_nanoseconds * 1e-9
                end = start + buff.sample_rate * buff.length
                segs = [(start, end)]
            out.append(nds2_availability(name, segs))
        return out

    NdsConnection.get_availability = get_availability

    return NdsConnection


def nds2_availability(name, segments):
    import nds2
    availability = mock.create_autospec(nds2.availability)
    availability.name = name
    availability.simple_list.return_value = list(map(nds2_segment, segments))
    return availability


def nds2_segment(segment):
    import nds2
    nds2seg = mock.create_autospec(nds2.simple_segment)
    nds2seg.gps_start = segment[0]
    nds2seg.gps_stop = segment[1]
    return nds2seg
