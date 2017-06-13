# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

"""NDS2 data query routines for the TimeSeries
"""

import sys

import operator
import warnings
from math import ceil

from six.moves import reduce

from ...io import nds2 as io_nds2
from ...segments import (Segment, SegmentList)
from ...utils import gprint
from .. import (TimeSeries)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def print_verbose(*args, **kwargs):
    if kwargs.pop('verbose', False):
        gprint(*args, **kwargs)


def parse_nds_enum_dict_param(channels, key, value):
    if key == 'type':
        enum = io_nds2.Nds2ChannelType
        default = enum.any() - enum.ONLINE.value
    else:
        enum = io_nds2.Nds2DataType
        default = enum.any()
    # set default
    if value is None:
        value = default
    # parse non-int enum representation
    if not isinstance(value, (dict, int)):
        value = enum.find(value).value
    # return dict of ints
    if isinstance(value, int):
        return dict((c, value) for c in channels)
    # here we know ``value`` is a dict, so just fill in the blanks
    value = value.copy()
    for chan in channels:
        value.setdefault(chan, default)
    return value


@io_nds2.open_connection
def fetch(channels, start, end, type=None, dtype=None, allow_tape=None,
          connection=None, host=None, port=None, pad=None, verbose=False,
          series_class=TimeSeries):

    # set allow_tape parameter in connection
    if allow_tape is not None:
        try:
            connection.set_parameter('ALLOW_DATA_ON_TAPE', str(allow_tape))
        except AttributeError:
            warnings.warn("This version of the nds2-client does not "
                          "support the allow_tape. This operation will "
                          "continue using whatever default is set by the "
                          "target NDS server", io_nds2.NDSWarning)

    type = parse_nds_enum_dict_param(channels, 'type', type)
    dtype = parse_nds_enum_dict_param(channels, 'dtype', dtype)

    # verify channels exist
    print_verbose("Checking channels list against NDS2 database...", end=' ',
                  verbose=verbose)
    utype = reduce(operator.or_, type.values())  # logical OR of types
    udtype = reduce(operator.or_, dtype.values())
    ndschannels = io_nds2.find_channels(channels, connection=connection,
                                        type=utype, dtype=udtype, unique=True,
                                        epoch=(start, end))
    names = ['%s,%s' % (c.name, c.channel_type_to_string(c.channel_type)) for
             c in ndschannels]
    print_verbose('done', verbose=verbose)

    # handle minute trend timing
    if (any(c.endswith('m-trend') for c in names) and
            (start % 60 or end % 60)):
        warnings.warn("Requested at least one minute trend, but "
                      "start and stop GPS times are not multiples of "
                      "60. Times will be expanded outwards to compensate")
        start, end = io_nds2.minute_trend_times(start, end)

    # get data availability
    span = SegmentList([Segment(start, end)])
    if pad is None:
        qsegs = span
        gap = None
    else:
        print_verbose("Querying for data availability...", end=' ',
                      verbose=verbose)
        gap = 'pad'
        qsegs = io_nds2.get_availability(
            ndschannels, start, end, connection=connection).intersection()
        qsegs &= span
        print_verbose('done\nFound {0} viable segments of data with {1}%% '
                      'coverage'.format(len(qsegs),
                                        abs(qsegs) / abs(span) * 100),
                      verbose=verbose)
        if span - qsegs:
            warnings.warn("Gaps were found in data available from {0}, "
                          "but will be padded with {1}".format(
                              connection.get_host(), pad))

    # query for each segment
    out = series_class.DictClass()
    for seg in qsegs:
        print_verbose("Downloading data... ", end='\r', verbose=verbose)
        data = connection.iterate(int(seg[0]), int(seg[1]), names)
        nsteps = i = 0
        for i, buffers in enumerate(data):
            for buffer_, c in zip(buffers, channels):
                ts = series_class.from_nds2_buffer(buffer_)
                out.append({c: ts}, pad=pad, gap=gap)
            if not nsteps:  # work out how many chunks we're going to get
                dur = int(buffer_.length / buffer_.channel.sample_rate)
                nsteps = ceil((abs(seg) / dur))
            print_verbose("Downloading data... {0}%%".format(
                              100 * (i + 1) // nsteps),
                          end='\r', verbose=verbose)
        print_verbose('', verbose=verbose)

    return out
