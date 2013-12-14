# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Read data from gravitational-wave frames using frameCPP.

Direct access to the frameCPP library is the easiest way to read multiple
channels from a single frame file in one go.
"""

from astropy.io import registry

try:
    import frameCPP
except ImportError:
    HASFRAMECPP = False
else:
    HASFRAMECPP = True

from .. import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

from ..timeseries import (TimeSeries, TimeSeriesDict)
from .gwf import identify_gwf


def read_channels(framefile, channels, start=None, end=None, type=None,
                  verbose=False):
    """Read the data for a list of channels from a given GWF-format
    framefile
    """
    if isinstance(channels, (unicode, str)):
        channels = channels.split(',')
    # open file
    stream = frameCPP.IFrameFStream(framefile)
    # read table of contents
    toc = stream.GetTOC()
    nframes = toc.GetNFrame()
    epochs = toc.GTimeS
    # read channel lists: XXX: this needs optimised, most of time taken is
    #                          building the channel lists
    if not type:
        try:
            adcs = toc.GetADC().keys()
        except AttributeError:
            adcs = []
        try:
            procs = toc.GetProc().keys()
        except AttributeError:
            procs = []

    out = TimeSeriesDict()
    for channel in channels:
        name = str(channel)
        if type:
            read_ = getattr(stream, 'ReadFr%sData' % type.title())
        else:
            read_ = (name in adcs and stream.ReadFrAdcData or
                     name in procs and stream.ReadFrProcData or None)
        if read_ is None:
            raise ValueError("Channel %s not found in frame table of contents"
                             % name)
        ts = None
        for i in range(nframes):
            data = read_(i, name)
            offset = data.GetTimeOffset()
            fs = data.GetSampleRate()
            for vect in data.data:
                arr = vect.GetDataArray().copy()
                print(arr)
                if ts is None:
                    unit = vect.GetUnitY()
                    ts = TimeSeries(arr, epoch=epochs[i] + offset,
                                    sample_rate=fs, name=name, channel=channel,
                                    unit=unit)
                else:
                    ts.append(arr)
        if ts is not None:
            out[name] = ts

    return out


if HASFRAMECPP:
    registry.register_reader('gwf', TimeSeriesDict, read_channels)
    registry.register_identifier('gwf', TimeSeriesDict, identify_gwf)
