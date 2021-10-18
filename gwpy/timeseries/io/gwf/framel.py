# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2020)
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

"""Read gravitational-wave frame (GWF) files using the FrameL API

The frame foramt is defined in LIGO-T970130 available from dcc.ligo.org
"""

import warnings

import framel

from ....io.gwf import _series_name
from ....io.utils import (
    file_list,
    file_path,
)
from ....segments import Segment
from ... import TimeSeries

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

FRAME_LIBRARY = "framel"


# -- read ---------------------------------------------------------------------

def read(source, channels, start=None, end=None, series_class=TimeSeries,
         scaled=None):
    """Read data from one or more GWF files using the FrameL API
    """
    # scaled must be provided to provide a consistent API with frameCPP
    if scaled is not None:
        warnings.warn(
            "the `scaled` keyword argument is not supported by framel, "
            "if you require ADC scaling, please install "
            "python-ldas-tools-framecpp",
        )

    # parse input source
    source = file_list(source)

    # get duration
    crop = start is None and end is not None
    duration = -1
    span = Segment(start or 0, end or 0)
    framelstart = start or -1
    if start and end:
        duration = end - start

    # read each file and channel individually and append
    out = series_class.DictClass()
    for i, file_ in enumerate(source):
        for name in channels:
            new = _read_channel(
                file_,
                name,
                framelstart,
                duration,
                series_class,
            )
            if crop and end < new.x0.value:
                raise ValueError(
                    "read() given end GPS earlier than start GPS for "
                    "{} in {}".format(
                        name,
                        file_,
                    ),
                )
            elif crop:
                new = new.crop(end=end)
            out.append({name: new})

        # if we have all of the data we want, stop now
        if all(span in out[channel].span for channel in out):
            break

    return out


def _read_channel(filename, channel, start, duration, series_class):
    """Read one channel from one file
    """
    try:
        data, gps, offset, dx, xunit, yunit = framel.frgetvect1d(
            str(filename),
            str(channel),
            start=start,
            span=duration,
        )
    except KeyError as exc:  # upstream errors
        raise ValueError(str(exc)) from exc
    return series_class(
        data,
        name=channel,
        x0=gps+offset,
        dx=dx,
        xunit=xunit,
        unit=yunit,
    )


# -- write --------------------------------------------------------------------

def write(
        tsdict,
        outfile,
        start=None,
        end=None,
        type=None,
        name=None,
        run=None,
):
    """Write data to a GWF file using the FrameL API
    """
    if name is not None:
        warnings.warn(
            "specifying a FrHistory name via FrameL is not supported, "
            "this value will be ignored",
        )
    if run is not None:
        warnings.warn(
            "specifying a FrHistory run number via FrameL is not supported, "
            "this value will be ignored",
        )

    # format and crop each series
    channellist = []
    for series in tsdict.values():
        channellist.append(_channel_data_to_write(
            series.crop(start=start, end=end),
            type,
        ))
    return framel.frputvect(file_path(outfile), channellist)


def _channel_data_to_write(timeseries, type_):
    return {
        "name": _series_name(timeseries),
        "data": timeseries.value,
        "start": timeseries.x0.to("s").value,
        "dx": timeseries.dx.to("s").value,
        "x_unit": str(timeseries.xunit),
        "y_unit": str(timeseries.unit),
        "kind": (
            type_
            or getattr(timeseries.channel, "_ctype", "proc")
            or "proc"
        ).upper(),
        "type": 1,
        "subType": 0,
    }
