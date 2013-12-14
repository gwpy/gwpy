# coding=utf-8
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

"""Read and write gravitational-wave frame files.

The frame format is defined in LIGO-T970130 available from dcc.ligo.org.
"""

from __future__ import division

import threading
import time
from math import ceil
from multiprocessing import (Process, Queue as ProcessQueue)
from Queue import Queue

from astropy.io import registry

from glue.lal import (Cache, CacheEntry)

try:
    from glue.lal import CacheEntry
except ImportError:
    HASGLUE = False
else:
    HASGLUE = True

from ..detector import Channel
from ..time import Time
from ..timeseries import (TimeSeries, StateVector)


def read_gwf(framefile, channel, start=None, end=None, datatype=None,
             verbose=False, _target=TimeSeries):
    """Read a `TimeSeries` of data from a gravitational-wave frame file

    This method is a thin wrapper around `lalframe.frread.read_timeseries`
    and so can accept any input accepted by that function.

    Parameters
    ----------
    framefile : `str`, :class:`glue.lal.Cache`, :lalsuite:`LALCache`
        data source object, one of:

        - `str` : frame file path
        - :class:`glue.lal.Cache` : pure python cache object
        - :lalsuite:`LALCAche` : C-based cache object

    channel : :class:`~gwpy.detector.channel.Channel`, `str`
        data channel to read from frames
    start : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        start GPS time of desired data
    end : `Time`, :lalsuite:`LIGOTimeGPS`, optional
        end GPS time of desired data
    datatype : `type`, `numpy.dtype`, `str`, optional
        identifier for desired output data type
    verbose : `bool`, optional
        print verbose output

    Returns
    -------
    data : :class:`~gwpy.timeseries.core.TimeSeries`
        a new `TimeSeries` containing the data read from disk
    """
    try:
        from lalframe import frread
    except:
        raise ImportError("No module named lalframe. LALFrame or frameCPP are "
                          "required in order to read data from GWF-format "
                          "frame files.")
    # parse input arguments
    if isinstance(framefile, CacheEntry):
        framefile = framefile.path
    elif isinstance(framefile, file):
        framefile = framefile.name
    if isinstance(channel, Channel):
        channel = channel.name
    if start and isinstance(start, Time):
        start = start.gps
    if end and isinstance(end, Time):
        end = end.gps
    if start and end:
        duration = float(end - start)
    elif end:
        raise ValueError("If `end` is given, `start` must also be given")
    else:
        duration = None
    lalts = frread.read_timeseries(framefile, channel, start=start,
                                   duration=duration, datatype=datatype,
                                   verbose=verbose)
    return TimeSeries.from_lal(lalts)


def read_state_vector(*args, **kwargs):
    bitmask = kwargs.pop('bitmask', [])
    if isinstance(args[0], file):
        args = list(args)
        args[0] = args[0].name
    new = TimeSeries.read(*args, **kwargs).view(StateVector)
    new.bitmask = bitmask
    return new


def identify_gwf(*args, **kwargs):
    """Determine an input file as written in GWF-format.
    """
    filename = args[1]
    ce = args[3]
    if isinstance(filename, (str, unicode)) and filename.endswith('gwf'):
        return True
    elif HASGLUE and isinstance(ce, CacheEntry):
        return True
    else:
        return False


registry.register_reader('gwf', TimeSeries, read_gwf, force=True)
registry.register_identifier('gwf', TimeSeries, identify_gwf)
registry.register_reader('gwf', StateVector, read_state_vector, force=True)
registry.register_identifier('gwf', StateVector, identify_gwf)
