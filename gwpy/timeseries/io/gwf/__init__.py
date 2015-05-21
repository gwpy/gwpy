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

"""Input/output routines for gravitational-wave frame (GWF) format files.

The frame format is defined in LIGO-T970130 available from dcc.ligo.org.

Currently supported are two separate libraries:

- `lalframe` : using the LIGO Algorithm Library Frame API (based off the
  FrameL library)
- `framecpp` : using the alternative ``frameCPP`` library

Due to the lower-level nature of the frameCPP python package, it is
preferred, in the instance that both lalframe and frameCPP are available
on a system.
"""

import importlib

from astropy.io.registry import register_reader

from ....utils import with_import
from ....version import version
from ... import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version

BUILTIN_LIBRARIES = [
    'lalframe',
    'framecpp',
]


def channel_dict_kwarg(value, channels, types=None, astype=None):
    """Format the given kwarg value in a dict with one value per channel

    Parameters
    ----------
    value : any type
        keyword argument value as given by user
    channels : `list`
        list of channels being read
    types : `list` of `type`
        list of valid object types for value
    astype : `type`
        output type for `dict` values

    Returns
    -------
    dict : `dict`
        `dict` of values, one value per channel key, if parsing is successful
    None : `None`
        `None`, if parsing was unsuccessful
    """
    if types is not None and isinstance(value, tuple(types)):
        out = dict((c, value) for c in channels)
    elif isinstance(value, (tuple, list)):
        out = dict(zip(channels, value))
    elif value is None:
        out = dict()
    else:
        return None
    if astype is not None:
        return dict((key, astype(out[key])) for key in out)
    else:
        return out


def register_gwf_io_library(library, package='gwpy.timeseries.io.gwf'):
    """Register a full set of GWF I/O methods for the given library

    The given library must define a `read_timeseriesdict`
    method, taking a GWF source and a list of channels, and kwargs,
    and returning a `TimeSeriesDict` of data.

    Additionally, the library must store the name of the upstream I/O
    library it uses in the 'DEPENDS' variable.

    This method then wraps the `read_timeseriesdict` method from the
    given to provide readers for the `TimeSeries`, `StateVector`, and
    `StateVectorDict` objects
    """
    # import library
    lib = importlib.import_module('.%s' % library, package=package)
    dependency = lib.DEPENDS
    read_timeseriesdict = lib.read_timeseriesdict

    @with_import(dependency)
    def read_timeseries(source, channel, **kwargs):
        """Read `TimeSeries` from GWF source
        """
        return read_timeseriesdict(source, [channel], **kwargs)[channel]

    @with_import(dependency)
    def read_statevector(source, channel, bits=None,
                         _SeriesClass=StateVector, **kwargs):
        """Read `StateVector` from GWF source
        """
        sv = read_timeseries(
            source, channel, _SeriesClass=_SeriesClass, **kwargs)
        sv.bits = bits
        return sv

    @with_import(dependency)
    def read_statevectordict(source, channels, bitss=[],
                             _SeriesClass=StateVector, **kwargs):
        """Read `StateVectorDict` from GWF source
        """
        svd = StateVectorDict(read_timeseriesdict(
            source, channels, _SeriesClass=_SeriesClass, **kwargs))
        for (channel, bits) in zip(channels, bitss):
            svd[channel].bits = bits
        return svd

    # register format
    register_reader(library, TimeSeriesDict, read_timeseriesdict)
    register_reader(library, TimeSeries, read_timeseries)
    register_reader(library, StateVectorDict, read_statevectordict)
    register_reader(library, StateVector, read_statevector)

    # register generic 'GWF' format
    try:
        __import__(dependency, fromlist=[''])
    except ImportError:
        pass
    else:
        register_reader('gwf', TimeSeriesDict, read_timeseriesdict,
                        force=True)
        register_reader('gwf', TimeSeries, read_timeseries,
                        force=True)
        register_reader('gwf', StateVectorDict, read_statevectordict,
                        force=True)
        register_reader('gwf', StateVector, read_statevector,
                        force=True)
    return (read_timeseries, read_timeseriesdict,
            read_statevector, read_statevectordict)


# register builtin libraries
for library in BUILTIN_LIBRARIES:
    register_gwf_io_library(library)
