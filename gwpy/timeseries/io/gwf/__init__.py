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

from ....utils import with_import
from ....io.registry import (register_reader,
                             register_writer,
                             register_identifier)
from ....io.utils import identify_factory
from ... import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

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
    elif isinstance(value, dict):
        out = value.copy()
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
    reader = lib.read_timeseriesdict

    def read_timeseriesdict(source, *args, **kwargs):
        """Read `TimeSeriesDict` from GWF source
        """
        if isinstance(source, list) and not len(source):
            raise ValueError("Cannot read from empty %s"
                             % type(source).__name__)
        # use multiprocessing or padding
        nproc = kwargs.pop('nproc', 1)
        pad = kwargs.pop('pad', None)
        if nproc > 1 or pad is not None:
            from ..cache import read_cache
            kwargs['target'] = TimeSeriesDict
            kwargs['nproc'] = nproc
            kwargs['pad'] = pad
            return read_cache(source, *args, **kwargs)
        else:
            return reader(source, *args, **kwargs)

    @with_import(dependency)
    def read_timeseries(source, channel, *args, **kwargs):
        """Read `TimeSeries` from GWF source
        """
        return read_timeseriesdict(source, [channel], *args, **kwargs)[channel]

    @with_import(dependency)
    def read_statevector(source, channel, *args, **kwargs):
        """Read `StateVector` from GWF source
        """
        bits = kwargs.pop('bits', None)
        kwargs.setdefault('_SeriesClass', StateVector)
        sv = read_timeseries(source, channel, *args, **kwargs)
        sv.bits = bits
        return sv

    @with_import(dependency)
    def read_statevectordict(source, channels, *args, **kwargs):
        """Read `StateVectorDict` from GWF source
        """
        bitss = kwargs.pop('bits', {})
        # read list of bit lists
        if (isinstance(bitss, (list, tuple)) and len(bitss) and
                isinstance(bitss[0], (list, tuple))):
            bitss = dict(zip(channels, bitss))
        # read single list for all channels
        elif isinstance(bitss, (list, tuple)):
            bitss = dict((channel, bitss) for channel in channels)
        # otherwise assume dict of bit lists

        # read data as timeseriesdict and repackage with bits
        kwargs.setdefault('_SeriesClass', StateVector)
        svd = StateVectorDict(
            read_timeseriesdict(source, channels, *args, **kwargs))
        for (channel, bits) in bitss.iteritems():
            svd[channel].bits = bits
        return svd

    # register format
    register_reader(library, TimeSeriesDict, read_timeseriesdict)
    register_reader(library, TimeSeries, read_timeseries)
    register_reader(library, StateVectorDict, read_statevectordict)
    register_reader(library, StateVector, read_statevector)

    # -- writer

    try:
        write_timeseriesdict = lib.write_timeseriesdict
    except AttributeError:
        pass
    else:
        @with_import(dependency)
        def write_timeseries(timeseries, target, *args, **kwargs):
            """Write a `TimeSeries` to GWF file
            """
            return write_timeseriesdict({None: timeseries}, target,
                                        *args, **kwargs)

        # register format
        register_writer(library, TimeSeriesDict, write_timeseriesdict)
        register_writer(library, TimeSeries, write_timeseries)
        register_writer(library, StateVectorDict, write_timeseriesdict)
        register_writer(library, StateVector, write_timeseries)

    # -- identifier

    # register .gwf identifier
    for cls in [TimeSeriesDict, TimeSeries, StateVectorDict, StateVector]:
        register_identifier('gwf', cls, identify_factory('gwf'), force=True)

    # -- register generic 'GWF' format

    try:
        __import__(dependency, fromlist=[''])
    except ImportError:
        pass
    else:
        register_reader('gwf', TimeSeriesDict, read_timeseriesdict, force=True)
        register_reader('gwf', TimeSeries, read_timeseries, force=True)
        register_reader('gwf', StateVectorDict, read_statevectordict, force=True)
        register_reader('gwf', StateVector, read_statevector, force=True)
        try:
            register_writer('gwf', TimeSeriesDict, write_timeseriesdict,
                            force=True)
        except (NameError, UnboundLocalError):
            pass
        else:
            register_writer('gwf', TimeSeries, write_timeseries, force=True)
            register_writer('gwf', StateVectorDict, write_timeseriesdict,
                            force=True)
            register_writer('gwf', StateVector, write_timeseries, force=True)


# register builtin libraries
for library in BUILTIN_LIBRARIES:
    register_gwf_io_library(library)
