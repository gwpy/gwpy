# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2022)
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
import os
from functools import wraps

import numpy

from astropy.io.registry import (get_reader, get_writer)

from ligo.segments import segment as LigoSegment

from ....time import to_gps
from ....io import gwf as io_gwf
from ....io import cache as io_cache
from ....io.registry import compat as compat_registry
from ... import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

BACKENDS = [
    "frameCPP",
    "FrameL",
    "LALFrame",
]


# -- utilities ----------------------------------------------------------------

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
    return out


# -- generic I/O methods ------------------------------------------------------

def register_gwf_backend(backend):
    """Register a full set of GWF I/O methods for the given backend.

    The `timeseries.io.gwf` backend module must define the following methods:

    - `read` : which receives one of more frame files which can be assumed
               to be contiguous, and should return a `TimeSeriesDict`
    - `write` : which receives an output frame file path an a `TimeSeriesDict`
                and does all of the work

    Additionally, the backend must store the name of the third-party
    dependency using the ``FRAME_LIBRARY`` variable.
    """
    # import backend to get details (don't require backend to importable)
    try:
        lib = io_gwf.import_backend(backend, package=__package__)
    except ImportError:
        # _import_backend wrapper will fail before we get to this 'function'
        libread_ = None
        libwrite_ = None
    else:
        libread_ = lib.read
        libwrite_ = lib.write

    # set I/O format name
    fmt = f"gwf.{backend.lower()}"

    def _import_backend(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # import the frame backend to have any ImportErrors occur early
            io_gwf.import_backend(
                backend,
                package=__package__,
            )
            return func(*args, **kwargs)

        return wrapper


    # -- read -----------------------------------

    @_import_backend
    def read_timeseriesdict(source, channels, start=None, end=None,
                            gap=None, pad=None, nproc=1,
                            series_class=TimeSeries, **kwargs):
        """Read the data for a list of channels from a GWF data source

        Parameters
        ----------
        source : `str`, `list`
            Source of data, any of the following:

            - `str` path of single data file,
            - `str` path of LAL-format cache file,
            - `list` of paths.

        channels : `list`
            list of channel names (or `Channel` objects) to read from frame.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data, defaults to start of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        pad : `float`, optional
            value with which to fill gaps in the source data, if not
            given gaps will result in an exception being raised

        Returns
        -------
        dict : :class:`~gwpy.timeseries.TimeSeriesDict`
            dict of (channel, `TimeSeries`) data pairs
        """
        # import the frame backend here to have any ImportErrors occur early
        io_gwf.import_backend(
            backend,
            package=__package__,
        )

        # -- from here read data

        if start:
            start = to_gps(start)
        if end:
            end = to_gps(end)

        # format gap handling
        if gap is None and pad is not None:
            gap = 'pad'
        elif gap is None:
            gap = 'raise'

        # read cache file up-front
        if (
            (isinstance(source, str) and source.endswith(('.lcf', '.cache')))
            or (
                isinstance(source, io_cache.FILE_LIKE)
                and source.name.endswith(('.lcf', '.cache'))
            )
        ):
            source = io_cache.read_cache(source)
        # separate cache into contiguous segments
        if io_cache.is_cache(source):
            if start is not None and end is not None:
                source = io_cache.sieve(
                    source,
                    segment=LigoSegment(start, end),
                )
            source = list(io_cache.find_contiguous(source))
        # convert everything else into a list if needed
        if not isinstance(source, (list, tuple)):
            source = [source]

        # now read the data
        out = series_class.DictClass()
        for i, src in enumerate(source):
            if i == 1:  # force data into fresh memory so that append works
                for name in out:
                    out[name] = numpy.require(out[name], requirements=['O'])
            out.append(libread_(src, channels, start=start, end=end,
                                series_class=series_class, **kwargs),
                       gap=gap, pad=pad, copy=False)

        return out

    def read_timeseries(source, channel, *args, **kwargs):
        """Read `TimeSeries` from GWF source
        """
        return read_timeseriesdict(source, [channel], *args, **kwargs)[channel]

    def read_statevector(source, channel, *args, **kwargs):
        """Read `StateVector` from GWF source
        """
        bits = kwargs.pop('bits', None)
        kwargs.setdefault('series_class', StateVector)
        statevector = read_timeseries(source, channel, *args, **kwargs)
        statevector.bits = bits
        return statevector

    def read_statevectordict(source, channels, *args, **kwargs):
        """Read `StateVectorDict` from GWF source
        """
        bitss = channel_dict_kwarg(kwargs.pop('bits', {}), channels)
        # read data as timeseriesdict and repackage with bits
        kwargs.setdefault('series_class', StateVector)
        svd = StateVectorDict(
            read_timeseriesdict(source, channels, *args, **kwargs))
        for (channel, bits) in bitss.items():
            svd[channel].bits = bits
        return svd

    # -- write ----------------------------------

    write_timeseriesdict = _import_backend(libwrite_)

    def write_timeseries(series, *args, **kwargs):
        """Write a `TimeSeries` to disk in GWF format

        See also
        --------
        write_timeseriesdict
            for available arguments and keyword arguments
        """
        return write_timeseriesdict({None: series}, *args, **kwargs)

    # -- register -------------------------------

    # register specific format
    compat_registry.register_reader(fmt, TimeSeriesDict, read_timeseriesdict)
    compat_registry.register_reader(fmt, TimeSeries, read_timeseries)
    compat_registry.register_reader(fmt, StateVectorDict, read_statevectordict)
    compat_registry.register_reader(fmt, StateVector, read_statevector)
    compat_registry.register_writer(fmt, TimeSeriesDict, write_timeseriesdict)
    compat_registry.register_writer(fmt, TimeSeries, write_timeseries)
    compat_registry.register_writer(fmt, StateVectorDict, write_timeseriesdict)
    compat_registry.register_writer(fmt, StateVector, write_timeseries)


# -- generic API for 'gwf' format ---------------------------------------------

def register_gwf_format(container):
    """Register I/O methods for `format='gwf'`

    The created methods loop through the registered sub-formats.

    Parameters
    ----------
    container : `Series`, `dict`
        series class or series dict class to register
    """
    def _backend():
        return io_gwf.get_backend(
            package=__package__,
            backends=BACKENDS,
        )

    def read_(*args, **kwargs):
        fmt = f"gwf.{_backend().lower()}"
        reader = get_reader(fmt, container)
        return reader(*args, **kwargs)

    def write_(*args, **kwargs):
        fmt = f"gwf.{_backend().lower()}"
        writer = get_writer(fmt, container)
        return writer(*args, **kwargs)

    compat_registry.register_identifier('gwf', container, io_gwf.identify_gwf)
    compat_registry.register_reader('gwf', container, read_)
    compat_registry.register_writer('gwf', container, write_)


# -- register frame API -------------------------------------------------------

# register 'gwf.<backend>' sub-format for each backend
for backend in BACKENDS:
    register_gwf_backend(backend)

# register generic 'gwf' format for each container
for container in (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict):
    register_gwf_format(container)
