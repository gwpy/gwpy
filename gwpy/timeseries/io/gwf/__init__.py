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
import warnings

from six import string_types

import numpy

from astropy.io.registry import (get_formats, get_reader, get_writer)

try:
    from glue.lal import Cache
except ImportError:  # no lal
    HAS_CACHE = False
else:
    HAS_CACHE = True

from ....segments import Segment
from ....time import to_gps
from ....io.gwf import identify_gwf
from ....io.cache import (FILE_LIKE, read_cache as read_cache_file,
                          find_contiguous)
from ....io.registry import (register_reader,
                             register_writer,
                             register_identifier)
from ... import (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# list of available APIs
# --- read/write operations using format='gwf' will iterate in order
# --- through this list, using whichever one imports correctly first
APIS = [
    'framecpp',
    'lalframe',
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


def import_gwf_library(library, package=__package__):
    """Utility method to import the relevant timeseries.io.gwf frame API

    This is just a wrapper around :meth:`importlib.import_module` with
    a slightly nicer error message
    """
    # import the frame library here to have any ImportErrors occur early
    try:
        return importlib.import_module('.%s' % library, package=package)
    except ImportError as exc:
        exc.args = ('Cannot import %s frame API: %s' % (library, str(exc)),)
        raise


def get_default_gwf_api():
    """Return the preferred GWF library

    Examples
    --------
    If you have |LDAStools.frameCPP|_ installed:

    >>> from gwpy.timeseries.io.gwf import get_default_gwf_api
    >>> get_default_gwf_api()
    'framecpp'

    Or, if you don't have |lalframe|_:

    >>> get_default_gwf_api()
    'lalframe'

    Otherwise:

    >>> get_default_gwf_api()
    ImportError: no GWF API available, please install a third-party GWF library (framecpp, lalframe) and try again
    """
    for lib in APIS:
        try:
            import_gwf_library(lib)
        except ImportError as e:
            continue
        else:
            return lib
    raise ImportError("no GWF API available, please install a third-party GWF "
                      "library ({}) and try again".format(', '.join(APIS)))


# -- generic I/O methods ------------------------------------------------------

def register_gwf_api(library):
    """Register a full set of GWF I/O methods for the given library

    The given frame library must define the following methods

    - `read` : which receives one of more frame files which can be assumed
               to be contiguous, and should return a `TimeSeriesDict`
    - `write` : which receives an output frame file path an a `TimeSeriesDict`
                and does all of the work

    Additionally, the library must store the name of the third-party
    dependency using the ``FRAME_LIBRARY`` variable.
    """
    # import library to get details (don't require library to importable)
    try:
        lib = import_gwf_library(library)
    except ImportError:
        pass  # means any reads will fail at run-time
    else:
        libread_ = lib.read
        libwrite_ = lib.write

    # set I/O format name
    fmt = 'gwf.%s' % library

    # -- read -----------------------------------

    def read_timeseriesdict(source, channels, start=None, end=None,
                            dtype=None, resample=None,
                            gap=None, pad=None, nproc=1,
                            series_class=TimeSeries, **kwargs):
        """Read the data for a list of channels from a GWF data source

        Parameters
        ----------
        source : `str`, :class:`glue.lal.Cache`, `list`
            data source object, one of:

            - `str` : frame file path
            - :class:`glue.lal.Cache`, `list` : contiguous list of frame paths

        channels : `list`
            list of channel names (or `Channel` objects) to read from frame.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data, defaults to start of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        dtype : `numpy.dtype`, `str`, `type`, or `dict`, optional
            desired numeric data type for returned data, or
            `dict` of ``(channel, dtype)`` pairs

        resample : `float`, `dict`, optional
            desired output sample rate for all returned data, or
            `dict` of ``(channel, rate)`` pairs

        pad : `float`, optional
            value with which to fill gaps in the source data, if not
            given gaps will result in an exception being raised

        Returns
        -------
        dict : :class:`~gwpy.timeseries.TimeSeriesDict`
            dict of (channel, `TimeSeries`) data pairs
        """
        # import the frame library here to have any ImportErrors occur early
        import_gwf_library(library)

        # -- from here read data

        if start:
            start = float(to_gps(start))
        if end:
            end = float(to_gps(end))

        # parse output format kwargs
        if not isinstance(resample, dict):
            resample = dict((c, resample) for c in channels)
        if not isinstance(dtype, dict):
            dtype = dict((c, dtype) for c in channels)

        # format gap handling
        if gap is None and pad is not None:
            gap = 'pad'
        elif gap is None:
            gap = 'raise'

        # read cache file up-front
        if (isinstance(source, string_types) and
                source.endswith(('.lcf', '.cache'))) or (
                    isinstance(source, FILE_LIKE) and
                    source.name.endswith(('.lcf', '.cache'))):
            source = read_cache_file(source)
        # separate cache into contiguous segments
        if HAS_CACHE and isinstance(source, Cache):
            if start is not None and end is not None:
                source = source.sieve(segment=Segment(start, end))
            source = list(find_contiguous(source))
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

        # apply resampling and dtype-casting
        for name in out:
            if (resample.get(name) and
                    resample[name] != out[name].sample_rate.value):
                out[name] = out[name].resample(resample[name])
            if dtype.get(name) is not None and (
                    numpy.dtype(dtype[name]) != out[name].dtype):
                out[name] = out[name].astype(dtype[name])
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

    def write_timeseriesdict(data, outfile, start=None, end=None,
                             name='gwpy', run=0):
        """Write a `TimeSeriesDict` to disk in GWF format

        Parameters
        ----------
        tsdict : `TimeSeriesDict`
            the data to write

        outfile : `str`
            the path of the output frame file

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        name : `str`, optional
            name of the project that created this GWF

        run : `int`, optional
            run number to write into frame header
        """
        # import the frame library here to have any ImportErrors occur early
        import_gwf_library(library)

        # then write using the relevant API
        return libwrite_(data, outfile, start=start, end=end,
                         name=name, run=run)

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
    register_reader(fmt, TimeSeriesDict, read_timeseriesdict)
    register_reader(fmt, TimeSeries, read_timeseries)
    register_reader(fmt, StateVectorDict, read_statevectordict)
    register_reader(fmt, StateVector, read_statevector)
    register_writer(fmt, TimeSeriesDict, write_timeseriesdict)
    register_writer(fmt, TimeSeries, write_timeseries)
    register_writer(fmt, StateVectorDict, write_timeseriesdict)
    register_writer(fmt, StateVector, write_timeseries)

    # register deprecated format - DEPRECATED
    for container in (TimeSeries, TimeSeriesDict,
                      StateVector, StateVectorDict):
        register_library_format(container, library)


# -- generic API for 'gwf' format ---------------------------------------------

def register_gwf_format(container):
    """Register I/O methods for `format='gwf'`

    The created methods loop through the registered sub-formats.

    Parameters
    ----------
    container : `Series`, `dict`
        series class or series dict class to register
    """
    def read_(*args, **kwargs):
        fmt = 'gwf.{}'.format(get_default_gwf_api())
        reader = get_reader(fmt, container)
        return reader(*args, **kwargs)

    def write_(*args, **kwargs):
        fmt = 'gwf.{}'.format(get_default_gwf_api())
        writer = get_writer(fmt, container)
        return writer(*args, **kwargs)

    register_identifier('gwf', container, identify_gwf)
    register_reader('gwf', container, read_)
    register_writer('gwf', container, write_)


# -- DEPRECATED - register old format name ------------------------------------

def register_library_format(container, library):
    """Register methods for the given library format

    E.g. for lalframe this functions creates methods and registers them
    for the ``lalframe`` format name.

    This format has been deprecated and will be removed prior to the 1.0
    release in favour of the ``gwf.<library>`` contention. All this method
    does is create directes from `format='<library'` to
    `format=gwf.<library>'`.

    Parameters
    ----------
    container : `Series`, `dict`
        series class or series dict class to register

    library : `str`
        name of frame library
    """
    fmt = 'gwf.%s' % library
    reader = get_reader(fmt, container)
    writer = get_writer(fmt, container)

    def read_(*args, **kwargs):
        warnings.warn("Reading with format=%r is deprecated and will be "
                      "disabled in an upcoming release, please use "
                      "format=%r instead" % (library, fmt),
                      DeprecationWarning)
        return reader(*args, **kwargs)

    def write_(*args, **kwargs):
        warnings.warn("Writing with format=%r is deprecated and will be "
                      "disabled in an upcoming release, please use "
                      "format=%r instead" % (library, fmt),
                      DeprecationWarning)
        return writer(*args, **kwargs)

    register_reader(library, container, read_)
    register_writer(library, container, write_)


# -- register frame API -------------------------------------------------------

# register 'gwf.<api>' sub-format for each API
for api in APIS:
    register_gwf_api(api)

# register generic 'gwf' format for each container
for container in (TimeSeries, TimeSeriesDict, StateVector, StateVectorDict):
    register_gwf_format(container)
