# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

The GWF format is defined in :dcc:`LIGO-T970130`.

The functions in this module are the ones exposed to the user, but
connect to the backend-specific functions in the other modules in
the same subpackage.
"""

from __future__ import annotations

import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

from ....io import (
    cache as io_cache,
    gwf as io_gwf,
)
from ....io.utils import (
    FileLike,
    file_path,
)
from ....segments import Segment
from ....time import to_gps
from ....utils.decorators import deprecated_function
from ... import (
    StateVector,
    StateVectorDict,
    TimeSeries,
    TimeSeriesDict,
)
from .utils import _channel_dict_kwarg

if TYPE_CHECKING:
    from typing import IO

    from ....time import SupportsToGps
    from ... import (
        Bits,
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# preferentially ordered list of GWF backends
BACKENDS: list[str] = [
    "frameCPP",
    "FrameL",
    "LALFrame",
]


# -- read ----------------------------

def read_timeseriesdict(
    source: str | list[str],
    channels: list[str],
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    scaled: bool | None = None,
    type: str | dict[str, str] | None = None,
    backend: str | None = None,
    series_class: type[TimeSeriesBase] = TimeSeries,
) -> TimeSeriesBaseDict:
    """Read the data for a list of channels from a GWF data source.

    Parameters
    ----------
    source : `str`, `Path`, `file`, or `list`
        Source of data, any of the following:

        - path to a single GWF file
        - path to a cache file
        - `list` of GWF file paths.

    channels : `list`
        List of data channel names (or `Channel` objects) to read from GWF.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data.

    type : `dict`, optional
        A `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    backend : `str`, optional
        GWF backend to use when reading.
        Default is the value of the ``GWPY_FRAME_LIBRARY`` environment
        variable, or 'frameCPP'.

    series_class : `type[TimeSeriesBase]`, optional
        The type to use for all series objects.
        Default is the class being used for `.read()`.

    Returns
    -------
    dict : :class:`~gwpy.timeseries.TimeSeriesDict`
        dict of (channel, `TimeSeries`) data pairs
    """
    read_func = io_gwf.get_backend_function(
        "read",
        backend=backend,
        backends=BACKENDS,
        package=__package__,
    )

    # -- from here read data

    if start:
        start = to_gps(start)
    if end:
        end = to_gps(end)

    # read cache file up-front
    if (
        (isinstance(source, str) and source.endswith((".lcf", ".cache")))
        or (
            isinstance(source, FileLike)
            and hasattr(source, "name")
            and source.name.endswith((".lcf", ".cache"))
        )
    ):
        source = io_cache.read_cache(source)
    # separate cache into contiguous segments
    if io_cache.is_cache(source):
        if start is not None and end is not None:
            source = io_cache.sieve(
                source,
                segment=Segment(start, end),
            )
        source = list(io_cache.find_contiguous(source))
    # convert everything else into a list if needed
    if not isinstance(source, list | tuple):
        source = [source]

    # now read the data
    return read_func(
        source,
        channels,
        start=start,
        end=end,
        scaled=scaled,
        type=type,
        series_class=series_class,
    )


def read_timeseries(
    source: str | list[str],
    channel: str,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    scaled: bool | None = None,
    type: str | dict[str, str] | None = None,
    backend: str | None = None,
    series_class: type[TimeSeriesBase] = TimeSeries,
) -> TimeSeries | StateVector:
    """Read a `TimeSeriesa from one or more GWF files.

    Parameters
    ----------
    source : `str`, `Path`, `file`, or `list`
        Source of data, any of the following:

        - path to a single GWF file
        - path to a cache file
        - `list` of GWF file paths.

    channel : `str`, `~gwpy.detector.Channel`
        Name of data channel to read from GWF.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data.

    type : `dict`, optional
        A `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    backend : `str`, optional
        GWF backend to use when reading.
        Default is 'frameCPP'.

    series_class : `type[TimeSeriesBase]`, optional
        The type to use for all series objects.
        Default is the class being used for `.read()`.

    Returns
    -------
    series : `~gwpy.timeseries.TimeSeries`
        A `TimeSeries` containing the data read from GWF.

    Raises
    ------
    ValueError
        If the given channel name isn't found, or the ``[start, end)``
        interval doesn't match the data available from the GWF source.
    """
    return read_timeseriesdict(
        source,
        [channel],
        start=start,
        end=end,
        scaled=scaled,
        type=type,
        backend=backend,
        series_class=series_class,
    )[channel]


def read_statevector(
    source: str | list[str],
    channel: str,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    bits: list[str] | Bits | None = None,
    scaled: bool | None = None,
    type: str | dict[str, str] | None = None,
    backend: str | None = None,
    series_class: type[StateVector] = StateVector,
) -> StateVector:
    """Read a `StateVector` from one or more GWF files.

    Parameters
    ----------
    source : `str`, `Path`, `file`, or `list`
        Source of data, any of the following:

        - path to a single GWF file
        - path to a cache file
        - `list` of GWF file paths.

    channel : `str`, `~gwpy.detector.Channel`
        Name of data channel to read from GWF.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    bits : `Bits`, `list`, optional
        List of bits defining this `StateVector`.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data.

    type : `dict`, optional
        A `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    backend : `str`, optional
        GWF backend to use when reading.
        Default is 'frameCPP'.

    series_class : `type[TimeSeriesBase]`, optional
        The type to use for all series objects.
        Default is the class being used for `.read()`.

    Returns
    -------
    statevector : `~gwpy.timeseries.StateVector`
        A `StateVector` containing the data read from GWF.

    Raises
    ------
    ValueError
        If the given channel name isn't found, or the ``[start, end)``
        interval doesn't match the data available from the GWF source.
    """
    statevector = read_timeseries(
        source,
        channel,
        start=start,
        end=end,
        scaled=scaled,
        type=type,
        backend=backend,
        series_class=series_class,
    )
    statevector.bits = bits
    return statevector


def read_statevectordict(
    source: str | list[str],
    channels: list[str],
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    bits: list[str] | Bits | None = None,
    scaled: bool | None = None,
    type: str | dict[str, str] | None = None,
    backend: str | None = None,
    series_class: type[StateVector] = StateVector,
) -> StateVectorDict:
    """Read a `StateVectorDict` from GWF files.

    Parameters
    ----------
    source : `str`, `Path`, `file`, or `list`
        Source of data, any of the following:

        - path to a single GWF file
        - path to a cache file
        - `list` of GWF file paths.

    channels : `list`
        List of data channel names (or `Channel` objects) to read from GWF.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    bits : `dict` of ``(str, bits)`` pairs, `list`
        A `dict` giving the bit definition for each channel, or a `list`
        given the bits to assign identically to all channels.

    scaled : `bool`, optional
        Apply slope and bias calibration to ADC data.

    type : `dict`, optional
        A `dict` of ``(name, channel-type)`` pairs, where ``channel-type``
        can be one of ``'adc'``, ``'proc'``, or ``'sim'``.

    backend : `str`, optional
        GWF backend to use when reading.
        Default is the value of the ``GWPY_FRAME_LIBRARY`` environment
        variable, or 'frameCPP'.

    series_class : `type[TimeSeriesBase]`, optional
        The type to use for all series objects.
        Default is the class being used for `.read()`.

    Returns
    -------
    dict : :class:`~gwpy.timeseries.TimeSeriesDict`
        dict of (channel, `TimeSeries`) data pairs
    """
    # read data
    svd = StateVectorDict(read_timeseriesdict(
        source,
        channels,
        start=start,
        end=end,
        scaled=scaled,
        type=type,
        backend=backend,
        series_class=series_class,
    ))

    # add bits definitions
    bitss = _channel_dict_kwarg(
        bits,
        channels,
        list,
        varname="bits",
    )
    for (channel, bits_) in bitss.items():
        svd[channel].bits = bits_
    return svd


# -- write ----------------------------------

def write_timeseriesdict(
    seriesdict: TimeSeriesBaseDict,
    target: str | Path | IO,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    type: str | None = None,
    name: str | None = None,
    run: int = 0,
    compression: str | int | None = None,
    compression_level: int | None = None,
    backend: str | None = None,
    *,
    overwrite: bool = True,
    append: bool = False,
) -> None:
    """Write a `TimeSeriesDict` in GWF format.

    Parameters
    ----------
    seriesdict : `TimeSeriesDict`
        The data to write.

    target : `str`, `pathlib.Path`, `file`
        The target file or path to write to.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        The GPS start time (``GTime``) of the output ``FrameH`` structure.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        The GPS end time of the output ``FrameH``.
        Used with ``start`` to set the ``Dt`` attribute.

    type : `str`, optional
        The type of the channel, one of 'adc', 'proc', 'sim'.
        Default is 'proc' unless stored in the ``_ctype`` attribute
        of the channel structure.

    name : `str`, optional
        The name of the frame.
        Default is ``"gwpy"`` when using FrameCPP and LALFrame backends,
        or the name of the first channel in the dict when writing with
        FrameL.

        This keyword is not supported when writing with FrameL.

    run : `int`, optional
        The ``FrameH`` run number.
        Values less than ``0`` are reserved for simulated data, should be a
        monotonically increasing integer for experimental runs.

        This keyword is not supported when writing with FrameL.

    compression : `int`, `str`, optional
        Name of compresion algorithm to use, or its endian-appropriate
        ID. One of:

        - ``'RAW'``
        - ``'GZIP'`` (default)
        - ``'DIFF_GZIP'``
        - ``'ZERO_SUPPRESS'``
        - ``'ZERO_SUPPRESS_OTHERWISE_GZIP'``

        Only ``'GZIP'`` is supported when writing with FrameL.

    compression_level : `int`, optional
        Compression level for given method, default is ``6`` for GZIP-based
        methods, otherwise ``0``.

    backend : `str`, optional
        GWF backend to use when reading.
        Default is 'frameCPP'.

    overwrite : `bool`, optional
        If `True` (default), overwrite existing files.

    append : `bool`, optional
        DO NOT USE. This option is not supported for writing GWF files;
        the entire file will be overwritten.
        This option is only present for API consistency with other
        GWpy I/O functions.

    Raises
    ------
    OSError
        If the target file exists and neither ``overwrite`` nor ``append``
        is `True`.
    """
    try:
        target_path = Path(file_path(target))
    except ValueError:
        # not a file path, so cannot check for existence, that's fine
        pass
    else:
        _exists = target_path.exists()
        if _exists and not overwrite:
            msg = f"File exists: {target_path}"
            raise OSError(msg)
        if _exists and append:
            warnings.warn(
                "append=True is not supported for writing GWF files; "
                "the entire file will be overwritten.",
                category=UserWarning,
                stacklevel=2,
            )

    # get backend function
    write_func = io_gwf.get_backend_function(
        "write",
        backend=backend,
        backends=BACKENDS,
        package=__package__,
    )

    # pre-format GPS times
    span = seriesdict.span
    if start is None:
        startgps = span[0]
    else:
        startgps = to_gps(start)
    if end is None:
        endgps = span[1]
    else:
        endgps = to_gps(end)

    write_func(
        seriesdict,
        target,
        startgps,
        endgps,
        type=type,
        name=name,
        run=run,
        compression=compression,
        compression_level=compression_level,
    )


def write_timeseries(
    series: TimeSeriesBase,
    target: str | Path | IO,
    start: SupportsToGps | None = None,
    end: SupportsToGps | None = None,
    type: str | None = None,
    name: str | None = None,
    run: int = 0,
    compression: str | int | None = None,
    compression_level: int | None = None,
    backend: str | None = None,
    *,
    overwrite: bool = True,
    append: bool = False,
) -> None:
    """Write a `TimeSeries` to disk in GWF format.

    Parameters
    ----------
    series : `TimeSeries`
        The data to write.

    target : `str`, `pathlib.Path`, `file`
        The target file or path to write to.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        The GPS start time (``GTime``) of the output ``FrameH`` structure.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        The GPS end time of the output ``FrameH``.
        Used with ``start`` to set the ``Dt`` attribute.

    type : `str`, optional
        The type of the channel, one of 'adc', 'proc', 'sim'.
        Default is 'proc' unless stored in the ``_ctype`` attribute
        of the channel structure.

    name : `str`, optional
        The name of the frame.
        Default is ``"gwpy"`` when using FrameCPP and LALFrame backends,
        or the name of the first channel in the dict when writing with
        FrameL.

        This keyword is not supported when writing with FrameL.

    run : `int`, optional
        The ``FrameH`` run number.
        Values less than ``0`` are reserved for simulated data, should be a
        monotonically increasing integer for experimental runs.

        This keyword is not supported when writing with FrameL.

    compression : `int`, `str`, optional
        Name of compresion algorithm to use, or its endian-appropriate
        ID. One of:

        - ``'RAW'``
        - ``'GZIP'`` (default)
        - ``'DIFF_GZIP'``
        - ``'ZERO_SUPPRESS'``
        - ``'ZERO_SUPPRESS_OTHERWISE_GZIP'``

        Only ``'GZIP'`` is supported when writing with FrameL.

    compression_level : `int`, optional
        Compression level for given method, default is ``6`` for GZIP-based
        methods, otherwise ``0``.


    backend : `str`, optional
        GWF backend to use when reading.
        Default is 'frameCPP'.

    overwrite : `bool`, optional
        If `True` (default), overwrite existing files.

    append : `bool`, optional
        DO NOT USE. This option is not supported for writing GWF files;
        the entire file will be overwritten.
        This option is only present for API consistency with other
        GWpy I/O functions.

    Raises
    ------
    OSError
        If the target file exists and neither ``overwrite`` nor ``append``
        is `True`.
    """
    write_timeseriesdict(
        series.DictClass({None: series}),
        target=target,
        start=start,
        end=end,
        type=type,
        name=name,
        run=run,
        compression=compression,
        compression_level=compression_level,
        backend=backend,
        overwrite=overwrite,
        append=append,
    )


# -- register -------------------------------

def register_gwf_backend(backend: str | None) -> None:
    """Register a full set of GWF I/O methods for the given backend.

    The `timeseries.io.gwf` backend module must define the following methods:

    - `read` : which receives one of more frame files which can be assumed
               to be contiguous, and should return a `TimeSeriesDict`
    - `write` : which receives an output frame file path and a `TimeSeriesDict`
                and does all of the work

    Parameters
    ----------
    backend : `str`
        The name of the backend to register.
        Also accepts `None` to register the ``"gwf"`` format without a
        specific backend; this is applied automatically when this module is
        loaded.
    """
    if backend:
        if backend not in BACKENDS:  # allow self-registration
            BACKENDS.append(backend)
        fmt = f"gwf.{backend.lower()}"
    else:
        fmt = "gwf"

    for klass, reader, writer in (
        (TimeSeries, read_timeseries, write_timeseries),
        (TimeSeriesDict, read_timeseriesdict, write_timeseriesdict),
        (StateVector, read_statevector, write_timeseries),
        (StateVectorDict, read_statevectordict, write_timeseriesdict),
    ):
        if backend:
            message = (
                f"`format='{fmt}` is deprecated, "
                f"please instead use `format='gwf', backend='{backend}'"
            )
            read_ = deprecated_function(
                partial(reader, backend=backend),
                message=message,
            )
            write_ = deprecated_function(
                partial(writer, backend=backend),
                message=message,
            )
        else:
            read_ = reader
            write_ = writer

        klass.read.registry.register_reader(fmt, klass, read_)
        klass.write.registry.register_writer(fmt, klass, write_)
        if backend is None:
            klass.read.registry.register_identifier(
                fmt,
                klass,
                io_gwf.identify_gwf,
            )


# register for timeseries objects
register_gwf_backend(None)  # format='gwf'
