# Copyright (C) Cardiff University (2025-)
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

"""Unified I/O for :mod:`gwpy.timeseries` objects."""

from __future__ import annotations

import typing

from ..io import cache as io_cache
from ..io.registry import (
    UnifiedRead,
)
from ..types.connect import SeriesWrite

if typing.TYPE_CHECKING:
    from ...time._tconvert import GpsConvertible
    from ...types import Series


# -- utilities -----------------------

def _join_factory(
    cls: type,
    gap: str | None,
    pad: float | None,
    start: GpsConvertible | None = None,
    end: GpsConvertible | None = None,
):
    """Build a joiner for the given cls, and the given padding options."""
    from . import TimeSeriesBaseDict

    if issubclass(cls, TimeSeriesBaseDict):
        def _join(data):
            out = cls()
            data = list(data)
            while data:
                tsd = data.pop(0)
                out.append(tsd, gap=gap, pad=pad)
                del tsd
            if gap in ("pad", "raise"):
                for key in out:
                    out[key] = _pad_series(
                        out[key],
                        pad,
                        start,
                        end,
                        error=(gap == "raise"),
                    )
            return out
    else:
        from . import TimeSeriesBaseList

        def _join(arrays):
            if len(arrays) == 1:  # don't copy a single array
                joined = arrays[0]
            else:
                list_ = TimeSeriesBaseList(*arrays)
                joined = list_.join(pad=pad, gap=gap)
            if gap in ("pad", "raise"):
                return _pad_series(
                    joined,
                    pad,
                    start,
                    end,
                    error=(gap == "raise"),
                )
            return joined
    return _join


def _pad_series(
    ts: Series,
    pad: float | None,
    start: GpsConvertible | None = None,
    end: GpsConvertible | None = None,
    error: bool = False,
):
    """Pad a timeseries to match the specified [start, end) limits.

    To cover a gap in data returned from a data source.

    Parameters
    ----------
    ts : `gwpy.types.Series`
        The input series.

    pad : `float`, `astropy.units.Quantity`
        The value with which to pad.

    start : `float`, `astropy.units.Quantity`, optional
        The desired start point of the X-axis, defaults to
        the start point of the incoming series.

    end : `float`, `astropy.units.Quantity`, optional
        The desired end point of the X-axis, defaults to
        the end point of the incoming series.

    error : `bool`, optional
        Raise `ValueError` when gaps are present, rather than padding
        anything.

    Returns
    -------
    series : instance of incoming series type
        A padded version of the series. This may be the same
        object if not padding is needed.

    Raises
    ------
    ValueError
        If `error=True` is given and padding would have been required
        to match the request.
    """
    span = ts.span
    if start is None:
        start = span[0]
    if end is None:
        end = span[1]
    pada = max(int((span[0] - start) * ts.sample_rate.value), 0)
    padb = max(int((end - span[1]) * ts.sample_rate.value), 0)
    if not (pada or padb):  # if noop, just return the input
        return ts
    if error:  # if error, bail out now
        raise ValueError(
            f"{type(ts).__name__} with span {span} does not cover "
            "requested interval {type(span)(start, end)}",
        )
    # otherwise applying the padding
    return ts.pad((pada, padb), mode="constant", constant_values=(pad,))


# -- Unified I/O singletons ----------

class _TimeSeriesRead(UnifiedRead):
    """Base `UnifiedRead` implementation.

    This defines the logic, but all concrete instances should be created
    from a subclass that gives a detailed docstring to support
    `Klass.read.help()`.
    """
    def __call__(  # type: ignore[override]
        self,
        source,
        *args,
        start: GpsConvertible | None = None,
        end: GpsConvertible | None = None,
        pad: float | None = None,
        gap: str | None = None,
        **kwargs,
    ):
        # if reading a cache, read it now and sieve
        if io_cache.is_cache(source):
            from .io.cache import preformat_cache
            source = preformat_cache(
                source,
                start=start,
                end=end,
            )

        # get join arguments
        if gap is None:
            gap = "raise" if pad is None else "pad"
        joiner = _join_factory(
            self._cls,
            gap,
            pad,
            start=start,
            end=end,
        )

        # read
        return super().__call__(
            joiner,
            source,
            *args,
            start=start,
            end=end,
            **kwargs,
        )


# -- TimeSeriesBase -----------------

class TimeSeriesBaseRead(_TimeSeriesRead):
    pass


class TimeSeriesBaseWrite(SeriesWrite):
   pass


# -- TimeSeries ---------------------

class TimeSeriesRead(TimeSeriesBaseRead):
    """Read data into a `TimeSeries`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the parameters
    below are common to all formats.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    format : `str`, optional
        source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""


class TimeSeriesWrite(TimeSeriesBaseWrite):
    """Write this `TimeSeries` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        output filename

    format : `str`, optional
        output format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    Notes
    -----"""


# -- StateVector --------------------

class StateVectorRead(TimeSeriesBaseRead):
    """Read data into a `StateVector`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the parameters
    below are common to most formats.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    name : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    bits : `list`, optional
        List of bits names for this `StateVector`, give `None` at
        any point in the list to mask that bit.

    format : `str`, optional
        source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""


class StateVectorWrite(SeriesWrite):
    """Write this `StateVector` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        output filename

    format : `str`, optional
        output format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    Notes
    -----"""


# -- TimeSeriesBaseDict -------------

class TimeSeriesBaseDictRead(_TimeSeriesRead):
    pass


class TimeSeriesBaseDictWrite(SeriesWrite):
    pass


# -- TimeSeriesDict -----------------

class TimeSeriesDictRead(TimeSeriesBaseDictRead):
    """Read data into a `TimeSeriesDict`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the parameters
    below are common to most formats.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    name : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    format : `str`, optional
        source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""


class TimeSeriesDictWrite(SeriesWrite):
    """Write this `TimeSeriesDict` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        output filename

    format : `str`, optional
        output format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    Notes
    -----"""


# -- StateVectorDict ----------------

class StateVectorDictRead(TimeSeriesBaseDictRead):
    """Read data into a `StateVectorDict`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the parameters
    below are common to most formats.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    name : `str`, `~gwpy.detector.Channel`
        the name of the channel to read, or a `Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine

    format : `str`, optional
        source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    gap : `str`, optional
        How to handle gaps in the data, one of

        "ignore"
            Do nothing, let the underlying reader method handle it.

        "warn"
            Do nothing except print a warning to the screen.

        "raise"
            Raise an exception upon finding a gap (default).

        "pad"
            Insert a value to fill the gaps.

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""


class StateVectorDictWrite(SeriesWrite):
    """Write this `StateVectorDict` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        output filename

    format : `str`, optional
        output format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    Notes
    -----"""
