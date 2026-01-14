# Copyright (c) 2025 Cardiff University
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

from abc import abstractmethod
from typing import TYPE_CHECKING

from ..io import cache as io_cache
from ..io.registry import (
    UnifiedGet,
    UnifiedRead,
)
from ..time import to_gps
from ..types.connect import SeriesWrite

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import (
        Any,
        Literal,
        SupportsFloat,
        TypeVar,
    )

    from numpy.typing import DTypeLike

    from ..detector import Channel
    from ..io.utils import NamedReadable
    from ..time import SupportsToGps
    from . import (
        TimeSeriesBase,
        TimeSeriesBaseDict,
    )

    T = TypeVar("T", bound=TimeSeriesBase | TimeSeriesBaseDict)
    TimeSeriesType = TypeVar("TimeSeriesType", bound=TimeSeriesBase)
    TimeSeriesDictType = TypeVar("TimeSeriesDictType", bound=TimeSeriesBaseDict)


# -- utilities -----------------------

def _pad_series(
    ts: TimeSeriesType,
    pad: float | None,
    start: SupportsFloat | None = None,
    end: SupportsFloat | None = None,
    *,
    error: bool = False,
) -> TimeSeriesType:
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
        msg = (
            f"{type(ts).__name__} with span {span} does not cover "
            f"requested interval {type(span)(start, end)}"
        )
        raise ValueError(msg)
    # otherwise applying the padding
    return ts.pad((pada, padb), mode="constant", constant_values=(pad,))


# -- Unified I/O singletons ----------

class _TimeSeriesRead(UnifiedRead):
    """Base `TimeSeriesRead` implementation.

    This defines the logic, but all concrete instances should be created
    from a subclass that gives a detailed docstring to support
    `Klass.read.help()`.
    """

    @abstractmethod
    def merge(  # type: ignore[override]
        self,
        items: Sequence[T],
        pad: float | None = None,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        start: SupportsFloat | None = None,
        end: SupportsFloat | None = None,
    ) -> T:
        """Combine a list of `TimeSeries` or `TimeSeriesDict` into one.

        Must be given at least one item.
        """

    def __call__(  # type: ignore[override]
        self,
        source: NamedReadable | Sequence[NamedReadable],
        name: str | Channel | None = None,
        start: SupportsToGps | None = None,
        end: SupportsToGps | None = None,
        *,
        pad: float | None = None,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        **kwargs,
    ) -> TimeSeriesBase:
        """Read a `TimeSeries` from a source."""
        if start is not None:
            start = to_gps(start)
        if end is not None:
            end = to_gps(end)

        # if reading a cache, read it now and sieve
        if io_cache.is_cache(source):
            from .io.cache import preformat_cache
            source = preformat_cache(
                source,
                start=start,
                end=end,
            )

        # construct parametrised merge function
        # (this allows pad, gap, start, end to be passed into
        #  the underlying read function)
        if gap is None:
            gap = "raise" if pad is None else "pad"
        def merge(items: Sequence[T]) -> T:
            return self.merge(
                items,
                pad=pad,
                gap=gap,
                start=start,
                end=end,
            )

        # read (with optional name)
        args = () if name is None else (name,)
        return super().__call__(
            source,
            *args,
            start=start,
            end=end,
            merge_function=merge,
            **kwargs,
        )


# -- TimeSeriesBase -----------------

class TimeSeriesBaseRead(_TimeSeriesRead):
    """Read data into a `TimeSeriesBase`.

    Notes
    -----"""

    def merge(  # type: ignore[override]
        self,
        items: Sequence[TimeSeriesBase],
        pad: float | None = None,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        start: SupportsFloat | None = None,
        end: SupportsFloat | None = None,
    ) -> TimeSeriesBase:
        """Combine a list of `TimeSeriesBase` objects into one `Series`."""
        from . import TimeSeriesBaseList
        if len(items) == 1:  # don't copy a single array
            joined = items[0]
        else:
            list_ = TimeSeriesBaseList(*items)
            joined = list_.join(pad=pad, gap=gap)
        if gap in ("pad", "raise"):
            return _pad_series(
                joined,
                pad,
                start,
                end,
                error=(gap == "raise"),
            )
        if not isinstance(joined, self._cls):
            joined = self._cls(joined)
        return joined


class TimeSeriesBaseWrite(SeriesWrite):
    """Write data from a `TimeSeriesBase`.

    Notes
    -----"""


class TimeSeriesBaseGet(UnifiedGet):
    """Get data from any source.

    Notes
    -----"""

    def __call__(  # type: ignore[override]
        self,
        name: str | Channel,
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        source: str | list[str | dict[str, Any]] | None = None,
        **kwargs,
    ) -> TimeSeriesBase:
        """Retrieve data from a source."""
        dict_method = getattr(self._cls.DictClass, self.method)
        kwargs.setdefault("series_class", self._cls)
        return dict_method(
            [name],
            start,
            end,
            source=source,
            **kwargs,
        )[name]


# -- TimeSeries ---------------------

class TimeSeriesRead(TimeSeriesBaseRead):
    """Read data into a `TimeSeries`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the parameters
    below are common to all formats.

    Parameters
    ----------
    source : `str`, `os.PathLike`, `file`, or `list` of these
        Source of data, any of the following:

        - Path of a single data file
        - List of data file paths
        - Path of LAL-format cache file

    name : `str`, optional
        The name of the data object to read, or a `Channel` object.
        This argument is required if ``source`` contains (or may contain)
        multiple datasets.

        - When reading from GWF, this argument should specify the
          name of the channel to read.
        - When reading from HDF5, this argument should specify the
          path or dataset name within the file.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        Number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        Value with which to fill gaps in the source data,
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


class TimeSeriesGet(TimeSeriesBaseGet):
    """Retrieve data for a channel from any data source.

    This method attempts to get data any way it can, potentially iterating
    over multiple available data sources.

    Parameters
    ----------
    channels : `list`
        Required data channels.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    source : `str`, `list`, `list` of `dict`.
        The data source to use. One of the following formats:

        - `str` - the name of a single source to use,
        - `list` - a list of source names to try in order,
        - `list` of `dict` - a list of source specifications to try in order;
          each `dict` must contain a `"source"` key giving the name of the
          source to use, and may contain other keys giving options to
          pass to the data access function for that source.

        See 'Notes' section below for valid source names.

    frametype : `str`
        Name of frametype in which this channel is stored, by default
        will search for all required frame types.

    pad : `float`
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    scaled : `bool`
        apply slope and bias calibration to ADC data, for non-ADC data
        this option has no effect.

    nproc : `int`, default: `1`
        Number of parallel processes to use, serial process by
        default.

    allow_tape : `bool`, default: `None`
        Allow the use of data files that are held on tape.
        Default is `None` to attempt to allow the `TimeSeries.fetch`
        method to intelligently select a server that doesn't use tapes
        for data storage (doesn't always work), but to eventually allow
        retrieving data from tape if required.

    verbose : `bool`
        This argument is deprecated and will be removed in a future release.
        Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

    kwargs
        Other keyword arguments to pass to the data access function for
        each data source.

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
    source : `str`, `os.PathLike`, `file`, or `list` of these
        Source of data, any of the following:

        - Path of a single data file
        - List of data file paths
        - Path of LAL-format cache file

    name : `str`, optional
        The name of the data object to read, or a `Channel` object.
        This argument is required if ``source`` contains (or may contain)
        multiple datasets.

        - When reading from GWF, this argument should specify the
          name of the channel to read.
        - When reading from HDF5, this argument should specify the
          path or dataset name within the file.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    bits : `list`, optional
        List of bits names for this `StateVector`, give `None` at
        any point in the list to mask that bit.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        Number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    Raises
    ------
    IndexError
        If ``source`` is an empty list.

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


class StateVectorGet(TimeSeriesBaseGet):
    """Retrieve `StateVector` data for a channel.

    This method attemps to get data any way it can, potentially iterating
    over multiple available data sources.

    Parameters
    ----------
    name : `str`, `~gwpy.detector.Channel`
        The name of the channel to read, or a `Channel` object.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine.

    bits : `Bits`, `list`, optional
        Definition of bits for this `StateVector`

    source : `str`, `list`, `list` of `dict`.
        The data source to use. One of the following formats:

        - `str` - the name of a single source to use,
        - `list` - a list of source names to try in order,
        - `list` of `dict` - a list of source specifications to try in order;
          each `dict` must contain a `"source"` key giving the name of the
          source to use, and may contain other keys giving options to
          pass to the data access function for that source.

        See 'Notes' section below for valid source names.

    nproc : `int`, optional
        Number of parallel processes to use, serial process by
        default.

    verbose : `bool`, optional
        This argument is deprecated and will be removed in a future release.
        Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

    kwargs
        Other keyword arguments to pass to the data access function for
        each data source.

    Notes
    -----"""


# -- TimeSeriesBaseDict -------------

class TimeSeriesBaseDictRead(_TimeSeriesRead):
    """Read data into a `TimeSeriesBaseDict`.

    Notes
    -----"""

    def merge(  # type: ignore[override]
        self,
        items: Sequence[TimeSeriesBaseDict],
        pad: float | None = None,
        gap: Literal["raise", "ignore", "pad"] | None = None,
        start: SupportsFloat | None = None,
        end: SupportsFloat | None = None,
    ) -> TimeSeriesBaseDict:
        """Combine a list of `TimeSeriesBaseDict` objects into one `Series`."""
        out = self._cls()
        for tsd in sorted(items, key=lambda x: x.span):
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


class TimeSeriesBaseDictWrite(SeriesWrite):
    """Write data from a `TimeSeriesBaseDict`.

    Notes
    -----"""


class TimeSeriesBaseDictGet(UnifiedGet):
    """Retrieve data for multiple names from any data source.

    Notes
    -----"""

    def __call__(  # type: ignore[override]
        self,
        names: Sequence[str | Channel],
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        source: str | list[str | dict[str, Any]] | None = None,
        pad: float | None = None,
        scaled: bool | None = None,
        dtype: DTypeLike | None = None,
        verbose: bool | None = None,
        allow_tape: bool | None = None,
        **kwargs,
    ) -> TimeSeriesBaseDict:
        """Retrieve data for multiple channels from any data source."""
        dictclass = self._cls
        entryclass = dictclass.EntryClass
        kwargs.setdefault("series_class", entryclass)
        try:
            return super().__call__(
                names,
                to_gps(start),
                to_gps(end),
                source=source,
                pad=pad,
                scaled=scaled,
                dtype=dtype,
                verbose=verbose,
                allow_tape=allow_tape,
                **kwargs,
            )
        except RuntimeError:
            # if we got here then we failed to get all data at once
            if len(names) == 1:
                raise
            self.logger.info(
                "Failed to access data for all names as a group, "
                "trying individually",
            )
            return dictclass((name, entryclass.get(
                name,
                start,
                end,
                source=source,
                pad=pad,
                scaled=scaled,
                dtype=dtype,
                verbose=verbose,
                allow_tape=allow_tape,
                **kwargs,
            )) for name in names)


class TimeSeriesDictGet(TimeSeriesBaseDictGet):
    """Retrieve data for multiple names from any data source.

    This method attemps to get data any way it can, potentially iterating
    over multiple available data sources.

    Parameters
    ----------
    names : `list`
        A `list` of channel names to find.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    source : `str`, `list`, `list` of `dict`.
        The data source to use. One of the following formats:

        - `str` - the name of a single source to use,
        - `list` - a list of source names to try in order,
        - `list` of `dict` - a list of source specifications to try in order;
          each `dict` must contain a `"source"` key giving the name of the
          source to use, and may contain other keys giving options to
          pass to the data access function for that source.

        See 'Notes' section below for valid source names.

    frametype : `str`
        Name of frametype in which this channel is stored, by default
        will search for all required frame types.

    pad : `float`
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    scaled : `bool`
        apply slope and bias calibration to ADC data, for non-ADC data
        this option has no effect.

    nproc : `int`, default: `1`
        Number of parallel processes to use, serial process by
        default.

    allow_tape : `bool`, default: `None`
        Allow the use of data files that are held on tape.
        Default is `None` to attempt to allow the `TimeSeries.fetch`
        method to intelligently select a server that doesn't use tapes
        for data storage (doesn't always work), but to eventually allow
        retrieving data from tape if required.

    verbose : `bool`
        This argument is deprecated and will be removed in a future release.
        Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

    kwargs
        Other keyword arguments to pass to the data access function for
        each data source. See `TimeSeriesDict.get.help(source=<>)` for details
        on the positional and keyword arguments supported for each data source.

    Notes
    -----"""


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

    names : Sequence of `str` or `~gwpy.detector.Channel`
        A list (or similar) of dataset names to read from ``source``.

        - When reading from GWF, this argument should specify the
          names of the channels to read.
        - When reading from HDF5, this argument should specify the
          paths or dataset names within the file.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        Number of parallel processes to use, serial process by
        default.

    pad : `float`, optional
        Value with which to fill gaps in the source data,
        by default gaps will result in a `ValueError`.

    Raises
    ------
    IndexError
        If ``source`` is an empty list.

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

    names : Sequence of `str` or `~gwpy.detector.Channel`
        A list (or similar) of dataset names to read from ``source``.

        - When reading from GWF, this argument should specify the
          names of the channels to read.
        - When reading from HDF5, this argument should specify the
          paths or dataset names within the file.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS start time of required data, defaults to start of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS end time of required data, defaults to end of data found;
        any input parseable by `~gwpy.time.to_gps` is fine.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    parallel : `int`, optional
        Number of parallel processes to use, serial process by
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
        If ``source`` is an empty list.

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


class StateVectorDictGet(TimeSeriesBaseDictGet):
    """Retrieve `StateVector` data for multiple channels.

    This method attemps to get data any way it can, potentially iterating
    over multiple available data sources.

    Parameters
    ----------
    channels : `list`
        Required data channels.

    start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS start time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        GPS end time of required data,
        any input parseable by `~gwpy.time.to_gps` is fine

    source : `str`, `list`, `list` of `dict`.
        The data source to use. One of the following formats:

        - `str` - the name of a single source to use,
        - `list` - a list of source names to try in order,
        - `list` of `dict` - a list of source specifications to try in order;
          each `dict` must contain a `"source"` key giving the name of the
          source to use, and may contain other keys giving options to
          pass to the data access function for that source.

        See 'Notes' section below for valid source names.

    kwargs
        Other keyword arguments to pass to the data access function for
        each data source. See `StateVectorDict.get.help(source=<>)` for details
        on the positional and keyword arguments supported for each data source.

    Notes
    -----"""
