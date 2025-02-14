# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

"""
The TimeSeriesBase.
==================

This module defines the following classes

--------------------  ---------------------------------------------------------
`TimeSeriesBase`      base of the `TimeSeries` and `StateVector` classes,
                      provides the constructor and all common methods
                      (mainly everything that isn't signal-processing related)
`TimeSeriesBaseDict`  base of the `TimeSeriesDict`, this exists mainly so that
                      the `TimeSeriesDict` and `StateVectorDict` can be
                      distinct objects
`TimeSeriesBaseList`  base of the `TimeSeriesList` and `StateVectorList`,
                      same reason for living as the `TimeSeriesBaseDict`
--------------------  ---------------------------------------------------------

**None of these objects are really designed to be used other than as bases for
user-facing objects.**
"""

from __future__ import annotations

import sys
import typing
import warnings
from collections import OrderedDict
from inspect import signature

import numpy
from astropy import units
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST

from ..detector import Channel
from ..io.registry import UnifiedReadWriteMethod
from ..log import logger
from ..segments import SegmentList
from ..time import (
    GPS_TYPES,
    LIGOTimeGPS,
    Time,
    to_gps,
)
from ..types import Series
from .connect import (
    TimeSeriesBaseDictRead,
    TimeSeriesBaseDictWrite,
    TimeSeriesBaseRead,
    TimeSeriesBaseWrite,
)

if typing.TYPE_CHECKING:
    import re
    from collections.abc import Callable
    from typing import Any

    import arrakis
    import nds2

    from ..typing import (
        GpsLike,
        Self,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "TimeSeriesBase",
    "TimeSeriesBaseDict",
    "TimeSeriesBaseList",
]


_UFUNC_STRING = {
    "less": "<",
    "less_equal": "<=",
    "equal": "==",
    "greater_equal": ">=",
    "greater": ">",
}


# -- utilities ----------------------------------------------------------------

def _format_time(gps):
    if isinstance(gps, GPS_TYPES):
        return float(gps)
    if isinstance(gps, Time):
        return gps.gps
    return gps


def _dynamic_scaled(scaled, channel):
    """Determine default for scaled based on channel name.

    This is mainly to work around LIGO not correctly recording ADC
    scaling parameters for most of Advanced LIGO (through 2023).
    Scaling parameters for H0 and L0 data are also not correct
    starting in mid-2020.

    Parameters
    ----------
    scaled : `bool`, `None`
        the scaled argument as given by the user

    channel : `str`
        the name of the channel to be read

    Returns
    -------
    scaled : `bool`
        `False` if channel is from LIGO, otherwise `True`

    Examples
    --------
    >>> _dynamic_scaled(None, "H1:channel")
    False
    >>> _dynamic_scaled(None, "V1:channel")
    True
    """
    if scaled is not None:
        return scaled
    return not str(channel).startswith(("H0", "L0", "H1", "L1"))


# -- TimeSeriesBase------------------------------------------------------------

class TimeSeriesBase(Series):
    """An `Array` with time-domain metadata.

    Parameters
    ----------
    value : array-like
        input data array

    unit : `~astropy.units.Unit`, optional
        physical unit of these data

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine

    dt : `float`, `~astropy.units.Quantity`, optional, default: `1`
        time between successive samples (seconds), can also be given inversely
        via `sample_rate`

    sample_rate : `float`, `~astropy.units.Quantity`, optional, default: `1`
        the rate of samples per second (Hertz), can also be given inversely
        via `dt`

    times : `array-like`
        the complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `t0` and `dt` so should be given
        in place of these if relevant, not alongside

    name : `str`, optional
        descriptive title for this array

    channel : `~gwpy.detector.Channel`, `str`, optional
        source data stream for these data

    dtype : `~numpy.dtype`, optional
        input data type

    copy : `bool`, optional, default: `False`
        choose to copy the input data to new memory

    subok : `bool`, optional, default: `True`
        allow passing of sub-classes by the array generator
    """
    _default_xunit = units.second
    _print_slots = ("t0", "dt", "name", "channel")
    DictClass: type[TimeSeriesBaseDict]

    def __new__(cls, data, unit=None, t0=None, dt=None, sample_rate=None,
                times=None, channel=None, name=None, **kwargs):
        """Generate a new `TimeSeriesBase`."""
        # parse t0 or epoch
        epoch = kwargs.pop("epoch", None)
        if epoch is not None and t0 is not None:
            raise ValueError("give only one of epoch or t0")
        if epoch is None and t0 is not None:
            kwargs["x0"] = _format_time(t0)
        elif epoch is not None:
            kwargs["x0"] = _format_time(epoch)
        # parse sample_rate or dt
        if sample_rate is not None and dt is not None:
            raise ValueError("give only one of sample_rate or dt")
        if sample_rate is None and dt is not None:
            kwargs["dx"] = dt
        # parse times
        if times is not None:
            kwargs["xindex"] = times

        # generate TimeSeries
        new = super().__new__(cls, data, name=name, unit=unit,
                              channel=channel, **kwargs)

        # manually set sample_rate if given
        if sample_rate is not None:
            new.sample_rate = sample_rate

        return new

    # -- TimeSeries properties ------------------

    # rename properties from the Series
    t0 = Series.x0
    dt = Series.dx
    span = Series.xspan
    times = Series.xindex

    # -- epoch
    # this gets redefined to attach to the t0 property
    @property
    def epoch(self):
        """GPS epoch for these data.

        This attribute is stored internally by the `t0` attribute

        :type: `~astropy.time.Time`
        """
        try:
            return Time(self.t0, format="gps", scale="utc")
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch):
        if epoch is None:
            del self.t0
        elif isinstance(epoch, Time):
            self.t0 = epoch.gps
        else:
            try:
                self.t0 = to_gps(epoch)
            except TypeError:
                self.t0 = epoch

    # -- sample_rate
    @property
    def sample_rate(self):
        """Data rate for this `TimeSeries` in samples per second (Hertz).

        This attribute is stored internally by the `dx` attribute

        :type: `~astropy.units.Quantity` scalar
        """
        return (1 / self.dt).to("Hertz")

    @sample_rate.setter
    def sample_rate(self, val):
        if val is None:
            del self.dt
            return
        self.dt = (1 / units.Quantity(val, units.Hertz)).to(self.xunit)

    # -- duration
    @property
    def duration(self):
        """Duration of this series in seconds.

        :type: `~astropy.units.Quantity` scalar
        """
        return units.Quantity(self.span[1] - self.span[0], self.xunit,
                              dtype=float)

    # -- TimeSeries i/o -------------------------

    read = UnifiedReadWriteMethod(TimeSeriesBaseRead)
    write = UnifiedReadWriteMethod(TimeSeriesBaseWrite)

    # -- TimeSeries accessors -------------------

    @classmethod
    def fetch(
        cls,
        channel: str | Channel,
        start: GpsLike,
        end: GpsLike,
        *,
        host: str | None = None,
        port: int | None = None,
        verbose: bool | str = False,
        connection: nds2.connection | None = None,
        verify: bool = False,
        pad: float | None = None,
        allow_tape: bool | None = None,
        scaled: bool | None = None,
        type: int | str | None = None,
        dtype: int | str | None = None,
    ):
        """Fetch data from NDS.

        Parameters
        ----------
        channel : `str`, `~gwpy.detector.Channel`
            The name (or representation) of the data channel to fetch.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        host : `str`, optional
            URL of NDS server to use, if blank will try any server
            (in a relatively sensible order) to get the data

            One of ``connection`` or ``host`` must be given.

        port : `int`, optional
            Port number for NDS server query, must be given with `host`.

        verify : `bool`, optional
            Check channels exist in database before asking for data.
            Default is `True`.

        verbose : `bool`, optional
            Print verbose progress information about NDS download.
            If ``verbose`` is specified as a string, this defines the
            prefix for the progress meter.

        connection : `nds2.connection`, optional
            Open NDS connection to use.
            Default is to open a new connection using ``host`` and ``port``
            arguments.

            One of ``connection`` or ``host`` must be given.

        pad : `float`, optional
            Float value to insert between gaps.
            Default behaviour is to raise an exception when any gaps are
            found.

        scaled : `bool`, optional
            Apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        allow_tape : `bool`, optional
            Allow data access from slow tapes.
            If ``host`` or ``connection`` is given, the default is to do
            whatever the server default is, otherwise servers will be searched
            with ``allow_tape=False`` first, then ``allow_tape=True`` if that
            fails.

        type : `int`, `str`, optional
            NDS2 channel type integer or string name to match.
            Default is to search for any channel type.

        dtype : `numpy.dtype`, `str`, `type`, or `dict`, optional
            NDS2 data type to match.
            Default is to search for any data type.
        """
        return cls.DictClass.fetch(
            [channel],
            start,
            end,
            host=host,
            port=port,
            verbose=verbose,
            connection=connection,
            verify=verify,
            pad=pad,
            scaled=scaled,
            allow_tape=allow_tape,
            type=type,
            dtype=dtype,
        )[str(channel)]

    @classmethod
    def fetch_open_data(cls, ifo, start, end, sample_rate=4096,
                        version=None, format="hdf5",
                        host=GWOSC_DEFAULT_HOST, verbose=False,
                        cache=None, **kwargs):
        """Fetch open-access data from GWOSC.

        Parameters
        ----------
        ifo : `str`
            the two-character prefix of the IFO in which you are interested,
            e.g. `'L1'`

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data, defaults to start of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        sample_rate : `float`, optional,
            the sample rate of desired data; most data are stored
            by GWOSC at 4096 Hz, however there may be event-related
            data releases with a 16384 Hz rate, default: `4096`

        version : `int`, optional
            version of files to download, defaults to highest discovered
            version

        format : `str`, optional
            the data format to download and parse, default: ``'h5py'``

            - ``'hdf5'``
            - ``'gwf'`` - requires |lalframe|_

        host : `str`, optional
            HTTP host name of GWOSC server to access

        verbose : `bool`, optional, default: `False`
            print verbose output while fetching data

        cache : `bool`, optional
            save/read a local copy of the remote URL, default: `False`;
            useful if the same remote data are to be accessed multiple times.
            Set `GWPY_CACHE=1` in the environment to auto-cache.

        timeout : `float`, optional
            the time to wait for a response from the GWOSC server.

        **kwargs
            any other keyword arguments are passed to the `TimeSeries.read`
            method that parses the file that was downloaded

        Examples
        --------
        >>> from gwpy.timeseries import (TimeSeries, StateVector)
        >>> print(TimeSeries.fetch_open_data('H1', 1126259446, 1126259478))
        TimeSeries([  2.17704028e-19,  2.08763900e-19,  2.39681183e-19,
                    ...,   3.55365541e-20,  6.33533516e-20,
                      7.58121195e-20]
                   unit: Unit(dimensionless),
                   t0: 1126259446.0 s,
                   dt: 0.000244140625 s,
                   name: Strain,
                   channel: None)
        >>> print(StateVector.fetch_open_data('H1', 1126259446, 1126259478))
        StateVector([127,127,127,127,127,127,127,127,127,127,127,127,
                     127,127,127,127,127,127,127,127,127,127,127,127,
                     127,127,127,127,127,127,127,127]
                    unit: Unit(dimensionless),
                    t0: 1126259446.0 s,
                    dt: 1.0 s,
                    name: quality/simple,
                    channel: None,
                    bits: Bits(0: data present
                               1: passes cbc CAT1 test
                               2: passes cbc CAT2 test
                               3: passes cbc CAT3 test
                               4: passes burst CAT1 test
                               5: passes burst CAT2 test
                               6: passes burst CAT3 test,
                               channel=None,
                               epoch=1126259446.0))

        For the `StateVector`, the naming of the bits will be
        ``format``-dependent, because they are recorded differently by GWOSC
        in different formats.

        Notes
        -----
        `StateVector` data are not available in ``txt.gz`` format.
        """
        from .io.losc import fetch_gwosc_data
        return fetch_gwosc_data(
            ifo,
            start,
            end,
            sample_rate=sample_rate,
            version=version,
            format=format,
            verbose=verbose,
            cache=cache,
            host=host,
            cls=cls,
            **kwargs,
        )

    @classmethod
    def find(
        cls,
        channel: str | Channel,
        start: GpsLike,
        end: GpsLike,
        *,
        observatory: str | None = None,
        frametype: str | None = None,
        pad: float | None = None,
        scaled: bool | None = None,
        allow_tape: bool | None = None,
        parallel: int = 1,
        verbose: bool | str = False,
        **readargs,
    ) -> Self:
        """Find and return data for multiple channels using GWDataFind.

        This method uses :mod:`gwdatafind` to discover the URLs
        that provide the requested data, then reads those files using
        :meth:`TimeSeriesDict.read()`.

        Parameters
        ----------
        channel : `str`
            Name of data channel to find.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        observatory : `str`, optional
            The observatory to use when searching for data.
            Default is to use the observatory from the channel name prefix,
            but this should be specified when searching for data in a
            multi-observatory dataset (e.g. `observatory='HLV'`).

        frametype : `str`, optional
            Name of frametype (dataset) in which this channel is stored.
            Default is to search all available datasets for a match, which
            can be very slow.

        frametype_match : `str`, optional
            Regular expression to use for frametype matching.

        pad : `float`, optional
            Value with which to fill gaps in the source data,
            by default gaps will result in a `ValueError`.

        scaled : `bool`, optional
            Apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        parallel : `int`, optional
            Number of parallel processes to use.

        allow_tape : `bool`, optional
            Allow reading from frame files on (slow) magnetic tape.

        verbose : `bool`, optional
            Print verbose output about read progress, if ``verbose``
            is specified as a string, this defines the prefix for the
            progress meter.

        readargs
            Any other keyword arguments to be passed to `.read()`.
        """
        return cls.DictClass.find(
            [channel],
            start,
            end,
            observatory=observatory,
            frametype=frametype,
            verbose=verbose,
            pad=pad,
            scaled=scaled,
            allow_tape=allow_tape,
            parallel=parallel,
            **readargs,
        )[str(channel)]

    @classmethod
    def get(
        cls,
        channel: str | Channel,
        start: GpsLike,
        end: GpsLike,
        *,
        source: str | None = None,
        **kwargs,
    ) -> Self:
        """Get data for this channel.

        This method attemps to get data any way it can, potentially iterating
        over multiple available data sources.

        Parameters
        ----------
        channel : `str`, `~gwpy.detector.Channel`
            the name of the channel to read, or a `Channel` object.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        source : `str`
            The data source to use.
            Give one of

            "files"
                Use |gwdatafind|_ to find the paths of local files
                and then read them.

            "nds2"
                Use |nds2|_.

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
            Print verbose output about data access progress.
            If ``verbose`` is specified as a string, this defines the prefix
            for the progress meter.

        kwargs
            Other keyword arguments to pass to the data access function for
            each data source.

        See also
        --------
        TimeSeries.fetch
            for grabbing data from a remote NDS2 server
        TimeSeries.find
            for discovering and reading data from local GWF files
        """
        return cls.DictClass.get(
            [channel],
            start,
            end,
            source=source,
            **kwargs,
        )[str(channel)]

    # -- utilities ------------------------------

    def plot(self, method="plot", figsize=(12, 4), xscale="auto-gps",
             **kwargs):
        """Plot the data for this timeseries.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            the newly created figure, with populated Axes.

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        matplotlib.axes.Axes.plot
            for documentation of keyword arguments used in rendering the data
        """
        kwargs.update(figsize=figsize, xscale=xscale)
        return super().plot(method=method, **kwargs)

    @classmethod
    def from_arrakis(
        cls,
        series: arrakis.Series,
        copy: bool = True,
        **metadata,
    ):
        """Construct a new series from an `arrakis.Series` object.

        Parameters
        ----------
        series : `arrakis.Series`
            The input Arrakis data series to read.

        copy : `bool`, optional
            If `True`, copy the contained data array to new to a new array.

        **metadata
            Any other metadata keyword arguments to pass to the `TimeSeries`
            constructor.

        Returns
        -------
        timeseries : `TimeSeries`
            A new `TimeSeries` containing the data from the `arrakis.Series`
            and the appropriate metadata.
        """
        # get Channel from buffer
        channel = Channel.from_arrakis(series.channel)

        # set default metadata
        metadata.setdefault("channel", channel)
        metadata.setdefault("epoch", LIGOTimeGPS(0, series.time_ns))
        metadata.setdefault("dt", series.dt)
        metadata.setdefault("unit", None)
        metadata.setdefault("name", series.name)

        # construct new TimeSeries-like object
        return cls(series.data, copy=copy, **metadata)

    @classmethod
    def from_nds2_buffer(cls, buffer_, scaled=None, copy=True, **metadata):
        """Construct a new series from an `nds2.buffer` object.

        **Requires:** |nds2|_

        Parameters
        ----------
        buffer_ : `nds2.buffer`
            the input NDS2-client buffer to read

        scaled : `bool`, optional
            apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect

        copy : `bool`, optional
            if `True`, copy the contained data array to new  to a new array

        **metadata
            any other metadata keyword arguments to pass to the `TimeSeries`
            constructor

        Returns
        -------
        timeseries : `TimeSeries`
            a new `TimeSeries` containing the data from the `nds2.buffer`,
            and the appropriate metadata
        """
        # get Channel from buffer
        channel = Channel.from_nds2(buffer_.channel)

        # set default metadata
        metadata.setdefault("channel", channel)
        metadata.setdefault("epoch", LIGOTimeGPS(buffer_.gps_seconds,
                                                 buffer_.gps_nanoseconds))
        metadata.setdefault("sample_rate", channel.sample_rate)
        metadata.setdefault("unit", channel.unit)
        metadata.setdefault("name", buffer_.name)

        # unwrap data
        scaled = _dynamic_scaled(scaled, channel.name)
        slope = buffer_.signal_slope
        offset = buffer_.signal_offset
        null_scaling = slope == 1. and offset == 0.
        if scaled and not null_scaling:
            data = buffer_.data.copy() * slope + offset
            copy = False
        else:
            data = buffer_.data

        # construct new TimeSeries-like object
        return cls(data, copy=copy, **metadata)

    @classmethod
    def from_lal(cls, lalts, copy=True):
        """Generate a new TimeSeries from a LAL TimeSeries of any type."""
        # convert the units
        from ..utils.lal import (from_lal_unit, from_lal_type)
        unit = from_lal_unit(lalts.sampleUnits)

        try:
            dtype = lalts.data.data.dtype
        except AttributeError:  # no data
            dtype = from_lal_type(lalts)
            data = []
        else:
            data = lalts.data.data

        # create new series
        out = cls(
            data,
            dtype=dtype,
            name=lalts.name or None,
            unit=unit,
            t0=lalts.epoch,
            dt=lalts.deltaT,
            channel=None,
            copy=False,
        )

        if copy:
            return out.copy()
        return out

    def to_lal(self):
        """Convert this `TimeSeries` into a LAL TimeSeries.

        .. note::

           This operation always copies data to new memory.
        """
        import lal
        from ..utils.lal import (find_typed_function, to_lal_unit)

        # map unit
        try:
            unit, scale = to_lal_unit(self.unit)
        except ValueError as exc:
            warnings.warn(f"{exc}, defaulting to lal.DimensionlessUnit")
            unit = lal.DimensionlessUnit
            scale = 1

        # create TimeSeries
        create = find_typed_function(self.dtype, "Create", "TimeSeries")
        lalts = create(
            self.name or str(self.channel or "") or None,
            LIGOTimeGPS(to_gps(self.epoch.gps)),
            0,
            self.dt.value,
            unit,
            self.shape[0],
        )

        # assign data
        lalts.data.data = self.value
        if scale != 1:
            lalts.data.data *= scale

        return lalts

    @classmethod
    def from_pycbc(cls, pycbcseries, copy=True):
        """Convert a `pycbc.types.timeseries.TimeSeries` into a `TimeSeries`.

        Parameters
        ----------
        pycbcseries : `pycbc.types.timeseries.TimeSeries`
            the input PyCBC `~pycbc.types.timeseries.TimeSeries` array

        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        timeseries : `TimeSeries`
            a GWpy version of the input timeseries
        """
        return cls(pycbcseries.data, t0=pycbcseries.start_time,
                   dt=pycbcseries.delta_t, copy=copy)

    def to_pycbc(self, copy=True):
        """Convert this `TimeSeries` into a PyCBC
        `~pycbc.types.timeseries.TimeSeries`.

        Parameters
        ----------
        copy : `bool`, optional, default: `True`
            if `True`, copy these data to a new array

        Returns
        -------
        timeseries : `~pycbc.types.timeseries.TimeSeries`
            a PyCBC representation of this `TimeSeries`
        """
        from pycbc import types
        return types.TimeSeries(self.value,
                                delta_t=self.dt.to("s").value,
                                epoch=self.epoch.gps, copy=copy)

    # -- TimeSeries operations ------------------

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = super().__array_ufunc__(ufunc, method, *inputs, **kwargs)
        if out.dtype is numpy.dtype(bool) and len(inputs) == 2:
            from .statevector import StateTimeSeries
            orig, value = inputs
            try:
                op_ = _UFUNC_STRING[ufunc.__name__]
            except KeyError:
                op_ = ufunc.__name__
            out = out.view(StateTimeSeries)
            out.__metadata_finalize__(orig)
            oname = orig.name if isinstance(orig, type(self)) else orig
            vname = value.name if isinstance(value, type(self)) else value
            out.name = "{0!s} {1!s} {2!s}".format(oname, op_, vname)
        return out

    # Quantity overrides __eq__ and __ne__ in a way that doesn't work for us,
    # so we just undo that
    def __eq__(self, other):
        return numpy.ndarray.__eq__(self, other)

    def __ne__(self, other):
        return numpy.ndarray.__ne__(self, other)


# -- TimeSeriesBaseDict -------------------------------------------------------

def as_series_dict_class(seriesclass):
    """Decorate a `dict` class to declare itself as the `DictClass` for
    its `EntryClass`.

    This method should be used to decorate sub-classes of the
    `TimeSeriesBaseDict` to provide a reference to that class from the
    relevant subclass of `TimeSeriesBase`.
    """
    def decorate_class(cls):
        """Set ``cls`` as the `DictClass` attribute for this series type."""
        seriesclass.DictClass = cls
        return cls
    return decorate_class


@as_series_dict_class(TimeSeriesBase)
class TimeSeriesBaseDict(OrderedDict):
    """Ordered key-value mapping of named `TimeSeriesBase` objects.

    This object is designed to hold data for many different sources (channels)
    for a single time span.

    The main entry points for this object are the
    :meth:`~TimeSeriesBaseDict.read` and :meth:`~TimeSeriesBaseDict.fetch`
    data access methods.
    """
    EntryClass = TimeSeriesBase

    @property
    def span(self):
        """The GPS ``[start, stop)`` extent of data in this `dict`.

        :type: `~gwpy.segments.Segment`
        """
        span = SegmentList()
        for value in self.values():
            span.append(value.span)
        try:
            return span.extent()
        except ValueError as exc:  # empty list
            exc.args = (
                "cannot calculate span for empty {0}".format(
                    type(self).__name__),
            )
            raise

    read = UnifiedReadWriteMethod(TimeSeriesBaseDictRead)
    write = UnifiedReadWriteMethod(TimeSeriesBaseDictWrite)

    def __iadd__(self, other):
        return self.append(other)

    def copy(self):
        """Return a copy of this dict with each value copied to new memory."""
        new = self.__class__()
        for key, val in self.items():
            new[key] = val.copy()
        return new

    def append(self, other, copy=True, **kwargs):
        """Append the dict ``other`` to this one.

        Parameters
        ----------
        other : `dict` of `TimeSeries`
            the container to append to this one

        copy : `bool`, optional
            if `True` copy data from ``other`` before storing, only
            affects those keys in ``other`` that aren't in ``self``

        **kwargs
            other keyword arguments to send to `TimeSeries.append`

        See also
        --------
        TimeSeries.append
            for details of the underlying series append operation
        """
        for key, series in other.items():
            if key in self:
                self[key].append(series, **kwargs)
            elif copy:
                self[key] = series.copy()
            else:
                self[key] = series
        return self

    def prepend(self, other, **kwargs):
        """Prepend the dict ``other`` to this one.

        Parameters
        ----------
        other : `dict` of `TimeSeries`
            the container to prepend to this one

        copy : `bool`, optional
            if `True` copy data from ``other`` before storing, only
            affects those keys in ``other`` that aren't in ``self``

        **kwargs
            other keyword arguments to send to `TimeSeries.prepend`

        See also
        --------
        TimeSeries.prepend
            for details of the underlying series prepend operation
        """
        for key, series in other.items():
            if key in self:
                self[key].prepend(series, **kwargs)
            else:
                self[key] = series
        return self

    def crop(self, start=None, end=None, copy=False):
        """Crop each entry of this `dict`.

        This method calls the :meth:`crop` method of all entries and
        modifies this dict in place.

        Parameters
        ----------
        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        copy : `bool`, optional, default: `False`
            If `True` copy the data for each entry to fresh memory,
            otherwise return a view.

        See also
        --------
        TimeSeries.crop
            for more details
        """
        for key, val in self.items():
            self[key] = val.crop(start=start, end=end, copy=copy)
        return self

    def resample(self, rate, **kwargs):
        """Resample items in this dict.

        This operation over-writes items inplace.

        Parameters
        ----------
        rate : `dict`, `float`
            either a `dict` of (channel, `float`) pairs for key-wise
            resampling, or a single float/int to resample all items.

        **kwargs
             other keyword arguments to pass to each item's resampling
             method.
        """
        if not isinstance(rate, dict):
            rate = dict((c, rate) for c in self)
        for key, resamp in rate.items():
            self[key] = self[key].resample(resamp, **kwargs)
        return self

    @classmethod
    def fetch(
        cls,
        channels: list[str | Channel],
        start: GpsLike,
        end: GpsLike,
        *,
        host: str | None = None,
        port: int | None = None,
        verbose: bool | str = False,
        connection: nds2.connection | None = None,
        verify: bool = False,
        pad: float | None = None,
        allow_tape: bool | None = None,
        scaled: bool | None = None,
        type: int | str | None = None,
        dtype: int | str | None = None,
    ):
        """Fetch data from NDS for a number of channels.

        Parameters
        ----------
        channel : `str`, `~gwpy.detector.Channel`
            The name (or representation) of the data channel to fetch.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        host : `str`, optional
            URL of NDS server to use, if blank will try any server
            (in a relatively sensible order) to get the data

            One of ``connection`` or ``host`` must be given.

        port : `int`, optional
            Port number for NDS server query, must be given with `host`.

        verify : `bool`, optional
            Check channels exist in database before asking for data.
            Default is `True`.

        verbose : `bool`, optional
            Print verbose progress information about NDS download.
            If ``verbose`` is specified as a string, this defines the
            prefix for the progress meter.

        connection : `nds2.connection`, optional
            Open NDS connection to use.
            Default is to open a new connection using ``host`` and ``port``
            arguments.

            One of ``connection`` or ``host`` must be given.

        pad : `float`, optional
            Float value to insert between gaps.
            Default behaviour is to raise an exception when any gaps are
            found.

        scaled : `bool`, optional
            Apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        allow_tape : `bool`, optional
            Allow data access from slow tapes.
            If ``host`` or ``connection`` is given, the default is to do
            whatever the server default is, otherwise servers will be searched
            with ``allow_tape=False`` first, then ``allow_tape=True`` if that
            fails.

        type : `int`, `str`, optional
            NDS2 channel type integer or string name to match.
            Default is to search for any channel type.

        dtype : `numpy.dtype`, `str`, `type`, or `dict`, optional
            NDS2 data type to match.
            Default is to search for any data type.

        Returns
        -------
        data : `TimeSeriesBaseDict`
            A new `TimeSeriesBaseDict` of (`str`, `TimeSeries`) pairs fetched
            from NDS.
        """
        from .io.nds2 import fetch_dict

        with logger(
            name=fetch_dict.__module__,
            level="DEBUG" if verbose else None,
        ):
            return fetch_dict(
                channels,
                start,
                end,
                host=host,
                port=port,
                verify=verify,
                verbose=verbose,
                connection=connection,
                pad=pad,
                scaled=scaled,
                allow_tape=allow_tape,
                type=type,
                dtype=dtype,
                series_class=cls.EntryClass,
            )

    @classmethod
    def find(
        cls,
        channels: list[str | Channel],
        start: GpsLike,
        end: GpsLike,
        *,
        observatory: str | None = None,
        frametype: str | None = None,
        frametype_match: str | re.Pattern | None = None,
        pad: float | None = None,
        scaled: bool | None = None,
        allow_tape: bool | None = None,
        parallel: int = 1,
        verbose: bool | str = False,
        **readargs,
    ) -> Self:
        """Find and read data from frames for a number of channels.

        This method uses :mod:`gwdatafind` to discover the (`file://`) URLs
        that provide the requested data, then reads those files using
        :meth:`TimeSeriesDict.read()`.

        Parameters
        ----------
        channels : `list`
            List of names of data channels to find.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine

        observatory : `str`, optional
            The observatory to use when searching for data.
            Default is to use the observatory from the channel name prefix,
            but this should be specified when searching for data in a
            multi-observatory dataset (e.g. `observatory='HLV'`).

        frametype : `str`, optional
            Name of frametype (dataset) in which this channel is stored.
            Default is to search all available datasets for a match, which
            can be very slow.

        frametype_match : `str`, optional
            Regular expression to use for frametype matching.

        pad : `float`, optional
            Value with which to fill gaps in the source data,
            by default gaps will result in a `ValueError`.

        scaled : `bool`, optional
            Apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        parallel : `int`, optional
            Number of parallel threads to use when reading data.

        allow_tape : `bool`, optional
            Allow reading from frame files on (slow) magnetic tape.

        verbose : `bool`, optional
            Print verbose output about read progress, if ``verbose``
            is specified as a string, this defines the prefix for the
            progress meter.

        readargs
            Any other keyword arguments to be passed to `.read()`.

        Raises
        ------
        requests.exceptions.HTTPError
            If the GWDataFind query fails for any reason.

        RuntimeError
            If no files are found to read, or if the read operation
            fails.
        """
        from .io.gwdatafind import find

        series_class = readargs.pop("series_class", cls.EntryClass)
        with logger(
            name=find.__module__,
            level="DEBUG" if verbose else None,
        ):
            return cls(find(
                channels,
                start,
                end,
                observatory=observatory,
                frametype=frametype,
                frametype_match=frametype_match,
                pad=pad,
                scaled=scaled,
                allow_tape=allow_tape,
                parallel=parallel,
                verbose=verbose,
                series_class=series_class,
                **readargs,
            ))

    @classmethod
    def get(  # type: ignore[override]
        cls,
        channels: list[str | Channel],
        start: GpsLike,
        end: GpsLike,
        *,
        source: str | list[str] | None = None,
        verbose: bool = False,
        **kwargs,
    ):
        """Retrieve data for multiple channels from any data source.

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

        source : `str`
            The data source to use.
            Give one of

            "files"
                Use |gwdatafind|_ to find the paths of local files
                and then read them.

            "nds2"
                Use |nds2|_.

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
            Print verbose output about data access progress.
            If ``verbose`` is specified as a string, this defines the prefix
            for the progress meter.

        kwargs
            Other keyword arguments to pass to the data access function for
            each data source.

        See also
        --------
        TimeSeries.find
            For details of how data are accessed for ``source="files"``
            and the supported keyword arguments.

        TimeSeries.fetch
            For details of how data are accessed for ``source="nds2"``
            and the supported keyword arguments.
        """
        # the list of places we can try to get data
        sources: list[str]
        if source is None:
            sources = [
                "files",
                "NDS2",
            ]
        elif isinstance(source, str):
            sources = [source]
        else:
            sources = list(source)
        nsources = len(sources)

        # record errors that happen along the way
        error: Exception | None = None

        GETTER: dict[str, tuple[Callable, dict[str, Any]]] = {
            "files": (cls.find, {}),
            "nds2": (cls.fetch, {}),
        }
        for source in sources:
            try:
                getter, default_kwargs = GETTER[source.lower()]
            except KeyError:
                raise ValueError(f"invalid data source '{source}'")
            params = [
                p.name
                for p in signature(getter).parameters.values()
                if p.kind == p.KEYWORD_ONLY
            ]
            these_kwargs = default_kwargs | {
                key: val for key, val in kwargs.items()
                if key in params and val is not None
            }
            if verbose:
                print(f"- Attempting data access from {source}", flush=True)
            try:
                return getter(
                    channels,
                    start,
                    end,
                    verbose=verbose,
                    **these_kwargs,
                )
            except (
                ImportError,  # optional dependency is missing
                RuntimeError,
                ValueError,
            ) as exc:
                if len(channels) == 1 and nsources == 1:
                    raise
                if error:
                    # add this error to the chain of errors
                    exc.__context__ = error
                error = exc
                if verbose:
                    print(str(exc), file=sys.stderr, flush=True)
                    print(f"Data access from {source} failed", flush=True)

        # if we got here then we failed to get all data at once
        if len(channels) == 1:
            raise RuntimeError("Failed to get data from any source.") from error
        if verbose:
            print(
                "Failed to access data for all channels as a group, "
                "trying individually:",
            )
        return cls((c, cls.EntryClass.get(
            c,
            start,
            end,
            verbose=verbose,
            **kwargs,
        )) for c in channels)

    @classmethod
    def from_arrakis(
        cls,
        block: arrakis.SeriesBlock,
        copy=True,
        **metadata,
    ):
        """Construct a new dict from an `arrakis.SeriesBlock`.

        Parameters
        ----------
        block : `arrakis.SeriesBlock`
            The input Arrakis data to read.

        copy : `bool`, optional
            If `True`, copy the contained data array to new  to a new array.

        **metadata
            Any other metadata keyword arguments to pass to the `TimeSeries`
            constructor.

        Returns
        -------
        dict : `TimeSeriesDict`
            A new `TimeSeriesDict` containing the data from the Arrakis block.
        """
        tsd = cls()
        for name, series in block.items():
            tsd[name] = tsd.EntryClass.from_arrakis(
                series,
                copy=copy,
                **metadata,
            )
        return tsd

    @classmethod
    def from_nds2_buffers(cls, buffers, scaled=None, copy=True, **metadata):
        """Construct a new dict from a list of `nds2.buffer` objects.

        **Requires:** |nds2|_

        Parameters
        ----------
        buffers : `list` of `nds2.buffer`
            the input NDS2-client buffers to read

        scaled : `bool`, optional
            apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        copy : `bool`, optional
            if `True`, copy the contained data array to new  to a new array

        **metadata
            any other metadata keyword arguments to pass to the `TimeSeries`
            constructor

        Returns
        -------
        dict : `TimeSeriesDict`
            a new `TimeSeriesDict` containing the data from the given buffers
        """
        tsd = cls()
        for buf in buffers:
            tsd[buf.channel.name] = tsd.EntryClass.from_nds2_buffer(
                buf, scaled=scaled, copy=copy, **metadata)
        return tsd

    def plot(self, label="key", method="plot", figsize=(12, 4),
             xscale="auto-gps", **kwargs):
        """Plot the data for this `TimeSeriesBaseDict`.

        Parameters
        ----------
        label : `str`, optional
            labelling system to use, or fixed label for all elements
            Special values include

            - ``'key'``: use the key of the `TimeSeriesBaseDict`,
            - ``'name'``: use the :attr:`~TimeSeries.name` of each element

            If anything else, that fixed label will be used for all lines.

        **kwargs
            all other keyword arguments are passed to the plotter as
            appropriate
        """
        kwargs.update({
            "method": method,
            "label": label,
        })

        # make plot
        from ..plot import Plot

        if kwargs.get("separate", False):
            plot = Plot(*self.values(), **kwargs)
        else:
            plot = Plot(self.values(), **kwargs)

        # update labels
        artmap = {"plot": "lines", "scatter": "collections"}
        artists = [x for ax in plot.axes for
                   x in getattr(ax, artmap.get(method, "lines"))]
        for key, artist in zip(self, artists, strict=True):
            if label.lower() == "name":
                lab = self[key].name
            elif label.lower() == "key":
                lab = key
            else:
                lab = label
            artist.set_label(lab)

        return plot

    def step(self, label="key", where="post", figsize=(12, 4),
             xscale="auto-gps", **kwargs):
        """Create a step plot of this dict.

        Parameters
        ----------
        label : `str`, optional
            labelling system to use, or fixed label for all elements
            Special values include

            - ``'key'``: use the key of the `TimeSeriesBaseDict`,
            - ``'name'``: use the :attr:`~TimeSeries.name` of each element

            If anything else, that fixed label will be used for all lines.

        **kwargs
            all other keyword arguments are passed to the plotter as
            appropriate
        """
        kwargs.setdefault(
            "drawstyle",
            "steps-{}".format(where),
        )
        tmp = type(self)()
        for key, series in self.items():
            tmp[key] = series.append(series.value[-1:], inplace=False)

        return tmp.plot(label=label, figsize=figsize, xscale=xscale,
                        **kwargs)


# -- TimeSeriesBaseList -------------------------------------------------------

class TimeSeriesBaseList(list):
    """Fancy list representing a list of `TimeSeriesBase`.

    The `TimeSeriesBaseList` provides an easy way to collect and organise
    `TimeSeriesBase` for a single `Channel` over multiple segments.

    Parameters
    ----------
    *items
        any number of `TimeSeriesBase`

    Returns
    -------
    list
        a new `TimeSeriesBaseList`

    Raises
    ------
    TypeError
        if any elements are not `TimeSeriesBase`
    """
    EntryClass = TimeSeriesBase

    def __init__(self, *items):
        """Initialise a new list."""
        super().__init__()
        for item in items:
            self.append(item)

    @property
    def segments(self):
        """The `span` of each series in this list."""
        from ..segments import SegmentList
        return SegmentList([item.span for item in self])

    def append(self, item):
        if not isinstance(item, self.EntryClass):
            raise TypeError("Cannot append type '%s' to %s"
                            % (type(item).__name__, type(self).__name__))
        super().append(item)
        return self
    append.__doc__ = list.append.__doc__

    def extend(self, item):
        item = TimeSeriesBaseList(*item)
        super().extend(item)
    extend.__doc__ = list.extend.__doc__

    def coalesce(self):
        """Merge contiguous elements of this list into single objects.

        This method implicitly sorts and potentially shortens this list.
        """
        self.sort(key=lambda ts: ts.t0.value)
        i = j = 0
        N = len(self)
        while j < N:
            this = self[j]
            j += 1
            if j < N and this.is_contiguous(self[j]) == 1:
                while j < N and this.is_contiguous(self[j]):
                    try:
                        this = self[i] = this.append(self[j])
                    except ValueError as exc:
                        if "cannot resize this array" in str(exc):
                            this = this.copy()
                            this = self[i] = this.append(self[j])
                        else:
                            raise
                    j += 1
            else:
                self[i] = this
            i += 1
        del self[i:]
        return self

    def join(self, pad=None, gap=None):
        """Concatenate all of the elements of this list into a single object.

        Parameters
        ----------
        pad : `float`, optional
            value with which to fill gaps in the source data,
            by default gaps will result in a `ValueError`.

        gap : `str`, optional, default: `'raise'`
            what to do if there are gaps in the data, one of

            - ``'raise'`` - raise a `ValueError`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

            If `pad` is given and is not `None`, the default is ``'pad'``,
            otherwise ``'raise'``.

        Returns
        -------
        series : `gwpy.types.TimeSeriesBase` subclass
             a single series containing all data from each entry in this list

        See also
        --------
        TimeSeries.append
            for details on how the individual series are concatenated together
        """
        if not self:
            return self.EntryClass(numpy.empty((0,) * self.EntryClass._ndim))
        self.sort(key=lambda t: t.epoch.gps)
        out = self[0].copy()
        for series in self[1:]:
            out.append(series, gap=gap, pad=pad)
        return out

    def __getslice__(self, i, j):
        return type(self)(*super().__getslice__(i, j))

    def __getitem__(self, key):
        if isinstance(key, slice):
            return type(self)(
                *super().__getitem__(key))
        return super().__getitem__(key)

    def copy(self):
        """Return a copy of this list with each element copied to new memory."""
        out = type(self)()
        for series in self:
            out.append(series.copy())
        return out
