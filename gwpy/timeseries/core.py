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

"""The base time-series array object.

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

import warnings
from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
    cast,
    overload,
)

import numpy
from astropy import units
from astropy.units import Quantity
from gwosc.api import DEFAULT_URL as GWOSC_DEFAULT_HOST

from ..detector import Channel
from ..io.registry import UnifiedReadWriteMethod
from ..segments import SegmentList
from ..time import (
    LIGOTimeGPS,
    Time,
    to_gps,
)
from ..types import Series
from ..utils.misc import property_alias
from .connect import (
    TimeSeriesBaseDictGet,
    TimeSeriesBaseDictRead,
    TimeSeriesBaseDictWrite,
    TimeSeriesBaseGet,
    TimeSeriesBaseRead,
    TimeSeriesBaseWrite,
)

if TYPE_CHECKING:
    import re
    from collections.abc import (
        Callable,
        Iterable,
        Mapping,
    )
    from typing import (
        ClassVar,
        Literal,
        Self,
        SupportsFloat,
        SupportsIndex,
    )

    import arrakis
    import nds2
    import pycbc
    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike
    from numpy.typing import (
        DTypeLike,
        NDArray,
    )

    from ..plot import Plot
    from ..segments import Segment
    from ..time import SupportsToGps
    from ..typing import (
        ArrayLike1D,
        UnitLike,
    )
    from ..utils.lal import LALTimeSeriesType
    from .statevector import StateTimeSeries

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


# -- utilities -----------------------

def _format_time(gps: Time | Quantity | SupportsFloat) -> float:
    """Format a GPS time into a float."""
    if isinstance(gps, Time):
        return gps.gps
    if isinstance(gps, Quantity):
        return gps.to(units.second).value
    return float(gps)


def _dynamic_scaled(
    scaled: bool | None,  # noqa: FBT001
    channel: str | Channel,
) -> bool:
    """Determine default for scaled based on channel name.

    This is mainly to work around LIGO not correctly recording ADC
    scaling parameters for most of Advanced LIGO (through 2023).
    Scaling parameters for H0 and L0 data are also not correct
    starting in mid-2020.

    Parameters
    ----------
    scaled : `bool`, `None`
        The scaled argument as given by the user.

    channel : `str`, `Channel`
        The name of the channel to be read.

    Returns
    -------
    scaled : `bool`
        `False` if channel is from LIGO, otherwise `True`.

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


# -- TimeSeriesBase-------------------

class TimeSeriesBase(Series):
    """An `Array` with time-domain metadata.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    dt : `float`, `~astropy.units.Quantity`, optional
        Time between successive samples (seconds), can also be given inversely
        via `sample_rate`.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        The rate of samples per second (Hertz), can also be given inversely
        via `dt`.

    times : `array-like`
        The complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `t0` and `dt` so should be given
        in place of these if relevant, not alongside.

    name : `str`, optional
        Descriptive title for this array.

    channel : `~gwpy.detector.Channel`, `str`, optional
        Source data stream for these data.

    dtype : `~numpy.dtype`, optional
        Input data type.

    copy : `bool`, optional, default: `False`
        Choose to copy the input data to new memory.

    subok : `bool`, optional, default: `True`
        Allow passing of sub-classes by the array generator.
    """

    _default_xunit: ClassVar[UnitBase] = units.second
    _print_slots: ClassVar[tuple[str, ...]] = (
        "t0",
        "dt",
        "name",
        "channel",
    )
    DictClass: ClassVar[type[TimeSeriesBaseDict]]

    def __new__(
        cls,
        data: ArrayLike1D,
        unit: UnitLike = None,
        t0: SupportsToGps | None = None,
        dt: float | Quantity | None = None,
        sample_rate: float | Quantity | None = None,
        times: ArrayLike1D | None = None,
        channel: Channel | str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> Self:
        """Generate a new `TimeSeriesBase`."""
        # parse t0 or epoch
        epoch = kwargs.pop("epoch", None)
        if epoch is not None and t0 is not None:
            msg = "give only one of epoch or t0"
            raise ValueError(msg)
        if epoch is None and t0 is not None:
            kwargs["x0"] = _format_time(t0)
        elif epoch is not None:
            kwargs["x0"] = _format_time(epoch)

        # parse sample_rate or dt
        if sample_rate is not None and dt is not None:
            msg = "give only one of sample_rate or dt"
            raise ValueError(msg)
        if sample_rate is None and dt is not None:
            kwargs["dx"] = dt

        # parse times
        if times is not None:
            kwargs["xindex"] = times

        # generate TimeSeries
        new = super().__new__(
            cls,
            data,
            name=name,
            unit=unit,
            channel=channel,
            **kwargs,
        )

        # manually set sample_rate if given
        if sample_rate is not None:
            new.sample_rate = sample_rate

        return new

    # -- TimeSeries properties -------

    # rename properties from the Series
    t0 = property_alias(Series.x0, "GPS start time of this series.")  # type: ignore[arg-type]
    dt = property_alias(Series.dx, "Time (seconds) between successive samples.")  # type: ignore[arg-type]
    span = property_alias(Series.xspan, "Time (seconds) spanned by this series.")  # type: ignore[arg-type]
    times = property_alias(Series.xindex, "Array of GPS times for each sample.")  # type: ignore[arg-type]

    # -- epoch
    # this gets redefined to attach to the t0 property
    @property
    def epoch(self) -> Time | None:
        """GPS epoch for these data.

        This attribute is stored internally by the `t0` attribute.
        """
        try:
            return Time(self.t0, format="gps", scale="utc")
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch: Time | SupportsToGps | None) -> None:
        if epoch is None:
            del self.t0
        elif isinstance(epoch, Time):
            self.t0 = epoch.gps
        else:
            try:
                self.t0 = to_gps(epoch)  # type: ignore[assignment]
            except TypeError:
                self.t0 = epoch

    # -- sample_rate
    @property
    def sample_rate(self) -> Quantity:
        """Data rate for this `TimeSeries` in samples per second (Hertz).

        This attribute is stored internally by the `dx` attribute
        """
        return (1 / self.dt).to("Hertz")

    @sample_rate.setter
    def sample_rate(self, val: QuantityLike | None) -> None:
        if val is None:
            del self.dt
            return
        self.dt = (1 / Quantity(val, units.Hertz)).to(self.xunit)

    # -- duration
    @property
    def duration(self) -> Quantity:
        """Duration of this series in seconds.

        :type: `~astropy.units.Quantity` scalar
        """
        return Quantity(
            abs(self.span),
            self.xunit,
            dtype=float,
        )

    # -- TimeSeries i/o --------------

    read = UnifiedReadWriteMethod(TimeSeriesBaseRead)
    write = UnifiedReadWriteMethod(TimeSeriesBaseWrite)
    get = UnifiedReadWriteMethod(TimeSeriesBaseGet)

    # -- TimeSeries accessors --------

    @classmethod
    def fetch(
        cls,
        channel: str | Channel,
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        host: str | None = None,
        port: int | None = None,
        verbose: bool | str = False,
        connection: nds2.connection | None = None,
        verify: bool = False,
        pad: float | None = None,
        allow_tape: bool | None = None,
        scaled: bool | None = None,
        type: int | str | None = None,  # noqa: A002
        dtype: int | str | None = None,
    ) -> Self:
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
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

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
        return cls.get(
            channel,
            start,
            end,
            source="nds2",
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
        )

    @classmethod
    def fetch_open_data(
        cls,
        ifo: str,
        start: SupportsToGps,
        end: SupportsToGps,
        sample_rate: float = 4096,
        version: int | None = None,
        format: Literal["gwf", "hdf5"] = "hdf5",  # noqa: A002
        host: str = GWOSC_DEFAULT_HOST,
        *,
        verbose: bool | None = None,
        cache: bool | None = None,
        **kwargs,
    ) -> Self:
        """Fetch open-access data from GWOSC.

        This is just a shim around ``TimeSeries.get(..., source='gwosc')``.

        Parameters
        ----------
        ifo : `str`
            The two-character prefix of the IFO in which you are interested,
            e.g. `'L1'`.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS start time of required data, defaults to start of data found;
            any input parseable by `~gwpy.time.to_gps` is fine.

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine.

        sample_rate : `float`, optional,
            The sample rate of desired data; most data are stored
            by GWOSC at 4096 Hz, however there may be event-related
            data releases with a 16384 Hz rate, default: `4096`.

        version : `int`, optional
            Version of files to download, defaults to highest discovered
            version.

        format : `str`, optional
            The data format to download and parse, default: ``'h5py'``

            - ``'hdf5'``
            - ``'gwf'`` - requires |lalframe|_

        host : `str`, optional
            HTTP host name of GWOSC server to access.

        verbose : `bool`, optional
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

        cache : `bool`, optional
            Save/read a local copy of the remote URL, default: `False`;
            useful if the same remote data are to be accessed multiple times.
            Set ``GWPY_CACHE=1`` in the environment to auto-cache.

        timeout : `float`, optional
            The time to wait for a response from the GWOSC server.

        kwargs
            Any other keyword arguments are passed to the `TimeSeries.read`
            method that parses the file that was downloaded.

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
        return cls.get(
            ifo,
            start,
            end,
            source="gwosc",
            sample_rate=sample_rate,
            version=version,
            format=format,
            verbose=verbose,
            cache=cache,
            host=host,
            series_class=cls,
            **kwargs,
        )

    @classmethod
    def find(
        cls,
        channel: str | Channel,
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        observatory: str | None = None,
        frametype: str | None = None,
        frametype_match: str | re.Pattern | None = None,
        host: str | None = None,
        urltype: str | None = "file",
        ext: str = "gwf",
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

        This is just a shim around ``TimeSeries.get(..., source='gwdatafind')``.

        Parameters
        ----------
        channel : `str`
            Name of data channel to find.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine.

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data, defaults to end of data found;
            any input parseable by `~gwpy.time.to_gps` is fine.

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

        host : `str`, optional
            Name of the GWDataFind server to use.
            Default is set by `gwdatafind.utils.get_default_host`.

        urltype : `str`, optional
            The URL type to use.
            Default is "file" to use paths available on the file system.

        ext : `str`, optional
            The file extension for which to search.
            "gwf" is the only file extension supported, but this may be
            extended in the future.

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
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

        readargs
            Any other keyword arguments to be passed to `.read()`.
        """
        return cls.get(
            channel,
            start,
            end,
            source="gwdatafind",
            observatory=observatory,
            frametype=frametype,
            frametype_match=frametype_match,
            host=host,
            urltype=urltype,
            ext=ext,
            verbose=verbose,
            pad=pad,
            scaled=scaled,
            allow_tape=allow_tape,
            parallel=parallel,
            **readargs,
        )

    # -- utilities -------------------

    def plot(
        self,
        method: str = "plot",
        figsize: tuple[int, int] = (12, 4),
        xscale: str = "auto-gps",
        **kwargs,
    ) -> Plot:
        """Plot the data for this timeseries.

        Returns
        -------
        figure : `~matplotlib.figure.Figure`
            The newly created figure, with populated Axes.

        See Also
        --------
        matplotlib.pyplot.figure
            For documentation of keyword arguments used to create the
            figure.
        matplotlib.figure.Figure.add_subplot
            For documentation of keyword arguments used to create the
            axes.
        matplotlib.axes.Axes.plot
            For documentation of keyword arguments used in rendering the data.
        """
        kwargs.update(figsize=figsize, xscale=xscale)
        return super().plot(method=method, **kwargs)

    @classmethod
    def from_arrakis(
        cls,
        series: arrakis.block.Series,
        *,
        copy: bool = True,
        **metadata,
    ) -> Self:
        """Construct a new series from an `arrakis.Series` object.

        Parameters
        ----------
        series : `arrakis.Series`
            The input Arrakis data series to read.

        copy : `bool`, optional
            If `True`, copy the contained data array to new to a new array.

        metadata
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
        defaults = {
            "channel": channel,
            "epoch": LIGOTimeGPS(0, series.time_ns),
            "dt": series.dt,
            "unit": None,
            "name": series.name,
        }
        metadata = {**defaults, **metadata}

        # construct new TimeSeries-like object
        return cls(series.data, copy=copy, **metadata)

    @classmethod
    def from_nds2_buffer(
        cls,
        buffer: nds2.buffer,
        *,
        scaled: bool | None = None,
        copy: bool = True,
        **metadata,
    ) -> Self:
        """Construct a new series from an `nds2.buffer` object.

        **Requires:** |nds2|_

        Parameters
        ----------
        buffer : `nds2.buffer`
            The input NDS2-client buffer to read.

        scaled : `bool`, optional
            Apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        copy : `bool`, optional
            Tf `True`, copy the contained data array to new  to a new array.

        metadata
            Any other metadata keyword arguments to pass to the `TimeSeries`
            constructor.

        Returns
        -------
        timeseries : `TimeSeries`
            A new `TimeSeries` containing the data from the `nds2.buffer`,
            and the appropriate metadata.
        """
        # get Channel from buffer
        channel = Channel.from_nds2(buffer.channel)

        # set default metadata
        defaults = {
            "channel": channel,
            "epoch": LIGOTimeGPS(buffer.gps_seconds, buffer.gps_nanoseconds),
            "sample_rate": channel.sample_rate,
            "unit": channel.unit,
            "name": buffer.name,
        }
        metadata = {**defaults, **metadata}

        # unwrap data
        scaled = _dynamic_scaled(scaled, channel.name)
        slope = buffer.signal_slope
        offset = buffer.signal_offset
        null_scaling = slope == 1. and offset == 0.
        if scaled and not null_scaling:
            data = buffer.data.copy() * slope + offset
            copy = False
        else:
            data = buffer.data

        # construct new TimeSeries-like object
        return cls(data, copy=copy, **metadata)

    @classmethod
    def from_lal(
        cls,
        lalts: LALTimeSeriesType,
        *,
        copy: bool = True,
    ) -> Self:
        """Generate a new TimeSeries from a LAL TimeSeries of any type."""
        # convert the units
        from ..utils.lal import (
            from_lal_type,
            from_lal_unit,
        )

        unit = from_lal_unit(lalts.sampleUnits)

        dtype: DTypeLike
        try:
            dtype = lalts.data.data.dtype
        except AttributeError:  # no data
            dtype = from_lal_type(type(lalts))
            data = numpy.array([], dtype=dtype)
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

    def to_lal(self) -> LALTimeSeriesType:
        """Convert this `TimeSeries` into a LAL TimeSeries.

        .. note::

           This operation always copies data to new memory.
        """
        import lal

        from ..utils.lal import (
            find_typed_function,
            to_lal_unit,
        )

        # map unit
        try:
            unit, scale = to_lal_unit(self.unit)
        except ValueError as exc:
            warnings.warn(
                f"{exc}, defaulting to lal.DimensionlessUnit",
                stacklevel=2,
            )
            unit = lal.DimensionlessUnit
            scale = 1

        # create TimeSeries
        create = find_typed_function(self.dtype, "Create", "TimeSeries")
        lalts = create(
            self.name or str(self.channel or "") or None,
            LIGOTimeGPS(self.t0.value),
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
    def from_pycbc(
        cls,
        pycbcseries: pycbc.types.TimeSeries,
        *,
        copy: bool = True,
    ) -> Self:
        """Convert a `pycbc.types.timeseries.TimeSeries` into a `TimeSeries`.

        Parameters
        ----------
        pycbcseries : `pycbc.types.timeseries.TimeSeries`
            The input PyCBC `~pycbc.types.timeseries.TimeSeries` array.

        copy : `bool`, optional
            If `True`, copy these data to a new array.

        Returns
        -------
        timeseries : `TimeSeries`
            A GWpy version of the input timeseries.
        """
        return cls(
            pycbcseries.data,
            t0=pycbcseries.start_time,
            dt=pycbcseries.delta_t,
            copy=copy,
        )

    def to_pycbc(self, *, copy: bool = True) -> pycbc.types.TimeSeries:
        """Convert this `TimeSeries` into a PyCBC `~pycbc.types.timeseries.TimeSeries`.

        Parameters
        ----------
        copy : `bool`, optional, default: `True`
            If `True`, copy these data to a new array.

        Returns
        -------
        timeseries : `~pycbc.types.timeseries.TimeSeries`
            A PyCBC representation of this `TimeSeries`.
        """
        from pycbc import types
        return types.TimeSeries(
            self.value,
            delta_t=self.dt.to("s").value,
            epoch=self.t0.value, copy=copy,
        )

    # -- TimeSeries operations -------

    def __array_ufunc__(  # type: ignore[override]
        self,
        function: Callable,
        method: str,
        *inputs,
        **kwargs,
    ) -> Self | Quantity | StateTimeSeries:
        """Override the default array ufunc to handle TimeSeries metadata."""
        out = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if out.dtype is numpy.dtype(bool) and len(inputs) == 2:
            from .statevector import StateTimeSeries
            orig, value = inputs
            try:
                op_ = _UFUNC_STRING[function.__name__]
            except KeyError:
                op_ = function.__name__
            out = out.view(StateTimeSeries)
            out.__metadata_finalize__(orig)
            oname = orig.name if isinstance(orig, type(self)) else orig
            vname = value.name if isinstance(value, type(self)) else value
            out.name = f"{oname!s} {op_!s} {vname!s}"
        return out

    # Quantity overrides __eq__ and __ne__ in a way that doesn't work for us,
    # so we just undo that
    def __eq__(self, other: object) -> bool:
        """Return `True` if ``other`` is equal to this `TimeSeries`."""
        return numpy.ndarray.__eq__(self, other)

    def __ne__(self, other: object) -> bool:
        """Return `True` if ``other`` is not equal to this `TimeSeries`."""
        return numpy.ndarray.__ne__(self, other)


# -- TimeSeriesBaseDict --------------

def as_series_dict_class(
    seriesclass: type[TimeSeriesBase],
) -> Callable[[type[TimeSeriesBaseDict]], type[TimeSeriesBaseDict]]:
    """Return a decorator for a `dict` class to define `DictClass` for its `EntryClass`.

    This method should be used to decorate sub-classes of the
    `TimeSeriesBaseDict` to provide a reference to that class from the
    relevant subclass of `TimeSeriesBase`.
    """
    def decorate_class(cls: type[TimeSeriesBaseDict]) -> type[TimeSeriesBaseDict]:
        """Set ``cls`` as the `DictClass` attribute for this series type."""
        seriesclass.DictClass = cls
        return cls

    return decorate_class


# Type variable for generic dict values
_V = TypeVar("_V", bound=TimeSeriesBase)


@as_series_dict_class(TimeSeriesBase)
class TimeSeriesBaseDict(dict[str | Channel, _V], Generic[_V]):
    """Key-value mapping of named `TimeSeriesBase` objects.

    This object is designed to hold data for many different sources (channels)
    for a single time span. Dictionary keys are ordered by insertion order.

    The main entry points for this object are the
    :meth:`~TimeSeriesBaseDict.read` and :meth:`~TimeSeriesBaseDict.fetch`
    data access methods.
    """

    EntryClass: ClassVar[type[TimeSeriesBase]] = TimeSeriesBase

    @property
    def span(self) -> Segment:
        """The GPS ``[start, stop)`` extent of data in this `dict`."""
        span = SegmentList()
        for value in self.values():
            span.append(value.span)
        try:
            return span.extent()
        except ValueError as exc:  # empty list
            exc.args = (
                f"cannot calculate span for empty {type(self).__name__}",
            )
            raise

    read = UnifiedReadWriteMethod(TimeSeriesBaseDictRead)
    write = UnifiedReadWriteMethod(TimeSeriesBaseDictWrite)
    get = UnifiedReadWriteMethod(TimeSeriesBaseDictGet)

    def __iadd__(self, other: dict[str | Channel, numpy.ndarray]) -> Self:
        """Append a `TimeSeriesBase` or `numpy.ndarray` to this dict."""
        self.append(other)
        return self

    def copy(self) -> Self:
        """Return a copy of this dict with each value copied to new memory."""
        new = self.__class__()
        for key, val in self.items():
            new[key] = val.copy()
        return new

    def append(
        self,
        other: Mapping[str | Channel, NDArray],
        *,
        copy: bool = True,
        **kwargs,
    ) -> Self:
        """Append the dict ``other`` to this one.

        Parameters
        ----------
        other : `dict` of `TimeSeries`
            The container to append to this one.

        copy : `bool`, optional
            If `True` copy data from ``other`` before storing, only
            affects those keys in ``other`` that aren't in ``self``.

        **kwargs
            Other keyword arguments to send to `TimeSeries.append`.

        See Also
        --------
        TimeSeries.append
            For details of the underlying series append operation.
        """
        for key, series in other.items():
            if key in self:
                self[key].append(series, **kwargs)
            else:
                if not isinstance(series, self.EntryClass):
                    msg = (
                        f"cannot append {type(series).__name__} to "
                        f"{type(self).__name__} for new key {key!r}"
                    )
                    raise ValueError(msg)
                if copy:
                    series = series.copy()  # noqa: PLW2901
                self[key] = series
        return self

    def prepend(
        self,
        other: Mapping[str | Channel, _V],
        **kwargs,
    ) -> Self:
        """Prepend the dict ``other`` to this one.

        Parameters
        ----------
        other : `dict` of `TimeSeries`
            The container to prepend to this one.

        copy : `bool`, optional
            If `True` copy data from ``other`` before storing, only
            affects those keys in ``other`` that aren't in ``self``.

        kwargs
            Other keyword arguments to send to `TimeSeries.prepend`.

        See Also
        --------
        TimeSeries.prepend
            For details of the underlying series prepend operation.
        """
        for key, series in other.items():
            if key in self:
                self[key].prepend(series, **kwargs)
            else:
                self[key] = series
        return self

    def crop(
        self,
        start: SupportsToGps | None = None,
        end: SupportsToGps | None = None,
        *,
        copy: bool = False,
    ) -> Self:
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

        See Also
        --------
        TimeSeries.crop
            for more details
        """
        for key, val in self.items():
            self[key] = val.crop(start=start, end=end, copy=copy)
        return self

    def resample(
        self,
        rate: dict[str | Channel, float] | float,
        **kwargs,
    ) -> Self:
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
            rate = dict.fromkeys(self, rate)
        for key, resamp in rate.items():
            self[key] = self[key].resample(resamp, **kwargs)
        return self

    @classmethod
    def fetch(
        cls,
        channels: list[str | Channel],
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        host: str | None = None,
        port: int | None = None,
        verbose: bool | str = False,
        connection: nds2.connection | None = None,
        verify: bool = False,
        pad: float | None = None,
        allow_tape: bool | None = None,
        scaled: bool | None = None,
        type: int | str | None = None,  # noqa: A002
        dtype: int | str | None = None,
    ) -> Self:
        """Fetch data from NDS for a number of channels.

        This is just a shim around ``TimeSeriesDict.get(..., source='nds2')``.

        Parameters
        ----------
        channels : `str`, `~gwpy.detector.Channel`
            List of names of data channels to find.

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
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

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
        return cls.get(
            channels,
            start,
            end,
            source="nds2",
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
    def fetch_open_data(
        cls,
        detectors: str,
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        sample_rate: float = 4096,
        version: int | None = None,
        format: str = "hdf5",  # noqa: A002
        host: str = GWOSC_DEFAULT_HOST,
        verbose: bool | None = None,
        cache: bool | None = None,
        parallel: int = 1,
        **kwargs,
    ) -> Self:
        """Fetch open-access data from the LIGO Open Science Center.

        This is just a shim around ``TimeSeriesDict.get(..., source='gwosc')``.

        Parameters
        ----------
        detectors : `list` of `str`
            List of two-character prefices of the IFOs in which you
            are interested, e.g. `['H1', 'L1']`.

        start : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS start time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine.

        end : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
            GPS end time of required data,
            any input parseable by `~gwpy.time.to_gps` is fine.

        sample_rate : `float`, `Quantity`,
            The sample rate (Hertz) of desired data; most data are stored
            by GWOSC at 4096 Hz, however there may be event-related
            data releases with a 16384 Hz rate.

        version : `int`
            Version of files to download, defaults to highest discovered
            version.

        format : `str`
            The data format to download and parse.
            One of

            "hdf5"
                HDF5 data files, read using `h5py`.

            "gwf"
                Gravitational-Wave Frame files, requires |LDAStools.frameCPP|_.

        host : `str`
            Host name of GWOSC server to access.

        verbose : `bool`
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

        cache : `bool`
            Save/read a local copy of the remote URL, default: `False`;
            useful if the same remote data are to be accessed multiple times.
            Set `GWPY_CACHE=1` in the environment to auto-cache.

        parallel : `int`
            Number of parallel threads to use when downloading data for
            multiple detectors. Default is ``1``.

        kwargs
            Any other keyword arguments are passed to the `TimeSeries.read`
            method that parses the file that was downloaded.

        See Also
        --------
        TimeSeries.fetch_open_data
            For more examples.

        TimeSeries.read
            For details of how files are read.

        Examples
        --------
        >>> from gwpy.timeseries import TimeSeriesDict
        >>> print(TimeSeriesDict.fetch_open_data(['H1', 'L1'], 1126259446, 1126259478))
        TimeSeriesDict({'H1': <TimeSeries([2.17704028e-19, 2.08763900e-19, 2.39681183e-19, ...,
                     3.55365541e-20, 6.33533516e-20, 7.58121195e-20]
                    unit=Unit(dimensionless),
                    t0=<Quantity 1.12625945e+09 s>,
                    dt=<Quantity 0.00024414 s>,
                    name='Strain',
                    channel=None)>, 'L1': <TimeSeries([-1.04289994e-18, -1.03586274e-18, -9.89322445e-19,
                     ..., -1.01767748e-18, -9.82876816e-19,
                     -9.59276974e-19]
                    unit=Unit(dimensionless),
                    t0=<Quantity 1.12625945e+09 s>,
                    dt=<Quantity 0.00024414 s>,
                    name='Strain',
                    channel=None)>})
        """  # noqa: E501
        return cls.get(
            detectors,
            start,
            end,
            source="gwosc",
            sample_rate=sample_rate,
            version=version,
            format=format,
            host=host,
            verbose=verbose,
            cache=cache,
            parallel=parallel,
            series_class=cls.EntryClass,
            **kwargs,
        )

    @classmethod
    def find(
        cls,
        channels: list[str | Channel],
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        observatory: str | None = None,
        frametype: str | None = None,
        frametype_match: str | re.Pattern | None = None,
        host: str | None = None,
        urltype: str | None = "file",
        ext: str = "gwf",
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

        This is just a shim around ``TimeSeriesDict.get(..., source="gwdatafind")``.

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

        host : `str`, optional
            Name of the GWDataFind server to use.
            Default is set by `gwdatafind.utils.get_default_host`.

        urltype : `str`, optional
            The URL type to use.
            Default is "file" to use paths available on the file system.

        ext : `str`, optional
            The file extension for which to search.
            "gwf" is the only file extension supported, but this may be
            extended in the future.

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
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

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
        return cls.get(
            channels,
            start,
            end,
            source="gwdatafind",
            observatory=observatory,
            frametype=frametype,
            frametype_match=frametype_match,
            host=host,
            urltype=urltype,
            ext=ext,
            pad=pad,
            scaled=scaled,
            allow_tape=allow_tape,
            parallel=parallel,
            verbose=verbose,
            **readargs,
        )

    @classmethod
    def from_arrakis(
        cls,
        block: arrakis.SeriesBlock,
        *,
        copy: bool = True,
        **metadata,
    ) -> Self:
        """Construct a new dict from an `arrakis.SeriesBlock`.

        Parameters
        ----------
        block : `arrakis.SeriesBlock`
            The input Arrakis data to read.

        copy : `bool`, optional
            If `True`, copy the contained data array to new  to a new array.

        metadata
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
    def from_nds2_buffers(
        cls,
        buffers: Iterable[nds2.buffer],
        *,
        scaled: bool | None = None,
        copy: bool = True,
        **metadata,
    ) -> Self:
        """Construct a new dict from a list of `nds2.buffer` objects.

        **Requires:** |nds2|_

        Parameters
        ----------
        buffers : `list` of `nds2.buffer`
            The input NDS2-client buffers to read.

        scaled : `bool`, optional
            Apply slope and bias calibration to ADC data, for non-ADC data
            this option has no effect.

        copy : `bool`, optional
            If `True`, copy the contained data array to new  to a new array.

        metadata
            Any other metadata keyword arguments to pass to the `TimeSeries`
            constructor.

        Returns
        -------
        dict : `TimeSeriesDict`
            A new `TimeSeriesDict` containing the data from the given buffers.
        """
        tsd = cls()
        for buf in buffers:
            tsd[buf.channel.name] = tsd.EntryClass.from_nds2_buffer(
                buf,
                scaled=scaled,
                copy=copy,
                **metadata,
            )
        return tsd

    def plot(
        self,
        label: str = "key",
        method: str = "plot",
        figsize: tuple[float, float] = (12, 4),
        xscale: str = "auto-gps",
        **kwargs,
    ) -> Plot:
        """Plot the data for this `TimeSeriesBaseDict`.

        Parameters
        ----------
        label : `str`, optional
            Labelling system to use, or fixed label for all elements
            Special values include

            - ``'key'``: use the key of the `TimeSeriesBaseDict`,
            - ``'name'``: use the :attr:`~TimeSeries.name` of each element

            If anything else, that fixed label will be used for all lines.

        method : `str`, optional
            The plotting method to use. This can be any method supported by the
            underlying plotting library (e.g., Matplotlib).

        figsize : `tuple[float, float]`, optional
            The size of the figure to create, in inches.

        xscale : `str`, optional
            The scale of the x-axis. This can be one of

            - ``'linear'``: linear scale
            - ``'log'``: logarithmic scale
            - ``'auto-gps'``: automatically determine scale based on GPS time

        kwargs
            All other keyword arguments are passed to the plotter as appropriate.
        """
        kwargs.update({
            "method": method,
            "label": label,
            "figsize": figsize,
            "xscale": xscale,
        })

        # make plot
        from ..plot import Plot

        if kwargs.get("separate", False):
            plot = Plot(*self.values(), **kwargs)
        else:
            plot = Plot(self.values(), **kwargs)

        # update labels
        artmap = {
            "plot": "lines",
            "scatter": "collections",
        }
        artists = [
            x for ax in plot.axes
            for x in getattr(ax, artmap.get(method, "lines"))
        ]
        for key, artist in zip(self, artists, strict=True):
            if label.lower() == "name":
                lab = self[key].name
            elif label.lower() == "key":
                lab = str(key)
            else:
                lab = label
            artist.set_label(lab)

        return plot

    def step(
        self,
        label: str = "key",
        where: Literal["pre", "post", "mid"] = "post",
        figsize: tuple[float, float] = (12, 4),
        xscale: str = "auto-gps",
        **kwargs,
    ) -> Plot:
        """Create a step plot of this dict.

        Parameters
        ----------
        label : `str`, optional
            Labelling system to use, or fixed label for all elements.
            Special values include

            ``'key'``
                Use the key of the `TimeSeriesBaseDict`

            ``'name'``
                Use the :attr:`~TimeSeries.name` of each element

            If anything else, that fixed label will be used for all lines.

        where : `str`, optional
            The location of the step change. This can be one of

            - ``'pre'``: the step change occurs before the x value
            - ``'post'``: the step change occurs after the x value
            - ``'mid'``: the step change occurs at the midpoint of the x value

        figsize : `tuple[float, float]`, optional
            The size of the figure to create, in inches.

        xscale : `str`, optional
            The scale of the x-axis. This can be one of

            - ``'linear'``: linear scale
            - ``'log'``: logarithmic scale
            - ``'auto-gps'``: automatically determine scale based on GPS time

        kwargs
            All other keyword arguments are passed to the plotter as appropriate.
        """
        kwargs.setdefault(
            "drawstyle",
            f"steps-{where}",
        )
        tmp = cast("Self", type(self)())
        for key, series in self.items():
            tmp[key] = series.append(series.value[-1:], inplace=False)

        return tmp.plot(
            label=label,
            figsize=figsize,
            xscale=xscale,
            **kwargs,
        )


# -- TimeSeriesBaseList --------------

# Type variable for generic list entries
_T = TypeVar("_T", bound=TimeSeriesBase)


class TimeSeriesBaseList(list[_T], Generic[_T]):
    """Fancy list representing a list of `TimeSeriesBase`.

    The `TimeSeriesBaseList` provides an easy way to collect and organise
    `TimeSeriesBase` for a single `Channel` over multiple segments.

    Parameters
    ----------
    items
        Any number of `TimeSeriesBase`.

    Returns
    -------
    list
        A new `TimeSeriesBaseList`.

    Raises
    ------
    TypeError
        if any elements are not `TimeSeriesBase`
    """

    EntryClass: ClassVar[type[TimeSeriesBase]] = TimeSeriesBase

    def __init__(self, *items: _T) -> None:
        """Initialise a new list."""
        if len(items) == 1 and isinstance(items[0], (list, tuple)):
            super().__init__(*items)
        else:
            super().__init__()
            for item in items:
                self.append(item)

    @property
    def segments(self) -> SegmentList:
        """The `span` of each series in this list."""
        return SegmentList([item.span for item in self])

    def append(self, item: _T) -> None:
        """Add a new element to the end of the list."""
        if not isinstance(item, self.EntryClass):
            msg = f"Cannot append type '{type(item).__name__}' to {type(self).__name__}"
            raise TypeError(msg)

        super().append(item)

    append.__doc__ = list.append.__doc__

    def extend(self, item: Iterable[_T]) -> None:
        """Add multiple elements to the end of the list."""
        item = TimeSeriesBaseList(*item)
        super().extend(item)

    extend.__doc__ = list.extend.__doc__

    def coalesce(self) -> Self:
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

    def join(
        self,
        pad: float | None = None,
        gap: Literal["raise", "ignore", "pad"] | None = None,
    ) -> _T:
        """Concatenate all of the elements of this list into a single object.

        Parameters
        ----------
        pad : `float`, optional
            Value with which to fill gaps in the source data,
            by default gaps will result in a `ValueError`.

        gap : `str`, optional
            What to do if there are gaps in the data, one of

            ``'raise'``
                Raise a `ValueError`

            ``'ignore'``
                Remove gap and join data

            ``'pad'``
                Pad gap with zeros

            If `pad` is given and is not `None`, the default is ``'pad'``,
            otherwise ``'raise'``.

        Returns
        -------
        series : `gwpy.types.TimeSeriesBase` subclass
             A single series containing all data from each entry in this list.

        See Also
        --------
        TimeSeries.append
            For details on how the individual series are concatenated together.
        """
        if not self:
            return self.EntryClass(numpy.empty((0,) * self.EntryClass._ndim))  # noqa: SLF001
        self.sort(key=lambda t: t.x0.value)
        out = self[0].copy()
        for series in self[1:]:
            out.append(series, gap=gap, pad=pad)
        return out

    @overload
    def __getitem__(self, key: SupportsIndex) -> _T: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...

    def __getitem__(self, key: SupportsIndex | slice) -> Self | _T:
        """Get an item from this list.

        If the key is a slice, return a new `TimeSeriesBaseList` containing
        the sliced elements, otherwise return the element at that index.
        """
        if isinstance(key, slice):
            return type(self)(*super().__getitem__(key))
        return super().__getitem__(key)

    def copy(self) -> Self:
        """Return a copy of this list with each element copied to new memory."""
        out = type(self)()
        for series in self:
            out.append(series.copy())
        return out
