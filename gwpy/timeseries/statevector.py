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

"""The boolean `StateTimeSeries` and bit field `StateVector`.

Each bit represents a boolean condition.
Such states are typically the comparison of a `TimeSeries` against some
threshold, where sub-threshold is good and sup-threshold is bad,
for example.

Single `StateTimeSeries` can be bundled together to form `StateVector`
arrays, representing a bit mask of states that combine to make a detailed
statement of instrumental operation
"""

from __future__ import annotations

import os
from contextlib import suppress
from functools import wraps
from math import (
    ceil,
    log2,
)
from typing import (
    TYPE_CHECKING,
    overload,
)

import numpy
from astropy import units

from ..detector import Channel
from ..io.registry import UnifiedReadWriteMethod
from ..segments import Segment
from ..time import Time
from ..types import Array2D
from .connect import (
    StateVectorDictGet,
    StateVectorDictRead,
    StateVectorDictWrite,
    StateVectorGet,
    StateVectorRead,
    StateVectorWrite,
)
from .core import (
    TimeSeriesBase,
    TimeSeriesBaseDict,
    TimeSeriesBaseList,
    as_series_dict_class,
)

if TYPE_CHECKING:
    from collections.abc import (
        Callable,
        Iterable,
        Iterator,
        Sequence,
    )
    from typing import (
        Literal,
        NoReturn,
        Self,
        SupportsIndex,
        TypeAlias,
    )

    import nds2
    from astropy.units import (
        Quantity,
        Unit,
    )
    from numpy.typing import DTypeLike

    from ..plot import Plot
    from ..segments import (
        DataQualityDict,
        DataQualityFlag,
    )
    from ..time import SupportsToGps
    from ..typing import (
        ArrayLike1D,
        UnitLike,
    )

    BitsInput: TypeAlias = dict[int, str | None] | Sequence[str | None]

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = [
    "Bits",
    "StateTimeSeries",
    "StateTimeSeriesDict",
    "StateVector",
    "StateVectorDict",
    "StateVectorList",
]


# -- utilities ----------------------------------------------------------------

def _bool_segments(
    array: Iterable[bool | int],
    start: float = 0,
    delta: float = 1,
    minlen: int = 1,
) -> Iterator[Segment]:
    """Yield segments of consecutive `True` values in a boolean array.

    Parameters
    ----------
    array : `iterable`
        An iterable of boolean-castable values.

    start : `float`
        The value of the first sample on the indexed axis
        (e.g.the GPS start time of the array).

    delta : `float`
        The step size on the indexed axis (e.g. sample duration).

    minlen : `int`, optional
        The minimum number of consecutive `True` values for a segment.

    Yields
    ------
    segment : `tuple`
        ``(start + i * delta, start + (i + n) * delta)`` for a sequence
        of ``n`` consecutive True values starting at position ``i``.

    Notes
    -----
    This method is adapted from original code written by Kipp Cannon and
    distributed under GPLv3.

    The datatype of the values returned will be the larger of the types
    of ``start`` and ``delta``.

    Examples
    --------
    >>> print(list(_bool_segments([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]))
    [(1, 2), (5, 8), (9, 10)]
    >>> print(list(_bool_segments([0, 1, 0, 0, 0, 1, 1, 1, 0, 1]
    ...                           start=100., delta=0.1))
    [(100.1, 100.2), (100.5, 100.8), (100.9, 101.0)]
    """
    array = iter(array)
    i = 0
    while True:
        try:  # get next value
            val = next(array)
        except StopIteration:  # end of array
            return

        if val:  # start of new segment
            n = 1  # count consecutive True
            try:
                while next(array):  # run until segment will end
                    n += 1
            except StopIteration:  # have reached the end
                return  # stop
            finally:  # yield segment (including at StopIteration)
                if n >= minlen:  # ... if long enough
                    yield Segment(start + i * delta, start + (i + n) * delta)
            i += n
        i += 1


# -- StateTimeSeries -----------------

class StateTimeSeries(TimeSeriesBase):
    """Boolean array representing a good/bad state determination.

    Parameters
    ----------
    value : array-like
        Input data array.

    t0 : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    dt : `float`, `~astropy.units.Quantity`, optional
        Time between successive samples (seconds), can also be given inversely
        via `sample_rate`.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        The rate of samples per second (Hertz), can also be given inversely via `dt`.

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

    copy : `bool`, optional
        Choose to copy the input data to new memory.

    subok : `bool`, optional
        Allow passing of sub-classes by the array generator.

    Notes
    -----
    Key methods

    .. autosummary::

       ~StateTimeSeries.to_dqflag
    """

    def __new__(
        cls,
        data: ArrayLike1D,
        t0: SupportsToGps | None = None,
        dt: float | Quantity | None = None,
        sample_rate: float | Quantity | None = None,
        times: ArrayLike1D | None = None,
        channel: Channel | str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> Self:
        """Generate a new StateTimeSeries."""
        if kwargs.pop("unit", None) is not None:
            msg = f"{cls.__name__} does not accept keyword argument 'unit'"
            raise TypeError(msg)

        data = numpy.asarray(data)
        if not isinstance(data, cls):
            data = data.astype(bool, copy=False)

        return super().__new__(
            cls,
            data,
            t0=t0,
            dt=dt,
            sample_rate=sample_rate,
            times=times,
            name=name,
            channel=channel,
            **kwargs,
        )

    # -- unit handling (always dimensionless)

    @property  # type: ignore[misc]
    def unit(self) -> Unit:
        """The unit of this `StateTimeSeries`."""
        return units.dimensionless_unscaled

    def override_unit(
        self,
        unit: UnitLike,
        parse_strict: str = "raise",
    ) -> NoReturn:
        """Override the unit of this `StateTimeSeries`. UNSUPPORTED DO NOT USE."""
        msg = f"overriding units is not supported for {type(self).__name__}"
        raise NotImplementedError(msg)

    # -- math handling (always boolean)

    def __array_ufunc__(
        self,
        function: Callable,
        method: str,
        *inputs,
        **kwargs,
    ) -> Self | Quantity:
        """Handle ufuncs on this `StateTimeSeries`."""
        out = super().__array_ufunc__(function, method, *inputs, **kwargs)
        if out.ndim:
            return out.astype(bool)
        return out

    def diff(
        self,
        n: int = 1,
        axis: int = -1,
    ) -> Self:
        """Return the difference between successive samples.

        The difference is defined as the exclusive-or of the
        previous and current samples.

        Parameters
        ----------
        n : `int`, optional
            Number of times to apply the difference operation,
            defaults to `1`, i.e. the first difference.

        axis : `int`, optional
            Axis along which to compute the difference, defaults to the last
            axis, i.e. the time axis.

        Returns
        -------
        new : `StateTimeSeries`
            A new `StateTimeSeries` containing the difference of the
            current series, with the same metadata as the original.
        """
        slice1 = (slice(1, None),)
        slice2 = (slice(None, -1),)
        new = (self.value[slice1] ^ self.value[slice2]).view(type(self))
        new.__metadata_finalize__(self)
        try:  # shift x0 to the right by one place
            new.x0 = self._xindex[1]
        except AttributeError:
            new.x0 = self.x0 + self.dx
        if n > 1:
            return new.diff(n-1, axis=axis)
        return new

    @wraps(numpy.ndarray.all, assigned=("__doc__",))
    def all(
        self,
        axis: int | None = None,
        out: numpy.ndarray | None = None,
    ) -> bool:
        """Return `True` if all values are `True` along the given axis."""
        return bool(numpy.all(self.value, axis=axis, out=out))

    # -- useful methods --------------

    def to_dqflag(
        self,
        name: str | None = None,
        minlen: int = 1,
        dtype: type | None = None,
        *,
        round: bool = False,  # noqa: A002
        label: str | None = None,
        description: str | None = None,
    ) -> DataQualityFlag:
        """Convert this series into a `~gwpy.segments.DataQualityFlag`.

        Each contiguous set of `True` values are grouped as a
        `~gwpy.segments.Segment` running from the GPS time the first
        found `True`, to the GPS time of the next `False` (or the end
        of the series)

        Parameters
        ----------
        name: `str`, optional
            Name of the segment.

        minlen : `int`, optional
            Minimum number of consecutive `True` values to identify as a
            `~gwpy.segments.Segment`. This is useful to ignore single
            bit flips, for example.

        dtype : `type`, `callable`
            Output segment entry type, can pass either a type for simple
            casting, or a callable function that accepts a float and returns
            another numeric type, defaults to the `dtype` of the time index.

        round : `bool`, optional
            Choose to round each `~gwpy.segments.Segment` to its
            inclusive integer boundaries.

        label : `str`, optional
            The :attr:`~gwpy.segments.DataQualityFlag.label` for the
            output flag.

        description : `str`, optional
            The :attr:`~gwpy.segments.DataQualityFlag.description` for the
            output flag.

        Returns
        -------
        dqflag : `~gwpy.segments.DataQualityFlag`
            A segment representation of this `StateTimeSeries`, the span
            defines the `known` segments, while the contiguous `True`
            sets defined each of the `active` segments.
        """
        from ..segments import DataQualityFlag

        # format dtype
        if dtype is None:
            dtype = self.t0.dtype
        if isinstance(dtype, numpy.dtype):  # use callable dtype
            dtype = dtype.type
        start = dtype(self.t0.value)
        dt = dtype(self.dt.value)

        # build segmentlists (can use simple objects since DQFlag converts)
        active = _bool_segments(self.value, start, dt, minlen=int(minlen))
        known = [tuple(map(dtype, self.span))]

        # build flag and return
        out = DataQualityFlag(
            name=name or self.name,
            active=active,
            known=known,
            label=label or self.name,
            description=description,
        )
        if round:
            return out.round()
        return out

    def to_lal(self) -> NoReturn:
        """Bogus function inherited from superclass, do not use."""
        msg = (
            "The to_lal method, inherited from the TimeSeries, cannot be used with the "
            "StateTimeSeries because LAL has no BooleanTimeSeries structure"
        )
        raise NotImplementedError(msg)

    @classmethod
    def from_nds2_buffer(
        cls,
        buffer: nds2.buffer,
        **metadata,
    ) -> Self:
        """Create a `StateTimeSeries` from an NDS2 buffer."""
        metadata.setdefault("unit", None)
        return super().from_nds2_buffer(buffer, **metadata)

    from_nds2_buffer.__doc__ = TimeSeriesBase.from_nds2_buffer.__doc__

    @overload  # type: ignore[override]
    def __getitem__(self, key: SupportsIndex) -> bool: ...
    @overload
    def __getitem__(self, key: slice) -> Self: ...
    def __getitem__(
        self,
        key: SupportsIndex | slice,
    ) -> bool | Self:
        """Return the value at the given index or slice."""
        if isinstance(key, float | int):
            return numpy.ndarray.__getitem__(self, key)
        return super().__getitem__(key)

    @wraps(TimeSeriesBase.tolist)
    def tolist(self) -> list:
        """Convert this `StateTimeSeries` to a list of boolean values."""
        return self.value.tolist()


# -- Bits ----------------------------

class Bits(list):
    """Definition of the bits in a `StateVector`.

    Parameters
    ----------
    bits : `list`
        list of bit names

    channel : `Channel`, `str`, optional
        data channel associated with this Bits

    epoch : `float`, optional
        defining GPS epoch for this `Bits`

    description : `dict`, optional
        (bit, desc) `dict` of longer descriptions for each bit
    """

    def __init__(
        self,
        bits: BitsInput,
        channel: Channel | str | None = None,
        epoch: Time | float | Quantity | None = None,
        description: dict[str, str] | None = None,
    ) -> None:
        """Initialize a new `Bits` object."""
        if isinstance(bits, dict):
            # dict of (index, bitname) pairs
            depth = max(map(int, bits.keys())) + 1
            super().__init__([None] * depth)
            for key, val in bits.items():
                self[int(key)] = val
        else:
            # list of names
            super().__init__((b or None) for i, b in enumerate(bits))

        # populate metadata
        if channel is not None:
            self.channel = channel
        if epoch is not None:
            self.epoch = epoch

        # populate descriptions
        if description is None:
            description = {}
        self.description = {}
        for i, bit in enumerate(self):
            if bit is None or bit in self.description:
                continue
            if channel:
                self.description[bit] = f"{self.channel} bit {i}"
            else:
                self.description[bit] = None

    @property
    def epoch(self) -> Time | None:
        """Starting GPS time epoch for these `Bits`.

        This attribute is recorded as a `~astropy.time.Time` object in the
        GPS format, allowing native conversion into other formats.

        See :mod:`~astropy.time` for details on the `Time` object.
        """
        try:
            return Time(self._epoch, format="gps")
        except AttributeError:
            return None

    @epoch.setter
    def epoch(self, epoch: Time | float | Quantity) -> None:
        if isinstance(epoch, Time):
            self._epoch = epoch.gps
        elif isinstance(epoch, units.Quantity):
            self._epoch = epoch.value
        else:
            self._epoch = float(epoch)

    @property
    def channel(self) -> Channel | None:
        """Data channel associated with these `Bits`."""
        try:
            return self._channel
        except AttributeError:
            return None

    @channel.setter
    def channel(self, chan: Channel | str) -> None:
        if isinstance(chan, Channel):
            self._channel = chan
        else:
            self._channel = Channel(chan)

    @property
    def description(self) -> dict[str, str | None]:
        """(name, desc) mapping of long bit descriptions."""
        return self._description

    @description.setter
    def description(
        self,
        desc: dict[str, str | None] | None,
    ) -> None:
        if desc is None:
            self._description = {}
        else:
            self._description = desc

    @description.deleter
    def description(self) -> None:
        self._description = {}

    def __repr__(self) -> str:
        """Return a string representation of this `Bits` object."""
        indent = " " * len(f"<{self.__class__.__name__}(")
        mask = (os.linesep + indent).join([
            f"{idx}: {bit!r}" for idx, bit in enumerate(self) if bit
        ])
        return os.linesep.join([
            f"<{self.__class__.__name__}({mask},",
            f"{indent}channel={self.channel!r},",
            f"{indent}epoch={self.epoch!r})>",
        ])

    def __str__(self) -> str:
        """Return a printable string representation of this `Bits` object."""
        indent = " " * len(f"{self.__class__.__name__}(")
        mask = (os.linesep + indent).join([
            f"{idx}: {bit}" for idx, bit in enumerate(self) if bit
        ])
        return os.linesep.join([
            f"{self.__class__.__name__}({mask},",
            f"{indent}channel={self.channel!s},",
            f"{indent}epoch={self.epoch!s})",
        ])

    @wraps(TimeSeriesBase.__array__)
    def __array__(self, dtype: DTypeLike = "U") -> numpy.ndarray:
        """Return a numpy array representation of this `Bits` object."""
        return numpy.array([b or "" for b in self], dtype=dtype)


# -- StateVector ---------------------

class StateVector(TimeSeriesBase):
    """Binary array representing good/bad state determinations of some data.

    Each binary bit represents a single boolean condition, with the
    definitions of all the bits stored in the `StateVector.bits`
    attribute.

    Parameters
    ----------
    value : array-like
        Input data array.

    bits : `Bits`, `list`, optional
        List of bits defining this `StateVector`.

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

    copy : `bool`, optional
        Choose to copy the input data to new memory.

    subok : `bool`, optional
        Allow passing of sub-classes by the array generator.

    Notes
    -----
    Key methods:

    .. autosummary::

        ~StateVector.fetch
        ~StateVector.read
        ~StateVector.write
        ~StateVector.to_dqflags
        ~StateVector.plot

    """

    _metadata_slots = (*TimeSeriesBase._metadata_slots, "bits")
    _print_slots = (*TimeSeriesBase._print_slots, "_bits")

    def __new__(
        cls,
        data: ArrayLike1D,
        bits: BitsInput | None = None,
        t0: SupportsToGps | None = None,
        dt: float | Quantity | None = None,
        sample_rate: float | Quantity | None = None,
        times: ArrayLike1D | None = None,
        channel: Channel | str | None = None,
        name: str | None = None,
        **kwargs,
    ) -> Self:
        """Generate a new `StateVector`."""
        new = super().__new__(
            cls,
            data,
            t0=t0,
            dt=dt,
            sample_rate=sample_rate,
            times=times,
            channel=channel,
            name=name,
            **kwargs,
        )
        new.bits = bits
        return new

    # -- StateVector properties ------

    # -- bits
    @property
    def bits(self) -> Bits:
        """List of `Bits` for this `StateVector`.

        :type: `Bits`
        """
        try:
            return self._bits
        except AttributeError as exc:
            if self.dtype.kind in "iu":
                nbits = self.itemsize * 8
                self.bits = Bits(
                    [f"Bit {b}" for b in range(nbits)],
                    channel=self.channel,
                    epoch=self.epoch,
                )
                return self.bits

            if hasattr(self.channel, "bits"):
                self.bits = self.channel.bits  # type: ignore[union-attr]
                return self.bits

            msg = (
                "cannot determine bits for this StateVector, please set them "
                "explicitly via the 'bits' argument or attribute."
            )
            raise ValueError(msg) from exc

    @bits.setter
    def bits(
        self,
        mask: Bits | BitsInput | None,
    ) -> None:
        if mask is None:
            del self.bits
            return
        if not isinstance(mask, Bits):
            mask = Bits(
                mask,
                channel=self.channel,
                epoch=self.epoch,
            )
        self._bits = mask

    @bits.deleter
    def bits(self) -> None:
        with suppress(AttributeError):
            del self._bits

    # -- boolean
    @property
    def boolean(self) -> Array2D:
        """A 2-D boolean array representation of this `StateVector`."""
        try:
            return self._boolean
        except AttributeError:
            nbits = len(self.bits)
            boolean = numpy.zeros((self.size, nbits), dtype=bool)
            for i, sample in enumerate(self.value):
                boolean[i, :] = [int(sample) >> j & 1 for j in range(nbits)]
            self._boolean = Array2D(
                boolean,
                name=self.name,
                x0=self.x0,
                dx=self.dx,
                y0=0,
                dy=1,
            )
            return self.boolean

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(StateVectorRead)
    write = UnifiedReadWriteMethod(StateVectorWrite)
    get = UnifiedReadWriteMethod(StateVectorGet)

    # -- StateVector methods ---------

    def get_bit_series(
        self,
        bits: Iterable[int | str] | None = None,
    ) -> StateTimeSeriesDict:
        """Get the `StateTimeSeries` for each bit of this `StateVector`.

        Parameters
        ----------
        bits : `list`, optional
            A list of bit indices or bit names, defaults to all bits.

        Returns
        -------
        bitseries : `StateTimeSeriesDict`
            A `dict` of `StateTimeSeries`, one for each given bit.
        """
        if bits is None:
            bits = [b for b in self.bits if b]
        bindex = []
        try:
            for bit in bits:
                if isinstance(bit, int):
                    bindex.append((bit, self.bits[bit]))
                else:
                    bindex.append((self.bits.index(bit), bit))
        except (
            IndexError,
            ValueError,
        ) as exc:
            exc.args = (f"Bit {bit!r} not found in StateVector",)
            raise

        self._bitseries = StateTimeSeriesDict()
        for i, bit in bindex:
            self._bitseries[bit] = StateTimeSeries(
                self.value >> i & 1,
                name=bit,
                epoch=self.x0.value,
                channel=self.channel,
                sample_rate=self.sample_rate,
            )
        return self._bitseries

    def to_dqflags(
        self,
        bits: Iterable[int | str] | None = None,
        minlen: int = 1,
        dtype: type = float,
        *,
        round: bool = False,  # noqa: A002
    ) -> DataQualityDict:
        """Convert this `StateVector` into a `~gwpy.segments.DataQualityDict`.

        The `StateTimeSeries` for each bit is converted into a
        `~gwpy.segments.DataQualityFlag` with the bits combined into a dict.

        Parameters
        ----------
        bits : `list`, optional
            A list of bit indices or bit names to select,
            defaults to `~StateVector.bits`.

        minlen : `int`, optional
           Minimum number of consecutive `True` values to identify as a
           `Segment`. This is useful to ignore single bit flips, for example.

        dtype : `type`, optional
            Output segment entry type, can pass either a type for simple
            casting, or a callable function that accepts a float and returns
            another numeric type, defaults to `float`.

        round : `bool`, optional
            Choose to round each `Segment` to its inclusive integer boundaries.

        Returns
        -------
        DataQualityFlag list : `list`
            A list of `~gwpy.segments.flag.DataQualityFlag`
            representations for each bit in this `StateVector`.

        See Also
        --------
        StateTimeSeries.to_dqflag
            For details on the segment representation method for `StateVector` bits.
        """
        from ..segments import DataQualityDict

        out = DataQualityDict()
        bitseries = self.get_bit_series(bits=bits)
        for bit, sts in bitseries.items():
            name = str(bit)
            out[bit] = sts.to_dqflag(
                name=name,
                minlen=minlen,
                round=round,
                dtype=dtype,
                description=self.bits.description[name],
            )
        return out

    @classmethod
    def fetch(
        cls,
        channel: str | Channel,
        start: SupportsToGps,
        end: SupportsToGps,
        *,
        bits: Bits | BitsInput | None = None,
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
        """Fetch data from NDS into a `StateVector`.

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

        bits : `Bits`, `list`, optional
            Definition of bits for this `StateVector`.

        host : `str`, optional
            URL of NDS server to use, if blank will try any server
            (in a relatively sensible order) to get the data

            One of ``connection`` or ``host`` must be given.

        port : `int`, optional
            Port number for NDS server query, must be given with `host`.

        verify : `bool`, optional
            Check channels exist in database before asking for data.
            Default is `True`.

        pad : `float`, optional
            Value with which to fill gaps in the source data.
            By default gaps will result in a `ValueError`.

        verbose : `bool`, optional
            This argument is deprecated and will be removed in a future release.
            Use DEBUG-level logging instead, see :ref:`gwpy-logging`.

        connection : `nds2.connection`, optional
            Open NDS connection to use.
            Default is to open a new connection using ``host`` and ``port``
            arguments.

            One of ``connection`` or ``host`` must be given.

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
        new = super().fetch(
            channel,
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
        )
        if bits:
            new.bits = bits
        return new

    def plot(  # type: ignore[override]
        self,
        format: Literal["timeseries", "segments"] = "segments",  # noqa: A002
        bits: Iterable[int | str] | None = None,
        **kwargs,
    ) -> Plot:
        """Plot the data for this `StateVector`.

        Parameters
        ----------
        format : `str`, optional
            The type of plot to make, either 'segments' to plot the
            SegmentList for each bit, or 'timeseries' to plot the raw
            data for this `StateVector`.

        bits : `list`, optional
            A list of bit indices or bit names, defaults to
            `~StateVector.bits`. This argument is ignored if ``format`` is
            not ``'segments'``.

        kwargs
            Other keyword arguments to be passed to either
            `~gwpy.plot.SegmentAxes.plot` or `~gwpy.plot.Axes.plot`, depending
            on ``format``.

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            Output plot object.

        See Also
        --------
        matplotlib.pyplot.figure
            For documentation of keyword arguments used to create the figure.

        matplotlib.figure.Figure.add_subplot
            For documentation of keyword arguments used to create the axes.

        gwpy.plot.SegmentAxes.plot_flag
            For documentation of keyword arguments used in rendering each
            statevector flag.
        """
        if format == "timeseries":
            return super().plot(**kwargs)

        if format == "segments":
            from ..plot import Plot

            kwargs.setdefault("xscale", "auto-gps")
            return Plot(
                *self.to_dqflags(bits=bits).values(),
                projection="segments",
                **kwargs,
            )

        msg = "'format' argument must be one of: 'timeseries' or 'segments'"
        raise ValueError(msg)

    def resample(self, rate: float | Quantity) -> StateVector:
        """Resample this `StateVector` to a new rate.

        Because of the nature of a state-vector, downsampling is done
        by taking the logical 'and' of all original samples in each new
        sampling interval, while upsampling is achieved by repeating
        samples.

        Parameters
        ----------
        rate : `float`
            Rate to which to resample this `StateVector`, must be a
            divisor of the original sample rate (when downsampling)
            or a multiple of the original (when upsampling).

        Returns
        -------
        vector : `StateVector`
            Resampled version of the input `StateVector`.
        """
        rate1 = self.sample_rate.value
        if isinstance(rate, units.Quantity):
            rate2 = rate.value
        else:
            rate2 = float(rate)

        # upsample
        if (rate2 / rate1).is_integer():
            msg = "StateVector upsampling has not been implemented yet, sorry."
            raise NotImplementedError(msg)

        # downsample
        if (rate1 / rate2).is_integer():
            factor = int(rate1 / rate2)
            # reshape incoming data to one column per new sample
            newsize = int(self.size / factor)
            old = self.value.reshape((newsize, self.size // newsize))
            # work out number of bits
            if self.bits:
                nbits = len(self.bits)
            else:
                max_ = self.value.max()
                nbits = ceil(log2(max_)) if max_ else 1
            bits = range(nbits)
            # construct an iterator over the columns of the old array
            itr = numpy.nditer(
                [old, None],
                flags=["external_loop", "reduce_ok"],
                op_axes=[None, [0, -1]],  # type: ignore[list-item]
                op_flags=[["readonly"], ["readwrite", "allocate"]],
            )
            dtype = self.dtype
            type_ = self.dtype.type
            # for each new sample, each bit is logical AND of old samples
            # bit is ON,
            for x, y in itr:
                y[...] = numpy.sum(
                    [type_((x >> bit & 1).all() * (2 ** bit)) for bit in bits],
                    dtype=self.dtype,
                )
            new = StateVector(itr.operands[1], dtype=dtype)
            new.__metadata_finalize__(self)
            new._unit = self.unit
            new.sample_rate = rate2
            return new

        # error for non-integer resampling factors
        if rate1 < rate2:
            msg = (
                "New sample rate must be multiple of input series rate if "
                "upsampling a StateVector"
            )
            raise ValueError(msg)
        msg = (
            "New sample rate must be divisor of input series rate if "
            "downsampling a StateVector"
        )
        raise ValueError(msg)


@as_series_dict_class(StateTimeSeries)
class StateTimeSeriesDict(TimeSeriesBaseDict):
    """Dictionary of `StateTimeSeries` objects."""

    __doc__ = TimeSeriesBaseDict.__doc__.replace("TimeSeriesBase", "StateTimeSeries")  # type: ignore[union-attr]
    EntryClass = StateTimeSeries


@as_series_dict_class(StateVector)
class StateVectorDict(TimeSeriesBaseDict):
    """Dictionary of `StateVector` objects."""

    __doc__ = TimeSeriesBaseDict.__doc__.replace("TimeSeriesBase", "StateVector")  # type: ignore[union-attr]
    EntryClass = StateVector

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(StateVectorDictRead)
    write = UnifiedReadWriteMethod(StateVectorDictWrite)
    get = UnifiedReadWriteMethod(StateVectorDictGet)


class StateVectorList(TimeSeriesBaseList):
    """List of `StateVector` objects."""

    __doc__ = TimeSeriesBaseList.__doc__.replace("TimeSeriesBase", "StateVector")  # type: ignore[union-attr]
    EntryClass = StateVector
