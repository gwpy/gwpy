# Copyright (c) 2014-2017 Louisiana State University
#               2017-2022 Cardiff University
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

"""Spectrogram object."""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy
from astropy.units import Quantity

from ..frequencyseries import FrequencySeries
from ..frequencyseries._fdcommon import _fdfilter
from ..io.registry import UnifiedReadWriteMethod
from ..timeseries import (
    TimeSeries,
    TimeSeriesList,
)
from ..timeseries.core import _format_time
from ..types import (
    Array2D,
    Series,
)
from ..utils.misc import property_alias
from .connect import (
    SpectrogramRead,
    SpectrogramWrite,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        ClassVar,
        Literal,
        Self,
    )

    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike
    from numpy.typing import ArrayLike

    from ..detector import Channel
    from ..frequencyseries import SpectralVariance
    from ..plot import Plot
    from ..signal.filter_design import FilterCompatible
    from ..time import SupportsToGps
    from ..types.sliceutils import SliceLike
    from ..typing import (
        ArrayLike1D,
        UnitLike,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

__all__ = ["Spectrogram", "SpectrogramList"]


def _ordinal(value: float) -> str:
    """Return the ordinal string for a given integer.

    See https://stackoverflow.com/a/20007730/1307974

    Parameters
    ----------
    value : `float`
        The number to convert to ordinal.

    Examples
    --------
    >>> _ordinal(11)
    '11th'
    >>> _ordinal(102)
    '102nd'
    """
    n = int(str(value)[-1])  # last digit

    # Numbers ending >=4 use 'th' as the ordinal suffix
    th_boundary = 4

    idx = int((n // 10 % 10 != 1) * (n % 10 < th_boundary) * n % 10)
    return f"{value}{'tsnrhtdd'[idx::4]}"


class Spectrogram(Array2D):
    """A 2D array holding a spectrogram of time-frequency data.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        The rate of samples per second (Hertz).

    times : `array-like`
        The complete array of GPS times accompanying the data for this series.
        This argument takes precedence over `epoch` and `sample_rate` so should
        be given in place of these if relevant, not alongside.

    f0 : `float`, `~astropy.units.Quantity`, optional
        Starting frequency for these data.

    df : `float`, `~astropy.units.Quantity`, optional
        Frequency resolution for these data.

    frequencies : `array-like`
        The complete array of frequencies indexing the data.
        This argument takes precedence over `f0` and `df` so should
        be given in place of these if relevant, not alongside.

    epoch : `~gwpy.time.LIGOTimeGPS`, `float`, `str`, optional
        GPS epoch associated with these data,
        any input parsable by `~gwpy.time.to_gps` is fine.

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

    Notes
    -----
    Key methods:

    .. autosummary::

       ~Spectrogram.read
       ~Spectrogram.write
       ~Spectrogram.plot
       ~Spectrogram.zpk
    """

    _metadata_slots: ClassVar[tuple[str, ...]] = (
        *Series._metadata_slots,  # noqa: SLF001
        "y0",
        "dy",
        "yindex",
    )
    _default_xunit: ClassVar[UnitBase] = TimeSeries._default_xunit  # noqa: SLF001
    _default_yunit: ClassVar[UnitBase] = FrequencySeries._default_xunit  # noqa: SLF001
    _rowclass = TimeSeries
    _columnclass = FrequencySeries

    def __new__(
        cls,
        data: ArrayLike,
        unit: UnitLike = None,
        t0: SupportsToGps | None = None,
        dt: Quantity | float | None = None,
        f0: Quantity | float | None = None,
        df: Quantity | float | None = None,
        times: ArrayLike1D | None = None,
        frequencies: ArrayLike1D | None = None,
        name: str | None = None,
        channel: Channel | str | None = None,
        **kwargs,
    ) -> Self:
        """Generate a new Spectrogram."""
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
        if dt is not None:
            kwargs["dx"] = dt
        # parse times
        if times is not None:
            kwargs["xindex"] = times

        # parse y-axis params
        if f0 is not None:
            kwargs["y0"] = f0
        if df is not None:
            kwargs["dy"] = df
        if frequencies is not None:
            kwargs["yindex"] = frequencies

        # generate Spectrogram
        return super().__new__(
            cls,
            data,
            unit=unit,
            name=name,
            channel=channel,
            **kwargs,
        )

    # -- Spectrogram properties ------

    epoch = property_alias(
        TimeSeries.epoch,  # type: ignore[arg-type]
        "GPS epoch for these data.",
    )
    t0 = property_alias(
        TimeSeries.t0,
        "GPS time of first time bin.",
    )
    dt = property_alias(
        TimeSeries.dt,
        "Time (seconds) between successive bins.",
    )
    span = property_alias(
        TimeSeries.span,
        "GPS [start, stop) span for these data.",
    )
    f0 = property_alias(
        Array2D.y0,  # type: ignore[arg-type]
        "Starting frequency for these data.",
    )
    df = property_alias(
        Array2D.dy,  # type: ignore[arg-type]
        "Frequency spacing for these data.",
    )
    times = property_alias(
        Array2D.xindex,  # type: ignore[arg-type]
        "Series of GPS times for each sample",
    )
    frequencies = property_alias(
        Array2D.yindex,  # type: ignore[arg-type]
        "Series of frequencies for these data.",
    )
    band = property_alias(
        Array2D.yspan,  # type: ignore[arg-type]
        "Frequency band described by these data.",
    )

    # -- Spectrogram i/o -------------

    read = UnifiedReadWriteMethod(SpectrogramRead)
    write = UnifiedReadWriteMethod(SpectrogramWrite)

    # -- Spectrogram methods ---------

    def ratio(
        self,
        operand: FrequencySeries | Quantity | Literal["mean", "median"],
    ) -> Spectrogram:
        """Calculate the ratio of this `Spectrogram` against a reference.

        Parameters
        ----------
        operand : `str`, `FrequencySeries`, `Quantity`
            A `~gwpy.frequencyseries.FrequencySeries` or `~astropy.units.Quantity`
            to weight against, or one of

            ``'mean'``
                Weight against the mean of each spectrum in this Spectrogram.

            ``'median'``
                Weight against the median of each spectrum in this Spectrogram.

        Returns
        -------
        spectrogram : `Spectrogram`
            A new `Spectrogram`.

        Raises
        ------
        ValueError
            If ``operand`` is given as a `str` that isn't supported.
        """
        if isinstance(operand, str):
            if operand == "mean":
                operand = self.mean(axis=0)
            elif operand == "median":
                operand = self.median(axis=0)
            else:
                msg = (
                    f"operand {operand!r} unrecognised, please give a "
                    "Quantity or one of: 'mean', 'median'"
                )
                raise ValueError(msg)
        return self / operand

    def plot(
        self,
        method: str = "pcolormesh",
        figsize: tuple[float, float] = (12, 6),
        xscale: str = "auto-gps",
        **kwargs,
    ) -> Plot:
        """Plot the data for this `Spectrogram`.

        Parameters
        ----------
        method : `str`, optional
            The plotting method to use to render this spectrogram,
            either ``'pcolormesh'`` (default) or ``'imshow'``.

        figsize : `tuple` of `float`, optional
            ``(width, height)`` (inches) of the output figure.

        xscale : `str`, optional
            The X-axis scale.

        kwargs
            All keyword arguments are passed along to underlying
            functions, see below for references.

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            The `Plot` containing the data.

        See Also
        --------
        matplotlib.pyplot.figure
            For documentation of keyword arguments used to create the figure.
        matplotlib.figure.Figure.add_subplot
            For documentation of keyword arguments used to create the axes.
        gwpy.plot.Axes.imshow
        gwpy.plot.Axes.pcolormesh
            For documentation of keyword arguments used in rendering the
            `Spectrogram` data.
        """
        return super().plot(
            method=method,
            figsize=figsize,
            xscale=xscale,
            **kwargs,
        )

    @classmethod
    def from_spectra(
        cls,
        *spectra: FrequencySeries,
        **kwargs,
    ) -> Spectrogram:
        """Build a new `Spectrogram` from a list of spectra.

        Parameters
        ----------
        *spectra : `~gwpy.frequencyseries.FrequencySeries`
            One or more frequency series to stack.

        dt : `float`, `~astropy.units.Quantity`, optional
            Stride between given spectra.

        kwargs
            Other keyword arguments to pass to the constructor.

        Returns
        -------
        Spectrogram
            A new `Spectrogram` from a vertical stacking of the spectra
            The new object takes the metadata from the first given
            `~gwpy.frequencyseries.FrequencySeries` if not given explicitly.

        Notes
        -----
        Each `~gwpy.frequencyseries.FrequencySeries` passed to this
        constructor must be the same length.
        """
        data = numpy.vstack([s.value for s in spectra])
        spec1 = spectra[0]
        if not all(s.f0 == spec1.f0 for s in spectra):
            msg = "cannot stack spectra with different f0"
            raise ValueError(msg)
        if not all(s.df == spec1.df for s in spectra):
            msg = "cannot stack spectra with different df"
            raise ValueError(msg)
        kwargs.setdefault("name", spec1.name)
        kwargs.setdefault("channel", spec1.channel)
        kwargs.setdefault("epoch", spec1.epoch)
        kwargs.setdefault("f0", spec1.f0)
        kwargs.setdefault("df", spec1.df)
        kwargs.setdefault("unit", spec1.unit)
        if (
            "dt" not in kwargs
            and "times" not in kwargs
        ):
            try:
                kwargs["dt"] = spectra[1].epoch.gps - spec1.epoch.gps  # type: ignore[union-attr]
            except (
                AttributeError,
                IndexError,
            ) as exc:
                msg = "cannot determine dt (time-spacing) for Spectrogram from inputs"
                raise ValueError(msg) from exc
        return cls(data, **kwargs)

    def percentile(
        self,
        percentile: float,
    ) -> FrequencySeries:
        """Calculate a given spectral percentile for this `Spectrogram`.

        Parameters
        ----------
        percentile : `float`
            percentile (0 - 100) of the bins to compute

        Returns
        -------
        spectrum : `~gwpy.frequencyseries.FrequencySeries`
            the given percentile `FrequencySeries` calculated from this
            `SpectralVaraicence`
        """
        out = numpy.percentile(self.value, percentile, axis=0)
        ordnl = f"{_ordinal(percentile)} percentile"
        if self.name is not None:
            name = f"{self.name}: {ordnl}"
        else:
            name = ordnl
        return FrequencySeries(
            out,
            epoch=self.epoch,
            channel=self.channel,
            name=name,
            f0=self.f0,
            df=self.df,
            unit=self.unit,
            frequencies=(hasattr(self, "_frequencies") and self.frequencies) or None,
        )

    def zpk(
        self,
        zeros: ArrayLike1D,
        poles: ArrayLike1D,
        gain: float,
        *,
        analog: bool = False,
        sample_rate: QuantityLike | None = None,
        unit: str = "rad/s",
        normalize_gain: bool = False,
    ) -> Self:
        """Filter this `Spectrogram` by applying a zero-pole-gain filter.

        Parameters
        ----------
        zeros : `array-like`
            List of zero frequencies (in Hertz).

        poles : `array-like`
            List of pole frequencies (in Hertz).

        gain : `float`
            DC gain of filter.

        analog : `bool`, optional
            Type of ZPK being applied, if `analog=True` all parameters
            will be converted in the Z-domain for digital filtering.

        sample_rate : `float`, `~astropy.units.Quantity`, optional
            Sample rate of data (in Hertz), used to apply a digital filter.
            Defaults to the last frequency value of this `Spectrogram`
            (i.e. the Nyquist frequency).

        unit : `str`, optional
            For analogue ZPK filters, the units in which the zeros and poles are
            specified. Either ``'Hz'`` or ``'rad/s'`` (default).

        normalize_gain : `bool`, optional
            Whether to normalize the gain when converting from Hz to rad/s.

            - `False` (default):
              Multiply zeros/poles by -2π but leave gain unchanged.
              This matches the LIGO GDS **'f' plane** convention
              (``plane='f'`` in ``s2z()``).

            - `True`:
              Normalize gain to preserve frequency response magnitude.
              Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
              Use this when your filter was designed with the transfer
              function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
              This matches the LIGO GDS **'n' plane** convention
              (``plane='n'`` in ``s2z()``).

            Only used for analogue filters in Hz (``analog=True, unit="Hz"``).

        Returns
        -------
        specgram : `Spectrogram`
            The frequency-domain filtered version of the input data.

        See Also
        --------
        Spectrogram.filter
            For details on how a digital ZPK-format filter is applied.

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(
            (zeros, poles, gain),
            analog=analog,
            unit=unit,
            sample_rate=sample_rate,
            normalize_gain=normalize_gain,
        )

    def filter(
        self,
        filt: FilterCompatible,
        *,
        analog: bool = False,
        sample_rate: QuantityLike | None = None,
        unit: str = "rad/s",
        normalize_gain: bool = False,
        inplace: bool = False,
        **kwargs,
    ) -> Self:
        """Apply the given filter to this `Spectrogram`.

        Parameters
        ----------
        filt : `numpy.ndarray` or `tuple`
            The filter to be applied.
            This can be specified in any of the following forms, with
            the appropriate number of elements in the tuple:

            - `numpy.ndarray` - 1D array of FIR filter coefficients.
            - `tuple[numpy.ndarray, numpy.ndarray]` - numerator/demoinator
              polynomials of the transfer function.
            - `numpy.ndarray` - 2D array of SOS coefficients.
            - `tuple[numpy.ndarray, numpy.ndarray, float]` - zero-pole-gain
              representation.

        analog : `bool`, optional
            If `True`, filter definition will be converted from Hertz
            to Z-domain digital representation, default: `False`.

        sample_rate : `float`, `~astropy.units.Quantity`, optional
            Sample rate of data (in Hertz), used to apply a digital filter.
            Defaults to the last frequency value of this `Spectrogram`
            (i.e. the Nyquist frequency).

        unit : `str`, optional
            For analogue ZPK filters, the units in which the zeros and poles are
            specified. Either ``'Hz'`` or ``'rad/s'`` (default).

        normalize_gain : `bool`, optional
            Whether to normalize the gain when converting from Hz to rad/s.

            - `False` (default):
              Multiply zeros/poles by -2π but leave gain unchanged.
              This matches the LIGO GDS **'f' plane** convention
              (``plane='f'`` in ``s2z()``).

            - `True`:
              Normalize gain to preserve frequency response magnitude.
              Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
              Use this when your filter was designed with the transfer
              function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
              This matches the LIGO GDS **'n' plane** convention
              (``plane='n'`` in ``s2z()``).

            Only used for analogue filters in Hz (``analog=True, unit="Hz"``).

        inplace : `bool`, optional
            If `True`, this array will be overwritten with the filtered
            version, default: `False`.

        kwargs
            Additional keyword arguments passed to the filter function.

        Returns
        -------
        result : `Spectrogram`
            The filtered version of the input `Spectrogram`,
            if ``inplace=True`` was given, this is just a reference to
            the modified input array.

        Raises
        ------
        ValueError
            If ``filt`` arguments cannot be interpreted properly.
        """
        return cast("Self", _fdfilter(
            self,
            filt,
            analog=analog,
            sample_rate=sample_rate,
            unit=unit,
            normalize_gain=normalize_gain,
            inplace=inplace,
            **kwargs,
        ))

    def variance(
        self,
        bins: ArrayLike1D | None = None,
        low: float | None = None,
        high: float | None = None,
        nbins: int = 500,
        *,
        log: bool = False,
        norm: bool = False,
        density: bool = False,
    ) -> SpectralVariance:
        """Calculate the `SpectralVariance` of this `Spectrogram`.

        Parameters
        ----------
        bins : `~numpy.ndarray`, optional, default `None`
            array of histogram bin edges, including the rightmost edge
        low : `float`, optional, default: `None`
            left edge of lowest amplitude bin, only read
            if ``bins`` is not given
        high : `float`, optional, default: `None`
            right edge of highest amplitude bin, only read
            if ``bins`` is not given
        nbins : `int`, optional, default: `500`
            number of bins to generate, only read if ``bins`` is not
            given
        log : `bool`, optional, default: `False`
            calculate amplitude bins over a logarithmic scale, only
            read if ``bins`` is not given
        norm : `bool`, optional, default: `False`
            normalise bin counts to a unit sum
        density : `bool`, optional, default: `False`
            normalise bin counts to a unit integral

        Returns
        -------
        specvar : `SpectralVariance`
            2D-array of spectral frequency-amplitude counts

        See Also
        --------
        numpy.histogram
            for details on specifying bins and weights
        """
        from ..frequencyseries import SpectralVariance

        return SpectralVariance.from_spectrogram(
            self,
            bins=bins,
            low=low,
            high=high,
            nbins=nbins,
            log=log,
            norm=norm,
            density=density,
        )

    # -- Spectrogram connectors ------

    def crop_frequencies(
        self,
        low: float | Quantity | None = None,
        high: float | Quantity | None = None,
        *,
        copy: bool = False,
    ) -> Spectrogram:
        """Crop this `Spectrogram` to the specified frequencies.

        Parameters
        ----------
        low : `float`, optional
            Lower frequency bound for cropped `Spectrogram`.

        high : `float`, optional
            Upper frequency bound for cropped `Spectrogram`.

        copy : `bool`, optional
            If `False` return a view of the original data,
            otherwise create a fresh memory copy.

        Returns
        -------
        spec : `Spectrogram`
            A new `Spectrogram` with a subset of data from the frequency
            axis
        """
        # Convert floats to Quantities
        if low is not None:
            low = Quantity(low, self._default_yunit)
        if high is not None:
            high = Quantity(high, self._default_yunit)

        # Cast for type checker
        low = cast("Quantity | None", low)
        high = cast("Quantity | None", high)

        # Check low frequency
        if low is not None and low == self.f0:
            low = None
        elif low is not None and low < self.f0:
            warnings.warn(
                "Spectrogram.crop_frequencies given low frequency "
                "cutoff below f0 of the input Spectrogram. Low "
                "frequency crop will have no effect.",
                stacklevel=2,
            )

        # Check high frequency
        peak = Quantity(self.band[1], self.yunit)
        if high is not None and high == peak:
            high = None
        elif high is not None and high > peak:
            warnings.warn(
                "Spectrogram.crop_frequencies given high frequency "
                "cutoff above cutoff of the input Spectrogram. High "
                "frequency crop will have no effect.",
                stacklevel=2,
            )

        # Find low index
        if low is None:
            idx0 = None
        else:
            idx0 = int(float(low.value - self.f0.value) // self.df.value)
        # find high index
        if high is None:
            idx1 = None
        else:
            idx1 = int(float(high.value - self.f0.value) // self.df.value)

        # Crop
        new = self[:, idx0:idx1]
        if copy:
            return new.copy()
        return new

    # -- Spectrogram ufuncs ----------

    def _wrap_function(
        self,
        function: Callable,
        *args,  # noqa: ANN002
        **kwargs,
    ) -> Self | FrequencySeries | TimeSeries | Quantity:
        """Wrap a numpy function."""
        out = super()._wrap_function(function, *args, **kwargs)

        # requested frequency axis, return a FrequencySeries
        if out.ndim == 1 and out.x0.unit == self.y0.unit:
            return self._columnclass(
                out.value,
                name=out.name,
                unit=out.unit,
                epoch=out.epoch,
                channel=out.channel,
                f0=out.x0.value,
                df=out.dx.value,
            )

        # requested time axis, return a TimeSeries
        if out.ndim == 1:
            return self._rowclass(
                out.value,
                name=out.name,
                unit=out.unit,
                epoch=out.epoch,
                channel=out.channel,
                dx=out.dx,
            )

        # otherwise return whatever we got back from super (Quantity)
        return out

    _wrap_function.__doc__ = Array2D._wrap_function.__doc__  # noqa: SLF001

    # -- other -----------------------

    def __getitem__(
        self,
        item: SliceLike | tuple[SliceLike, ...],
    ) -> Self | FrequencySeries:
        """Return a slice of this spectrogram."""
        out = super().__getitem__(item)

        # set epoch manually, because Spectrogram doesn't store self._epoch
        if isinstance(out, self._columnclass):
            out.epoch = self.epoch

        return out


class SpectrogramList(TimeSeriesList):
    """Fancy list representing a list of `Spectrogram`.

    The `SpectrogramList` provides an easy way to collect and organise
    `Spectrogram` for a single `Channel` over multiple segments.

    Parameters
    ----------
    items
        Any number of `Spectrogram` series.

    Returns
    -------
    list
        A new `SpectrogramList`.

    Raises
    ------
    TypeError
        If any elements are not of type `Spectrogram`.
    """

    EntryClass: ClassVar[type[Spectrogram]] = Spectrogram  # type: ignore[assignment]
