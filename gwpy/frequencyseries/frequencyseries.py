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

"""Representation of a frequency series."""

from __future__ import annotations

import warnings
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy
from astropy import units
from numpy import fft as npfft

from ..io.registry import UnifiedReadWriteMethod
from ..types import Series
from ..utils.misc import property_alias
from .connect import (
    FrequencySeriesRead,
    FrequencySeriesWrite,
)

if TYPE_CHECKING:
    from typing import (
        ClassVar,
        Self,
    )

    import pycbc.types
    from astropy.units import (
        Quantity,
        UnitBase,
    )

    from ..detector import Channel
    from ..plot import Plot
    from ..signal.filter_design import FilterCompatible
    from ..time import SupportsToGps
    from ..timeseries import TimeSeries
    from ..typing import (
        ArrayLike1D,
        QuantityLike,
        UnitLike,
    )
    from ..utils.lal import LALFrequencySeriesType

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class FrequencySeries(Series):
    """A data array holding some metadata to represent a frequency series.

    Parameters
    ----------
    value : array-like
        Input data array.

    unit : `~astropy.units.Unit`, optional
        Physical unit of these data.

    f0 : `float`, `~astropy.units.Quantity`, optional, default: `0`
        Starting frequency for these data.

    df : `float`, `~astropy.units.Quantity`, optional, default: `1`
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

       ~FrequencySeries.read
       ~FrequencySeries.write
       ~FrequencySeries.plot
       ~FrequencySeries.zpk
    """

    _default_xunit: ClassVar[UnitBase] = units.hertz
    _print_slots: ClassVar[tuple[str, ...]] = (
        "f0",
        "df",
        "epoch",
        "name",
        "channel",
    )

    def __new__(
        cls,
        data: ArrayLike1D,
        unit: UnitLike = None,
        f0: Quantity | float | None = None,
        df: Quantity | float | None = None,
        frequencies: ArrayLike1D | None = None,
        name: str | None = None,
        epoch: SupportsToGps | None = None,
        channel: Channel | str | None = None,
        **kwargs,
    ) -> Self:
        """Generate a new FrequencySeries."""
        if f0 is not None:
            kwargs["x0"] = f0
        if df is not None:
            kwargs["dx"] = df
        if frequencies is not None:
            kwargs["xindex"] = frequencies

        # generate FrequencySeries
        return super().__new__(
            cls,
            data,
            unit=unit,
            name=name,
            channel=channel,
            epoch=epoch,
            **kwargs,
        )

    # -- FrequencySeries properties --

    f0 = property_alias(Series.x0, "Starting frequency for this `FrequencySeries`")  # type: ignore[arg-type]
    df = property_alias(Series.dx, "Frequency spacing of this `FrequencySeries`")  # type: ignore[arg-type]
    frequencies = property_alias(Series.xindex, "Series of frequencies for each sample")  # type: ignore[arg-type]

    # -- FrequencySeries i/o ---------

    read = UnifiedReadWriteMethod(FrequencySeriesRead)
    write = UnifiedReadWriteMethod(FrequencySeriesWrite)

    # -- FrequencySeries methods -----

    def plot(
        self,
        method: str = "plot",
        xscale: str = "log",
        **kwargs,
    ) -> Plot:
        """Plot the data for this `FrequencySeries`."""
        # use log y-scale for ASD, PSD
        u = self.unit
        try:
            hzpow = u.powers[u.bases.index(units.Hz)]  # type: ignore[union-attr]
        except ValueError:
            pass
        else:
            if hzpow < 0:
                kwargs.setdefault("yscale", "log")

        return super().plot(method=method, xscale=xscale, **kwargs)

    def ifft(self) -> TimeSeries:
        """Compute the one-dimensional discrete inverse Fourier transform.

        Returns
        -------
        out : :class:`~gwpy.timeseries.TimeSeries`
            The normalised, real-valued `TimeSeries`.

        See Also
        --------
        numpy.fft.irfft
            The inverse (real) FFT function.

        Notes
        -----
        This method applies the necessary normalisation such that the
        condition holds:

        >>> timeseries = TimeSeries([1.0, 0.0, -1.0, 0.0], sample_rate=1.0)
        >>> timeseries.fft().ifft() == timeseries
        """
        from ..timeseries import TimeSeries

        nout = (self.size - 1) * 2
        # Undo normalization from TimeSeries.fft
        # The DC component does not have the factor of two applied
        # so we account for it here
        array = self.value.copy()
        array[1:] /= 2.0
        dift = npfft.irfft(
            array * nout,
            n=nout,
        )
        return TimeSeries(
            dift,
            epoch=self.epoch,
            channel=self.channel,
            name=self.name,
            unit=self.unit,
            dx=(1 / self.dx / nout).to("s"),
        )

    def zpk(
        self,
        zeros: ArrayLike1D,
        poles: ArrayLike1D,
        gain: float,
        *,
        analog: bool = True,
        sample_rate: QuantityLike | None = None,
        unit: str = "rad/s",
        normalize_gain: bool = False,
    ) -> Self:
        """Filter this `FrequencySeries` by applying a zero-pole-gain filter.

        Parameters
        ----------
        zeros : `array-like`
            List of zero frequencies (in Hertz).

        poles : `array-like`
            List of pole frequencies (in Hertz).

        gain : `float`
            DC gain of filter.

        analog : `bool`, optional
            Type of ZPK being applied, if ``analog=True`` all parameters
            will be converted in the Z-domain for digital filtering via
            the bilinear transform.

        sample_rate : `float`, `~astropy.units.Quantity`, optional
            Sample rate of data (in Hertz), used to apply a digital filter.
            Defaults to the last frequency value of this `FrequencySeries`
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
        spectrum : `FrequencySeries`
            The frequency-domain filtered version of the input data.

        See Also
        --------
        FrequencySeries.filter
            For details on how a digital ZPK-format filter is applied.

        Examples
        --------
        To apply a zpk filter with file poles at 100 Hz, and five zeros at
        1 Hz (giving an overall DC gain of 1e-10)::

            >>> data2 = data.zpk([100]*5, [1]*5, 1e-10)
        """
        return self.filter(
            (zeros, poles, gain),
            sample_rate=sample_rate,
            analog=analog,
            unit=unit,
            normalize_gain=normalize_gain,
        )

    def interpolate(self, df: float) -> Self:
        """Interpolate this `FrequencySeries` to a new resolution.

        Parameters
        ----------
        df : `float`
            Desired frequency resolution of the interpolated `FrequencySeries`, in Hz.

        Returns
        -------
        out : `FrequencySeries`
            The interpolated version of the input `FrequencySeries`.

        See Also
        --------
        numpy.interp
            For the underlying 1-D linear interpolation scheme.
        """
        f0 = self.f0.decompose().value
        n_samples = (self.size - 1) * (self.df.decompose().value / df) + 1
        fsamples = numpy.arange(
            0,
            numpy.rint(n_samples),
            dtype=self.real.dtype,
        ) * df + f0
        out = type(self)(
            numpy.interp(fsamples, self.frequencies.value, self.value),
        )
        out.__array_finalize__(self)
        out.f0 = f0
        out.df = df
        return out

    def filter(
        self,
        filt: FilterCompatible,
        *,
        analog: bool = False,
        sample_rate: QuantityLike | None = None,
        unit: str = "rad/s",
        normalize_gain: bool = False,
        inplace: bool = False,
    ) -> Self:
        """Apply a filter to this `FrequencySeries`.

        The input filter argument is designed to accept any filter created
        by the :mod:`scipy.signal` filter design functions, and operates on
        the conventions of that module.

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
            Type of ZPK being applied, if ``analog=True`` all parameters
            will be converted in the Z-domain for digital filtering via
            the bilinear transform.

        sample_rate : `float`, `~astropy.units.Quantity`, optional
            Sample rate of data (in Hertz), used to apply a digital filter.
            Defaults to the last frequency value of this `FrequencySeries`
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

        inplace : bool, optional
            If `True`, this array will be overwritten with the filtered
            version.

        Returns
        -------
        result : `FrequencySeries`
            The filtered version of the input `FrequencySeries`,
            if ``inplace=True`` was given, this is just a reference to
            the modified input array.

        Raises
        ------
        ValueError
            If ``filt`` arguments cannot be interpreted properly.

        See Also
        --------
        FrequencySeries.zpk
            For applying a zero-pole-gain filter, including in other
            units (e.g. poles and zeros specified in Hertz).
        """
        from ._fdcommon import _fdfilter
        return cast("Self", _fdfilter(
            self,
            filt,
            analog=analog,
            inplace=inplace,
            sample_rate=sample_rate,
            unit=unit,
            normalize_gain=normalize_gain,
        ))

    @classmethod
    def from_lal(
        cls,
        lalfs: LALFrequencySeriesType,
        *,
        copy: bool = True,
    ) -> Self:
        """Generate a new `FrequencySeries` from a LAL `FrequencySeries`.

        Any type of LAL FrequencySeries is supported.
        """
        # convert units
        from ..utils.lal import from_lal_unit

        try:
            unit = from_lal_unit(lalfs.sampleUnits)
        except TypeError:
            unit = None

        # create a new series
        return cls(
            lalfs.data.data,
            name=lalfs.name or None,
            unit=unit,
            f0=lalfs.f0,
            df=lalfs.deltaF,
            epoch=lalfs.epoch,
            channel=None,
            copy=copy,
        )

    def to_lal(self) -> LALFrequencySeriesType:
        """Convert this `FrequencySeries` into a LAL FrequencySeries.

        Returns
        -------
        lalspec : `FrequencySeries`
            An XLAL-format FrequencySeries of a given type, e.g.
            `REAL8FrequencySeries`.
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

        # convert epoch
        epoch = lal.LIGOTimeGPS(0 if self.epoch is None else self.epoch.gps)

        # create FrequencySeries
        create = find_typed_function(self.dtype, "Create", "FrequencySeries")
        lalfs = create(
            self.name or "",
            epoch,
            self.f0.value,
            self.df.value,
            unit,
            self.shape[0],
        )
        lalfs.data.data = self.value * scale

        return lalfs

    @classmethod
    def from_pycbc(
        cls,
        fs: pycbc.types.TimeSeries,
        *,
        copy: bool = True,
    ) -> Self:
        """Convert a `pycbc.types.frequencyseries.FrequencySeries`.

        Parameters
        ----------
        fs : `pycbc.types.frequencyseries.FrequencySeries`
            The input PyCBC `~pycbc.types.frequencyseries.FrequencySeries` array.

        copy : `bool`, optional
            If `True`, copy these data to a new array.

        Returns
        -------
        spectrum : `FrequencySeries`
            A GWpy version of the input frequency series.
        """
        return cls(
            fs.data,
            f0=0,
            df=fs.delta_f,
            epoch=fs.epoch,
            copy=copy,
        )

    def to_pycbc(self, *, copy: bool = True) -> pycbc.types.FrequencySeries:
        """Convert this `FrequencySeries` into a PyCBC FrequencySeries.

        Parameters
        ----------
        copy : `bool`, optional
            If `True`, copy these data to a new array.

        Returns
        -------
        frequencyseries : `pycbc.types.frequencyseries.FrequencySeries`
            A PyCBC representation of this `FrequencySeries`.
        """
        from pycbc import types

        epoch = None if self.epoch is None else self.epoch.gps
        if self.f0.to("Hz").value:
            msg = (
                f"Cannot convert FrequencySeries to PyCBC with f0 = {self.f0}."
                " Starting frequency must be equal to 0 Hz."
            )
            raise ValueError(msg)
        return types.FrequencySeries(
            self.value,
            delta_f=self.df.to("Hz").value,
            epoch=epoch,
            copy=copy,
        )
