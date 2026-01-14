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

"""A spectral-variation histogram class."""

from __future__ import annotations

from contextlib import suppress
from typing import TYPE_CHECKING

import numpy
from astropy.units import Quantity

from ..io.registry import UnifiedReadWriteMethod
from ..segments import Segment
from ..types import Array2D
from ..types.sliceutils import null_slice
from ..utils.misc import property_alias
from . import FrequencySeries
from .connect import (
    SpectralVarianceRead,
    SpectralVarianceWrite,
)

if TYPE_CHECKING:
    from typing import (
        ClassVar,
        Never,
        Self,
    )

    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike
    from numpy.typing import (
        ArrayLike,
        NDArray,
    )

    from ..detector import Channel
    from ..plot import Plot
    from ..spectrogram import Spectrogram
    from ..time import SupportsToGps
    from ..types import Series
    from ..typing import (
        Array1D,
        ArrayLike1D,
        UnitLike,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ["SpectralVariance"]


def _default_bins(
    data: NDArray,
    nbins: int,
    low: float | None,
    high: float | None,
    unit: UnitBase,
    *,
    log: bool,
) -> Array1D:
    if low is None:
        low = data.min() / 2
    if high is None:
        high = data.max() * 2

    if isinstance(low, Quantity):
        low = low.to(unit).value
    if isinstance(high, Quantity):
        high = high.to(unit).value

    if log:
        low = numpy.log10(low)
        high = numpy.log10(high)

    if log:
        return numpy.logspace(low, high, num=nbins + 1)
    return numpy.linspace(low, high, num=nbins + 1)


class SpectralVariance(Array2D):
    """A variance histogram of a `FrequencySeries`."""

    _metadata_slots: ClassVar[tuple[str, ...]] = (
        *FrequencySeries._metadata_slots,  # noqa: SLF001
        "bins",
    )
    _default_xunit: ClassVar[UnitBase] = FrequencySeries._default_xunit  # noqa: SLF001
    _rowclass: ClassVar[type[Series]] = FrequencySeries

    def __new__(
        cls,
        data: ArrayLike,
        bins: ArrayLike1D,
        unit: UnitLike = None,
        f0: Quantity | float | None = None,
        df: Quantity | float | None = None,
        frequencies: ArrayLike1D | None = None,
        name: str | None = None,
        channel: Channel | str | None = None,
        epoch: SupportsToGps | None = None,
        **kwargs,
    ) -> Self:
        """Generate a new SpectralVariance histogram."""
        # parse x-axis params
        if f0 is not None:
            kwargs["x0"] = f0
        if df is not None:
            kwargs["dx"] = df
        if frequencies is not None:
            kwargs["xindex"] = frequencies

        # generate SpectralVariance using the Series constructor
        new = super(Array2D, cls).__new__(
            cls,
            data,
            unit=unit,
            name=name,
            channel=channel,
            epoch=epoch,
            **kwargs,
        )

        # set bins
        new.bins = bins

        return new

    # -- properties ------------------

    @property
    def bins(self) -> Quantity:
        """Array of bin edges, including the rightmost edge.

        :type: `astropy.units.Quantity`
        """
        return self._bins

    @bins.setter
    def bins(self, bins: QuantityLike | None) -> None:
        if bins is None:
            del self.bins
            return
        bins = Quantity(bins)
        if bins.size != self.shape[1] + 1:
            msg = (
                "SpectralVariance.bins must be given as a list of bin edges, "
                "including the rightmost edge, and have length 1 greater than "
                "the y-axis of the SpectralVariance data"
            )
            raise ValueError(msg)
        self._bins = bins

    @bins.deleter
    def bins(self) -> None:
        with suppress(AttributeError):
            del self._bins

    # over-write yindex and yspan to communicate with bins
    @property  # type: ignore[misc]
    def yindex(self) -> Quantity:
        """List of left-hand amplitude bin edges."""
        return self.bins[:-1]

    @property
    def yspan(self) -> Segment:
        """Amplitude range (low, high) spanned by this array."""
        return Segment(self.bins.value[0], self.bins.value[-1])

    @property  # type: ignore[misc]
    def dy(self) -> Quantity:
        """Size of the first (lowest value) amplitude bin."""
        return self.bins[1] - self.bins[0]

    @property  # type: ignore[misc]
    def y0(self) -> Quantity:
        """Starting value of the first (lowest value) amplitude bin."""
        return self.bins[0]

    f0 = property_alias(
        Array2D.x0,  # type: ignore[arg-type]
        doc="Starting frequency for this `SpectralVariance`.",
    )
    df = property_alias(
        Array2D.dx,  # type: ignore[arg-type]
        doc="Frequency spacing of this `SpectralVariance`.",
    )
    frequencies = property_alias(
        Array2D.xindex,  # type: ignore[arg-type]
        doc="Array of frequencies for each sample",
    )

    @property
    def T(self) -> Never:  # noqa: N802
        """Transpose is not supported."""
        msg = f"transposing a {type(self).__name__} is not supported"
        raise NotImplementedError(msg)

    # -- i/o -------------------------

    read = UnifiedReadWriteMethod(SpectralVarianceRead)
    write = UnifiedReadWriteMethod(SpectralVarianceWrite)

    # -- methods ---------------------

    def __getitem__(
        self,
        item: slice | int | bool | ArrayLike,
    ) -> Self | Series | Quantity:
        """Get a slice of this SpectralVariance."""
        # disable slicing bins
        if isinstance(item, tuple) and not null_slice(item[1]):
            msg = "cannot slice SpectralVariance across bins"
            raise NotImplementedError(msg)
        return super().__getitem__(item)

    @classmethod
    def from_spectrogram(
        cls,
        *spectrograms: Spectrogram,
        bins: ArrayLike1D | None = None,
        low: float | None = None,
        high: float | None = None,
        nbins: int = 500,
        log: bool = False,
        norm: bool = False,
        density: bool = False,
    ) -> Self:
        """Calculate a new `SpectralVariance` from a Spectrogram.

        Parameters
        ----------
        *spectrograms : `~gwpy.spectrogram.Spectrogram`
            Input `Spectrogram` data.

        bins : `~numpy.ndarray`, optional
            Array of histogram bin edges, including the rightmost edge.

        low : `float`, optional
            Left edge of lowest amplitude bin, only read if ``bins`` is not given.

        high : `float`, optional
            Right edge of highest amplitude bin, only read if ``bins`` is not given.

        nbins : `int`, optional
            Number of bins to generate, only read if ``bins`` is not given,
            default: ``500``.

        log : `bool`, optional
            Calculate amplitude bins over a logarithmic scale,
            only read if ``bins`` is not given, default: `False`.

        norm : `bool`, optional
            Normalise bin counts to a unit sum, default: `False`.

        density : `bool`, optional
            Normalise bin counts to a unit integral, default: `False`.

        Returns
        -------
        specvar : `SpectralVariance`
            2D-array of spectral frequency-amplitude counts.

        See Also
        --------
        numpy.histogram
            The histogram function.
        """
        # parse args and kwargs
        if not spectrograms:
            msg = "Must give at least one Spectrogram"
            raise ValueError(msg)
        if norm and density:
            msg = "Cannot give both norm=True and density=True, please pick one"
            raise ValueError(msg)

        # get data and bins
        spectrogram = spectrograms[0]
        data = numpy.vstack([s.value for s in spectrograms])
        if bins is None:
            bins = _default_bins(
                data,
                nbins,
                low,
                high,
                spectrogram.unit,
                log=log,
            )
        else:
            bins = numpy.asarray(bins)
        nbins = bins.size - 1
        qbins = bins * spectrogram.unit

        # loop over frequencies
        out = numpy.zeros((data.shape[1], nbins))
        for i in range(data.shape[1]):
            out[i, :], bins = numpy.histogram(
                data[:, i],
                bins,
                density=density,
            )
            if norm and out[i, :].sum():  # normalise
                out[i, :] /= out[i, :].sum()

        # create and return SpectralVariance
        name = f"{spectrogram.name} variance"
        return cls(
            out,
            qbins,
            epoch=spectrogram.epoch,
            name=name,
            channel=spectrogram.channel,
            f0=spectrogram.f0,
            df=spectrogram.df,
        )

    def percentile(self, percentile: float) -> FrequencySeries:
        """Calculate a given spectral percentile for this `SpectralVariance`.

        Parameters
        ----------
        percentile : `float`
            Percentile (0 - 100) of the bins to compute.

        Returns
        -------
        spectrum : `~gwpy.frequencyseries.FrequencySeries`
            The given percentile `FrequencySeries` calculated from this
            `SpectralVariance`.
        """
        rows, _columns = self.shape
        out = numpy.zeros(rows)
        # Loop over frequencies
        for i in range(rows):
            # Calculate cumulative sum for array
            cumsumvals = numpy.cumsum(self.value[i, :])

            # Find value nearest requested percentile
            minindex = numpy.abs(cumsumvals - percentile).argmin()
            val = self.bins[minindex]
            out[i] = val

        name = f"{self.name} {percentile}% percentile"
        return FrequencySeries(
            out,
            epoch=self.epoch,
            channel=self.channel,
            frequencies=self.bins[:-1],
            name=name,
        )

    def plot(  # type: ignore[override]
        self,
        xscale: str = "log",
        method: str = "pcolormesh",
        **kwargs,
    ) -> Plot:
        """Plot the data for this SpectralVariance."""
        if method == "imshow":
            msg = (
                f"plotting a {type(self).__name__} with {method}() "
                "is not supported"
            )
            raise TypeError(msg)
        bins = self.bins.value
        if (
            numpy.all(bins > 0)
            and numpy.allclose(numpy.diff(numpy.log10(bins), n=2), 0)
        ):
            kwargs.setdefault("yscale", "log")
        kwargs.update(method=method, xscale=xscale)
        return super().plot(**kwargs)
