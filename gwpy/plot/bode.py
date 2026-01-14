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

"""Definition of a BodePlot."""

from __future__ import annotations

from math import pi
from typing import (
    TYPE_CHECKING,
    cast,
)

import numpy
from matplotlib.ticker import MaxNLocator

from ..signal import filter_design
from . import Plot

if TYPE_CHECKING:
    from astropy.units.typing import QuantityLike
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from numpy.typing import (
        ArrayLike,
        NDArray,
    )

    from ..frequencyseries import FrequencySeries
    from ..signal.filter_design import FilterType
    from ..typing import Array1D

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "BodePlot",
]


def _to_db(a: ArrayLike, *, power: bool = False) -> NDArray:
    """Convert the input array into decibels.

    Parameters
    ----------
    a : `float`, `numpy.ndarray`
        Value or array of values to convert to decibels.

    power : `bool`, optional
        If `True`, convert power-based quantity to dB (:math:`10 * log_{10}(a)`),
        otherwise (default) convert amplitude (:math:`20 * log_{10}(a)`).

    Returns
    -------
    dB : `float`
        The input value converted to decibels.

    Examples
    --------
    >>> to_db(1000)
    30.0
    """
    db = 10 * numpy.log10(a)
    if power:
        return db
    return 2 * db


class BodePlot(Plot):
    """A `Plot` class for visualising transfer functions.

    Parameters
    ----------
    filters : `~scipy.signal.lti`, `~gwpy.frequencyseries.FrequencySeries`
        Any number of the following:

        - linear time-invariant filters, either
          `~scipy.signal.lti` or `tuple` of the following length and form:
          - 2: (numerator, denominator)
          - 3: (zeros, poles, gain)
          - 4: (A, B, C, D)

        - complex-valued `spectra <gwpy.frequencyseries.FrequencySeries>`
          representing a transfer function

    frequencies : `numpy.ndarray`, `int`, optional
        List of frequencies (in Hertz) at which to plot, or an integer
        specifying the number of frequencies to generate.

    dB : `bool`, optional
        If `True`, display magnitude in decibels, otherwise display
        amplitude.

    analog : `bool`, optional
        If `True`, indicates that the input filters are analogue filters.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        The sampling rate of a digital filter.
        If ``analog=False`` this option is required.

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

    kwargs
        All other keyword arguments are passed to
        :meth:`~FrequencySeriesAxes.plot`.

    Returns
    -------
    plot : `BodePlot`
        A new BodePlot with two `Axes` - :attr:`~BodePlot.maxes` and
        :attr:`~BodePlot.paxes` - representing the magnitude and
        phase of the input transfer function(s) respectively.
    """

    def __init__(
        self,
        *filters: FilterType,
        dB: bool = True,  # noqa: N803
        frequencies: Array1D | int | None = None,
        **kwargs,
    ) -> None:
        """Initialise a new `BodePlot`."""
        from ..frequencyseries import FrequencySeries

        # parse plotting arguments
        figargs = {}
        title = kwargs.pop("title", None)
        for key in (
            "figsize",
            "dpi",
            "frameon",
            "subplotpars",
            "tight_layout",
        ):
            if key in kwargs:
                figargs[key] = kwargs.pop(key)

        # generate figure
        super().__init__(**figargs)

        # delete the axes, and create two more
        self.add_subplot(2, 1, 1)
        self.add_subplot(2, 1, 2, sharex=self.maxes)

        # add filters
        for filter_ in filters:
            if isinstance(filter_, FrequencySeries):
                self.add_frequencyseries(filter_, dB=dB, **kwargs)
            else:
                self.add_filter(
                    filter_,
                    frequencies=frequencies,
                    dB=dB,
                    **kwargs,
                )

        # format plots
        if dB:
            self.maxes.set_ylabel("Magnitude [dB]")
            ylim = self.maxes.get_ylim()
            if ylim[1] == 0:
                self.maxes.set_ybound(
                    upper=ylim[1] + (ylim[1] - ylim[0]) * 0.1)
        else:
            self.maxes.set_yscale("log")
            self.maxes.set_ylabel("Amplitude")
        if not kwargs.get("analog", False) or kwargs.get("unit") == "Hz":
            self.paxes.set_xlabel("Frequency [Hz]")
        else:
            self.paxes.set_xlabel("Frequency [rad/s]")
        self.paxes.set_ylabel("Phase [deg]")
        self.maxes.set_xscale("log")
        self.paxes.set_xscale("log")
        self.paxes.yaxis.set_major_locator(
            MaxNLocator(nbins="auto", steps=[4.5, 9]),
        )
        self.paxes.yaxis.set_minor_locator(
            MaxNLocator(nbins=20, steps=[4.5, 9]),
        )
        if title:
            self.maxes.set_title(title)

        # get xlim
        if (
            frequencies is None
            and len(filters) == 1
            and isinstance(filters[0], FrequencySeries)
        ):
            frequencies = filters[0].frequencies.value
        if not isinstance(frequencies, type(None) | int):
            frequencies = numpy.asarray(frequencies)
            frequencies = frequencies[frequencies > 0]
            self.maxes.set_xlim(frequencies.min(), frequencies.max())

    @property
    def maxes(self) -> Axes:
        """`FrequencySeriesAxes` for the Bode magnitude."""
        return self.axes[0]

    @property
    def paxes(self) -> Axes:
        """`FrequencySeriesAxes` for the Bode phase."""
        return self.axes[1]

    def add_filter(
        self,
        filter_: FilterType,
        frequencies: Array1D | int | None = None,
        *,
        dB: bool = True,  # noqa: N803
        analog: bool = False,
        sample_rate: QuantityLike = 1.0,
        unit: str | None = None,
        normalize_gain: bool = False,
        **kwargs,
    ) -> tuple[Line2D, Line2D]:
        """Add a linear time-invariant filter to this BodePlot.

        Parameters
        ----------
        filter_ : `~scipy.signal.lti`, `tuple`
            The filter to plot, either as a `~scipy.signal.lti`, or a
            `tuple` with the following number and meaning of elements

            - 2: (numerator, denominator)
            - 3: (zeros, poles, gain)
            - 4: (A, B, C, D)

        frequencies : `numpy.ndarray`, optional
            List of frequencies (in Hertz) at which to plot.

        dB : `bool`, optional
            If `True`, display magnitude in decibels, otherwise display
            amplitude, default: `True`.

        analog : `bool`, optional
            If `True`, indicates that ``filter_`` is an analogue filter.

        sample_rate : `float`, `~astropy.units.Quantity`, optional
            The sampling rate of a digital filter.
            If ``analog=False`` this option is required.

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

        kwargs
            All other keyword arguments are passed to
            :meth:`~matplotlib.axes.Axes.plot`.

        Returns
        -------
        mag, phase : `matplotlib.lines.Line2D`
            The lines drawn for the magnitude and phase of the filter.

        Raises
        ------
        ValueError
            If ``analog=False`` is given and ``sample_rate`` isn't.
        """
        frequencies, fresp = filter_design.frequency_response(
            filter_,
            frequencies,
            analog=analog,
            sample_rate=sample_rate,
            unit=unit or "rad/s",
            normalize_gain=normalize_gain,
        )
        if frequencies[0] == 0:
            frequencies = frequencies[1:]
            fresp = fresp[1:]

        mag = abs(fresp)
        if dB:
            mag = _to_db(mag, power=False)
        if analog:
            phase = numpy.unwrap(numpy.arctan2(fresp.imag, fresp.real)) * 180.0 / pi
        else:
            phase = numpy.rad2deg(numpy.unwrap(numpy.angle(fresp)))

        # draw
        mline = self.maxes.plot(frequencies, mag, **kwargs)[0]
        pline = self.paxes.plot(frequencies, phase, **kwargs)[0]
        return mline, pline

    def add_frequencyseries(
        self,
        spectrum: FrequencySeries,
        *,
        dB: bool = True,  # noqa: N803
        power: bool = False,
        **kwargs,
    ) -> tuple[Line2D, Line2D]:
        """Plot the magnitude and phase of a complex-valued `FrequencySeries`.

        Parameters
        ----------
        spectrum : `~gwpy.frequencyseries.FrequencySeries`
            the (complex-valued) `FrequencySeries` to display

        dB : `bool`, optional
            If `True`, display magnitude in decibels, otherwise display
            amplitude.

        power : `bool`, optional
            Give `True` to incidate that ``spectrum`` holds power values,
            so ``dB = 10 * log(abs(spectrum))``, otherwise
            ``db = 20 * log(abs(spectrum))``.
            This argument is ignored if ``db=False``.

        **kwargs
            All other keyword arguments are passed to
            :meth:`~matplotlib.axes.Axes.plot`.

        Returns
        -------
        mag, phase : `matplotlib.lines.Line2D`
            The lines drawn for the magnitude and phase of the filter.
        """
        # parse spectrum arguments
        kwargs.setdefault("label", spectrum.name)

        # get magnitude
        mag = numpy.absolute(cast("Array1D", spectrum.value))
        if dB:
            mag = _to_db(mag, power=power)

        # get phase
        phase = numpy.angle(spectrum.value, deg=True)

        # plot
        w = spectrum.frequencies.value
        mline = self.maxes.plot(w, mag, **kwargs)[0]
        pline = self.paxes.plot(w, phase, **kwargs)[0]
        return mline, pline
