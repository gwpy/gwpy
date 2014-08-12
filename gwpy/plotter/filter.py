# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Definition of a BodePlot
"""

from math import pi
import numpy

from scipy import signal
from matplotlib.ticker import MultipleLocator

from .core import Plot
from ..spectrum import Spectrum

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['BodePlot']


def to_db(a):
    """Convert the input array into decibels

    Parameters
    ----------
    a : `float`, `numpy.ndarray`
        value or array of values to convert to decibels

    Returns
    -------
    dB : ``10 * numpy.log10(a)``
    """
    return 10 * numpy.log10(a)


class BodePlot(Plot):
    """A `Plot` class for visualising transfer functions.

    Parameters
    ----------
    *filters : `~scipy.signal.lti`, `~gwpy.spectrum.Spectrum`, `tuple`
        any number of the following:

        - linear time-invariant filters, either
          `~scipy.signal.lti` or `tuple` of the following length and form:
             - 2: (numerator, denominator)
             - 3: (zeros, poles, gain)
             - 4: (A, B, C, D)

        - complex-valued `spectra <gwpy.spectrum.Spectrum>` representing
          a transfer function

    frequencies : `numpy.ndarray`, optional
        list of frequencies (in Hertz) at which to plot
    db : `bool`, optional, default: `True`
        if `True`, display magnitude in decibels, otherwise display
        amplitude.
    **kwargs
        other keyword arguments as applicable for `Plot` or
        :meth:`~SpectrumAxes.plot`

    Returns
    -------
    plot : `BodePlot`
        a new BodePlot with two `Axes` - :attr:`~BodePlot.maxes` and
        :attr:`~BodePlot.paxes` - representing the magnitude and
        phase of the input transfer function(s) respectively.
    """
    def __init__(self, *filters, **kwargs):
        """Initialise a new TimeSeriesPlot
        """
        dB = kwargs.pop('dB', True)
        frequencies = kwargs.pop('frequencies', None)

        # parse plotting arguments
        figargs = dict()
        for key in ['figsize', 'dpi', 'facecolor', 'edgecolor', 'linewidth',
                    'frameon', 'subplotpars', 'tight_layout']:
            if key in kwargs:
                figargs[key] = kwargs.pop(key)

        # generate figure
        super(BodePlot, self).__init__(**figargs)

        # delete the axes, and create two more
        self.add_subplot(2, 1, 1, projection='spectrum')
        self.add_subplot(2, 1, 2, projection='spectrum', sharex=self.maxes)

        # add filters
        for filter_ in filters:
            if isinstance(filter_, Spectrum):
                self.add_spectrum(filter_, dB=dB, **kwargs)
            else:
                self.add_filter(filter_, frequencies=frequencies,
                                dB=dB, **kwargs)

        # format plots
        if dB:
            self.maxes.set_ylabel('Magnitude [dB]')
        else:
            self.maxes.set_yscale('log')
            self.maxes.set_ylabel('Amplitude')
        self.paxes.set_xlabel('Frequency [Hz]')
        self.paxes.set_ylabel('Phase [deg]')
        self.maxes.set_xscale('log')
        self.paxes.set_xscale('log')
        self.paxes.yaxis.set_major_locator(MultipleLocator(base=90))
        self.paxes.set_ylim(-180, 180)

        # get xlim
        if (frequencies is None and len(filters) == 1 and
                isinstance(filters[0], Spectrum)):
            frequencies = filters[0].frequencies.data
        if frequencies is not None:
            frequencies = frequencies[frequencies > 0]
            self.maxes.set_xlim(frequencies.min(), frequencies.max())

    @property
    def maxes(self):
        """`SpectrumAxes` for the Bode magnitude
        """
        return self.axes[0]

    @property
    def paxes(self):
        """`SpectrumAxes` for the Bode phase
        """
        return self.axes[1]

    def add_filter(self, filter_, frequencies=None, dB=True, **kwargs):
        """Add a linear time-invariant filter to this BodePlot

        Parameters
        ----------
        filter_ : `~scipy.signal.lti`, `tuple`
            the filter to plot, either as a `~scipy.signal.lti`, or a
            `tuple` with the following number and meaning of elements

                 - 2: (numerator, denominator)
                 - 3: (zeros, poles, gain)
                 - 4: (A, B, C, D)

        frequencies : `numpy.ndarray`, optional
            list of frequencies (in Hertz) at which to plot
        db : `bool`, optional, default: `True`
            if `True`, display magnitude in decibels, otherwise display
            amplitude.
        **kwargs
            any other keyword arguments accepted by
            :meth:`~matplotlib.axes.Axes.plot`

        Returns
        -------
        mag, phase : `tuple` of `lines <matplotlib.lines.Line2D>`
            the lines drawn for the magnitude and phase of the filter.
        """
        if frequencies is None:
            w = None
        else:
            w = frequencies * 2. * pi
        if not isinstance(filter_, signal.lti):
            filter_ = signal.lti(*filter_)
        w, h = signal.freqs(filter_.num, filter_.den, w)
        w /= (2. * pi)
        mag = numpy.absolute(h)
        if dB:
            mag = 2 * to_db(mag)
        phase = numpy.degrees(numpy.unwrap(numpy.angle(h)))
        lm = self.maxes.plot(w, mag, **kwargs)
        lp = self.paxes.plot(w, phase, **kwargs)
        return lm, lp

    def add_spectrum(self, spectrum, dB=True, power=False, **kwargs):
        """Plot the magnitude and phase of a complex-valued Spectrum

        Parameters
        ----------
        spectrum : `~gwpy.spectrum.Spectrum`
            the (complex-valued) `Spectrum` to display
        db : `bool`, optional, default: `True`
            if `True`, display magnitude in decibels, otherwise display
            amplitude.
        power : `bool`, optional, default: `False`
            given `True` to incidate that ``spectrum`` holds power values,
            so ``dB = 10 * log(abs(spectrum))``, otherwise
            ``db = 20 * log(abs(spectrum))``. This argument is ignored if
            ``db=False``.
        **kwargs
            any other keyword arguments accepted by
            :meth:`~matplotlib.axes.Axes.plot`

        Returns
        -------
        mag, phase : `tuple` of `lines <matplotlib.lines.Line2D>`
            the lines drawn for the magnitude and phase of the filter.
        """
        # parse spectrum arguments
        kwargs.setdefault('label', spectrum.name)
        # get magnitude
        mag = numpy.absolute(spectrum.data)
        if dB:
            mag = to_db(mag)
            if not power:
                mag *= 2.
        # get phase
        phase = numpy.angle(spectrum.data, deg=True)
        # plot
        w = spectrum.frequencies.data
        lm = self.maxes.plot(w, mag, **kwargs)
        lp = self.paxes.plot(w, phase, **kwargs)
        return lm, lp
