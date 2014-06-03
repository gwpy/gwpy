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

from matplotlib.ticker import MultipleLocator

from .core import Plot
from ..spectrum import Spectrum
from ..utils import with_import

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['BodePlot']


def to_db(a):
    """Convert the input array from amplitude into decibels
    """
    return 10 * numpy.log10(a)


class BodePlot(Plot):
    """An extension of the :class:`~gwpy.plotter.core.Plot` class for
    visualising filters using the Bode representation

    Parameters
    ----------
    *filters : :class:`~scipy.signal.lti`, `tuple`
        any number of linear time-invariant filters to
        display on the plot. If filters are given as tuples, they will
        be interpreted according to the number of elements:

            - 2: (numerator, denominator)
            - 3: (zeros, poles, gain)
            - 4: (A, B, C, D)

    frequencies : `numpy.ndarray`, optional
        list of frequencies (in Hertz) at which to plot
    sample_rate : `float`, optional
        sample_rate (in Hertz) for time-domain filter
    logx : `bool`, optional, default: False
        display frequency on a log-scale
    **kwargs
        other keyword arguments as applicable for the
        :class:`~gwpy.plotter.core.Plot`
    """
    def __init__(self, *filters, **kwargs):
        """Initialise a new TimeSeriesPlot
        """
        dB = kwargs.pop('dB', True)
        frequencies = kwargs.pop('frequencies', None)
        sample_rate = kwargs.pop('sample_rate', None)

        # generate figure
        super(BodePlot, self).__init__(**kwargs)

        # delete the axes, and create two more
        self.add_subplot(2, 1, 1, projection='spectrum')
        self.add_subplot(2, 1, 2, projection='spectrum', sharex=self.maxes)

        # add filters
        for filter_ in filters:
            if isinstance(filter_, Spectrum):
                self.add_spectrum(filter_, dB=dB)
            else:
                self.add_filter(filter_, frequencies=frequencies,
                                sample_rate=sample_rate, dB=dB)

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
        """:class:`~matplotlib.axes.Axes` for the Bode magnitude
        """
        return self.axes[0]

    @property
    def paxes(self):
        """:class:`~matplotlib.axes.Axes` for the Bode phase
        """
        return self.axes[1]

    @with_import('scipy.signal')
    def add_filter(self, filter_, frequencies=None, sample_rate=None,
                   dB=True, **kwargs):
        """Add a linear time-invariant filter to this BodePlot
        """
        if frequencies is None:
            w = None
        else:
            w = frequencies * 2 * pi / numpy.float64(sample_rate)
        if not isinstance(filter_, signal.lti):
            filter_ = signal.lti(*filter_)
        w, h = signal.freqz(filter_.num, filter_.den, w)
        if sample_rate:
            w *= numpy.float64(sample_rate) / (2.0 * pi)
        mag = numpy.absolute(h)
        if dB:
            mag = 2 * to_db(mag)
        phase = numpy.degrees(numpy.unwrap(numpy.angle(h)))
        lm = self.maxes.plot(w, mag, **kwargs)
        lp = self.paxes.plot(w, phase, **kwargs)
        return lm, lp

    def add_spectrum(self, spectrum, dB=True, power=False, **kwargs):
        """Plot the magnitude and phase of a complex-valued Spectrum
        """
        mag = numpy.absolute(spectrum.data)
        if dB:
            mag = to_db(mag)
            if not power:
                mag *= 2.
        phase = numpy.angle(spectrum.data, deg=True)
        w = spectrum.frequencies.data
        lm = self.maxes.plot(w, mag, **kwargs)
        lp = self.paxes.plot(w, phase, **kwargs)
        return lm, lp

