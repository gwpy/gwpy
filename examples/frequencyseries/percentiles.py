#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>

"""Plotting an averaged ASD with percentiles.

As we have seen in :ref:`gwpy-example-frequencyseries-hoff`, the Amplitude
Spectral Density (ASD) is a key indicator of frequency-domain sensitivity for
a gravitational-wave detector.

However, the ASD doesn't allow you to see how much the sensitivity varies
over time.
One tool for that is the :ref:`spectrogram <gwpy-spectrogram>`, while another
is simply to show percentiles of a number of ASD measurements.

In this example we calculate the median ASD over 2048-seconds surrounding
the GW178017 event, and also the 5th and 95th percentiles of the ASD, and
plot them on a single figure.
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__currentmodule__ = 'gwpy.timeseries'

# First, as always, we get the data using :meth:`TimeSeries.fetch_open_data`:

from gwpy.timeseries import TimeSeries
hoft = TimeSeries.fetch_open_data('H1', 1187007040, 1187009088)

# Next we calculate a :class:`~gwpy.spectrogram.Spectrogram` by calculating
# a number of ASDs, using the :meth:`~gwpy.timeseries.TimeSeries.spectrogram2`
# method:

sg = hoft.spectrogram2(fftlength=4, overlap=2, window='hann') ** (1 / 2.)

# From this we can trivially extract the median, 5th and 95th percentiles:

median = sg.percentile(50)
low = sg.percentile(5)
high = sg.percentile(95)

# Finally, we can make plot, using :meth:`~gwpy.plot.Axes.plot_mmm` to
# display the 5th and 95th percentiles as a shaded region around the median:

from gwpy.plot import Plot
plot = Plot()
ax = plot.add_subplot(
    xscale='log', xlim=(10, 1500), xlabel='Frequency [Hz]',
    yscale='log', ylim=(3e-24, 2e-20),
    ylabel=r'Strain noise [1/$\sqrt{\mathrm{Hz}}$]',
)
ax.plot_mmm(median, low, high, color='gwpy:ligo-hanford')
ax.set_title('LIGO-Hanford strain noise variation around GW170817',
             fontsize=16)
plot.show()

# Now we can see that the ASD varies by factors of a few across most of the
# frequency band, with notable exceptions, e.g. around the 60-Hz power line
# harmonics (60 Hz, 120 Hz, 180 Hz, ...) where the noise is very stable.
