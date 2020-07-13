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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""Calculating and plotting a `SpectralVariance` histogram

The most common visualisation of the spectral content of a data series is
via the power or amplitude spectral density calculations (PSD or ASD).
However, these typically involve averaging of the data over a period, which
can wash out transient noise (as is often desired).

The `SpectralVariance` histogram provide by `gwpy.frequencyseries` allows
us to look at the spectral sensitivity in a different manner, displaying
which frequencies sit at which amplitude _most_ of the time, but also
highlighting excursions from normal behaviour.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# To demonstate this, we can load some data from the LIGO Livingston
# intereferometer around the time of the GW151226 gravitational wave detection:

from gwpy.timeseries import TimeSeries
llo = TimeSeries.fetch_open_data('L1', 1135136228, 1135140324, verbose=True)

# We can then call the :meth:`~gwpy.timeseries.TimeSeries.spectral_variance`
# method of the ``llo`` `~gwpy.timeseries.TimeSeries` by calculating an ASD
# every 5 seconds and counting the amount of time each frequency bin spends
# at each ASD value:

variance = llo.spectral_variance(5, fftlength=2, overlap=1, log=True,
                                 low=1e-24, high=1e-19, nbins=100)

# We can then :meth:`~SpectralVariance.plot` the `SpectralVariance`

plot = variance.plot(yscale='log', norm='log', vmin=.5, cmap='plasma')
ax = plot.gca()
ax.grid()
ax.set_xlim(20, 1500)
ax.set_ylim(1e-24, 1e-20)
ax.set_xlabel('Frequency [Hz]')
ax.set_ylabel(r'[strain/$\sqrt{\mathrm{Hz}}$]')
ax.set_title('LIGO-Livingston sensitivity variance')
plot.show()

# From this we see that in general the sensitivity varies a few parts in
# 10 :sup:`-23`, while many of the lines (narrow-band peaks in the spectrum)
# are much more stationary.
