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

"""Plotting a `Spectrogram` of the Rayleigh statistic

As described in :ref:`gwpy-example-frequencyseries-rayleigh`, the Rayleigh
statistic can be used to study non-Gaussianity in a timeseries.
We can study the time variance of these features by plotting a
time-frequency spectrogram where we calculate the Rayleigh statistic for
each time bin.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.spectrogram'

# To demonstate this, we can load some data from the LIGO Livingston
# intereferometer around the time of the GW151226 gravitational wave detection:

from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.fetch_open_data('L1', 'Dec 26 2015 03:37',
                                    'Dec 26 2015 03:47', verbose=True)

# Next, we can calculate a Rayleigh statistic `Spectrogram` using the
# :meth:`~gwpy.timeseries.TimeSeries.rayleigh_spectrogram` method of the
# `~gwpy.timeseries.TimeSeries` and a 5-second stride with a 2-second FFT and
# 1-second overlap (50%):
rayleigh = gwdata.rayleigh_spectrogram(5, fftlength=2, overlap=1)

# and can make a plot using the :meth:`~Spectrogram.plot` method
plot = rayleigh.plot(norm='log', vmin=0.25, vmax=4)
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(30, 1500)
ax.set_title('Sensitivity of LIGO-Livingston around GW151226')
ax.colorbar(cmap='coolwarm', label='Rayleigh statistic')
plot.show()
