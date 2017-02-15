#!/usr/bin/env python

# Copyright (C) Duncan Macleod (2013-2015)
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

"""Plotting a Rayleigh-statistic `Spectum`

In LIGO the 'Rayleigh' statistic is a calculation of the
`coefficient of variation
<https://en.wikipedia.org/wiki/Coefficient_of_variation>`_ of the
power spectral density (PSD) of a given set of data.
It is used to measure the 'Gaussianity' of those data, where a value of 1
indicates Gaussian behaviour, less than 1 indicates coherent variations,
and greater than 1 indicates incoherent variation.
It is a useful measure of the quality of the strain data being generated
and recorded at a LIGO site.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.frequencyseries'

# To demonstate, we can download some public LIGO data from the sixth science
# run (S6) for the H1 interferometer:

from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.fetch_open_data(
    'H1', 'September 16 2010 07:00', 'September 16 2010 07:10',
    verbose=True)

# Next, we can calculate a Rayleigh statistic `FrequencySeries` using the 
# :meth:`~gwpy.timeseries.TimeSeries.rayleigh_spectrum` method of the
# `~gwpy.timeseries.TimeSeries` with a 2-second FFT and 1-second overlap (50%):

rayleigh = gwdata.rayleigh_spectrum(2, 1)

# For easy comparison, we can calculate the spectral sensitivity ASD of the
# strain data and plot both on the same figure:

asd = gwdata.asd(2, 1)
plot = asd.plot()
plot.add_frequencyseries(rayleigh, newax=True, sharex=plot.axes[0])
plot.axes[0].set_xlabel('')
plot.axes[0].set_xlim(40, 2000)
plot.axes[0].set_ylim(1e-23, 5e-21)
plot.axes[0].set_ylabel(r'[strain/\rtHz]')
plot.axes[1].set_ylabel('Rayleigh statistic')
plot.show()

# So, we see sharp dips at certain frequencies associated with 'lines' in
# spectrum where noise at a fixed frequency is very coherent (e.g. harmonics
# of the 60Hz mains power supply), and bumps in specific frequency bands
# where the interferometer noise is non-stationary.
