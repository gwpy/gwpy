#!/usr/bin/env python

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

"""Plotting a `Spectrogram`

I would like to study the gravitational wave strain spectrogram around the time of an interesting simulated signal during the last science run (S6).
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.spectrogram'

# First, we import the :class:`~gwpy.timeseries.TimeSeries` and :meth:`~gwpy.timeseries.TimeSeries.get` the data:
from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.get(
    'H1:LDAS-STRAIN', 'September 16 2010 06:40', 'September 16 2010 06:50',
    verbose=True)

# Next, we can calculate a `Spectrogram` using the 
# :meth:`~gwpy.timeseries.TimeSeries.spectrogram` method of the #
# `~gwpy.timeseries.TimeSeries` and a 5-second stride with a 2-second FFT and 
# 1-second overlap (50%):
specgram = gwdata.spectrogram(5, fftlength=2, overlap=1) ** (1/2.)

# and can make a plot using the :meth:`~Spectrogram.plot` method
plot = specgram.plot(norm='log', vmin=1e-23, vmax=1e-19)
plot.set_yscale('log')
plot.set_ylim(40, 4000)
plot.add_colorbar(
    label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
plot.show()
