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

"""Plotting a whitened `Spectrogram`

I would like to study the gravitational wave strain spectrogram around the time of an interesting simulated signal during the last science run (S6).
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.spectrogram'

# As with :doc:`previous example <plot>`, we import the
# `~gwpy.timeseries.TimeSeries` class,
# :meth:`~gwpy.timeseries.TimeSeries.get` the data, and calculate a 
# `Spectrogram`
from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.get('H1:LDAS-STRAIN,rds', 'September 16 2010 06:40', 'September 16 2010 06:50')
specgram = gwdata.spectrogram(5, fftlength=2, overlap=1) ** (1/2.)

# To whiten the `specgram` we can use the :meth:`~Spectrogram.ratio` method
# to divide by the overall median:
medratio = specgram.ratio('median')

# Finally, we make a plot:
plot = medratio.plot(norm='log', vmin=0.1, vmax=10)
plot.set_yscale('log')
plot.set_ylim(40, 4096)
plot.add_colorbar(label='Amplitude relative to median')
plot.show()
