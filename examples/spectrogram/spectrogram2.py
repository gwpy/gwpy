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

"""Plotting an over-dense, short-duration `Spectrogram`

The normal `~TimeSeries.spectrogram` method uses non-overlapping intervals
to calculate discrete PSDs for each stride. This is fine for long-duration
data, but give poor resolution when studying short-duration phenomena.

The `~TimeSeries.spectrogram2` method allows for highly-overlapping FFT
calculations to over-sample the frequency content of the input `TimeSeries`
to produce a much more feature-rich output.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# As with the other `~gwpy.spectrogram.Spectrogram` examples, we import the
# `TimeSeries` class, and :meth:`~TimeSeries.get` the data, but in this
# example we only need 5 seconds of datam,
from gwpy.timeseries import TimeSeries
gwdata = TimeSeries.get(
    'L1:OAF-CAL_DARM_DQ', 'Feb 28 2015 06:02:05', 'Feb 28 2015 06:02:10')

# Now we can call the `~TimeSeries.spectrogram2` method of `gwdata` to
# calculate our over-dense `~gwpy.spectrogram.Spectrogram`
specgram = gwdata.spectrogram2(fftlength=0.15, overlap=0.14) ** (1/2.)

# To whiten the `specgram` we can use the :meth:`~Spectrogram.ratio` method
# to divide by the overall median:
medratio = specgram.ratio('median')

# Finally, we make a plot:
plot = medratio.plot(norm='log', vmin=0.5, vmax=10)
plot.set_yscale('log')
plot.set_ylim(40, 8192)
plot.add_colorbar(label='Amplitude relative to median')
plot.set_title('L1 $h(t)$ with noise interference')
plot.show()
