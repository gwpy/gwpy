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

"""Calculating the time-dependent coherence between two channels

The standard coherence calculation outputs a frequency series
(`~gwpy.frequencyseries.FrequencySeries`) giving a time-averaged measure
of coherence.

The `TimeSeries` method :meth:`~TimeSeries.coherence_spectrogram` performs the
same coherence calculation every ``stride``, giving a time-varying coherence
measure.

"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeriesDict`
from gwpy.timeseries import TimeSeriesDict

# and then :meth:`~TimeSeriesDict.get` both data sets:
data = TimeSeriesDict.get(['L1:LSC-SRCL_IN1_DQ', 'L1:LSC-CARM_IN1_DQ'],
                           'Feb 13 2015', 'Feb 13 2015 00:15')

# We can then use the :meth:`~TimeSeries.coherence_spectrogram` method
# of one `TimeSeries` to calcululate the time-varying coherence with
# respect to the other, using a 0.5-second FFT length, with a
# 0.45-second (90%) overlap, with a 8-second stride:
coh = data['L1:LSC-SRCL_IN1_DQ'].coherence_spectrogram(
    data['L1:LSC-CARM_IN1_DQ'], 8, 0.5, 0.45)

# Finally, we can :meth:`~gwpy.spectrogram.Spectrogram.plot` the
# resulting data
plot = coh.plot()
ax = plot.gca()
ax.set_ylabel('Frequency [Hz]')
ax.set_yscale('log')
ax.set_ylim(10, 8000)
ax.set_title('Coherence between SRCL and CARM for L1')
ax.grid(True, 'both', 'both')
plot.add_colorbar(label='Coherence', clim=[0, 1])
plot.show()
