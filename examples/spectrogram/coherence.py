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

"""Calculating the time-dependent coherence between two channels

The standard coherence calculation outputs a frequency series
(`~gwpy.frequencyseries.FrequencySeries`) giving a time-averaged measure
of coherence. See :ref:`gwpy-example-frequencyseries-coherence` for an
example.

The `TimeSeries` method :meth:`~TimeSeries.coherence_spectrogram` performs the
same coherence calculation every ``stride``, giving a time-varying coherence
measure.

"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeriesDict`
from gwpy.timeseries import TimeSeriesDict

# and then :meth:`~TimeSeriesDict.get` the data for the strain output
# (``H1:GDS-CALIB_STRAIN``) and the PSL periscope accelerometer
# (``H1:PEM-CS_ACC_PSL_PERISCOPE_X_DQ``):
data = TimeSeriesDict.get(
    ['H1:GDS-CALIB_STRAIN', 'H1:PEM-CS_ACC_PSL_PERISCOPE_X_DQ'],
    1126260017,
    1126260617,
)
hoft = data['H1:GDS-CALIB_STRAIN']
acc = data['H1:PEM-CS_ACC_PSL_PERISCOPE_X_DQ']

# We can then calculate the :meth:`~TimeSeries.coherence` of one
# `TimeSeries` with respect to the other, using an 2-second Fourier
# transform length, with a 1-second (50%) overlap:
coh = hoft.coherence_spectrogram(acc, 10, fftlength=.5, overlap=.25)

# Finally, we can :meth:`~gwpy.spectrogram.Spectrogram.plot` the
# resulting data
plot = coh.plot()
ax = plot.gca()
ax.set_ylabel('Frequency [Hz]')
ax.set_yscale('log')
ax.set_ylim(10, 8000)
ax.set_title(
    'Coherence between PSL periscope motion and LIGO-Hanford strain data',
)
ax.grid(True, 'both', 'both')
ax.colorbar(label='Coherence', clim=[0, 1], cmap='plasma')
plot.show()
