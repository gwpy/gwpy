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

"""Calculating the coherence between two channels

The `coherence <http://en.wikipedia.org/wiki/Coherence_(physics)>`_ between
two channels is a measure of the frequency-domain correlation between their
time-series data.

In LIGO, the coherence is a crucial indicator of how noise sources couple into
the main differential arm-length readout.
Here we use use the :meth:`TimeSeries.coherence` method to highlight coupling
of motion of a beam periscope attached to the main laser table into the
strain output of the LIGO-Hanford interferometer.
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
coh = hoft.coherence(acc, fftlength=2, overlap=1)

# Finally, we can :meth:`~gwpy.frequencyseries.FrequencySeries.plot` the
# resulting data:
plot = coh.plot(
    xlabel='Frequency [Hz]', xscale='log',
    ylabel='Coherence', yscale='linear', ylim=(0, 1),
)
plot.show()

# We can clearly see the correlation between the periscope motion and the
# strain data between 100 Hz and 1000 Hz. Once this was discovered the
# motion was damped by modifying the structure of the periscope itself,
# reducing the coupling into the gravitational-wave strain output.
