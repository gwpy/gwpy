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

"""Whitening a `TimeSeries`

Most data recorded from a gravitational-wave interferometer carry information
across a wide band of frequencies, typically up to a few kiloHertz, but
often it is the case that the low-frequency amplitude dwarfs that of the
high-frequency content, making discerning high-frequency features difficult.

We employ a technique called 'whitening' to normalize the power at all
frequencies so that excess power at any frequency is more obvious.

Consider GW150914, the first detected gravitational wave event that 
occurred at 09:50:45 UTC on September 14, 2015. For an explanation of
some of these steps, see [1, 2].
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'


# First, we import the `TimeSeries` and :meth:`~TimeSeries.get` the data:
from gwpy.timeseries import TimeSeries
gps = 1126259462
t0 = gps - 8
t1 = gps + 8
data = TimeSeries.fetch_open_data('H1', t0, t1)

# Now, we can `~TimeSeries.whiten` the data to enhance the higher-frequency
# content.
# In this instance, this refers to dividing each Fourier coefficient for
# a given frequency by the respective estimate of noise ASD, 
# effectively downweighting noisy frequencies [2].
white = data.whiten(2, 1, window="tukey")

# Next, we apply a bandpass filter between 35 Hz and 350 Hz,
# and compare to a TimeSeries without whitening.
nowhite = data.bandpass(35, 350)
white = white.bandpass(35, 350)

"""
.. plot::
"""
# Finally, we use `~TimeSeries.plot`:
from gwpy.plot import Plot
plot = Plot(data, white, nowhite, separate=True, sharex=True)
plot.axes[0].set_ylabel('$d(t)$', fontsize=16)
plot.axes[1].set_ylabel('$d_w(t)$', fontsize=16)
plot.axes[2].set_ylabel('$d_b(t)$', fontsize=16)
plot.show()

# We can see that, in the times series without pre-whitening, 
# the gravitational wave signal at approximately 8 seconds is 
# not easily visible. 
# We can also see
# tapering effects at the boundaries as the whitening filter settles in,
# meaning that the first and last ~second of data are corrupted and should
# be discarded before further processing.

# 1. Abbott, et al. "Observation of gravitational waves from a binary black hole merger." Physical review letters 116.6 (2016): 061102. 
# 2. Abbott, et al. "A guide to LIGO–Virgo detector noise and extraction of transient gravitational-wave signals." Classical and Quantum Gravity 37.5 (2020): 055002.
