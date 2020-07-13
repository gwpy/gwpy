#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2018-2020)
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

"""Inject a known signal into a `TimeSeries`

It can often be useful to add some known signal to an inherently random
or noisy timeseries. For example, one might want to examine what
would happen if a binary black hole merger signal occured at or near
the time of a glitch. In LIGO data analysis, this procedure is referred
to as an _injection_.

In the example below, we will create a stream of random, white Gaussian
noise, then inject a simulation of GW150914 into it at a known time.
"""

__author__ = "Alex Urban <alexander.urban@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we prepare one second of Gaussian noise:

from numpy import random
from gwpy.timeseries import TimeSeries
noise = TimeSeries(random.normal(scale=.1, size=16384), sample_rate=16384)

# Then we can download a simulation of the GW150914 signal from GWOSC:

from astropy.utils.data import get_readable_fileobj
url = ("https://www.gw-openscience.org/s/events/GW150914/P150914/"
       "fig2-unfiltered-waveform-H.txt")
with get_readable_fileobj(url) as f:
    signal = TimeSeries.read(f, format='txt')
signal.t0 = .5  # make sure this intersects with noise time samples

# Note, since this simulation cuts off before a certain time, it is
# important to taper its ends to zero to avoid ringing artifacts.
# We can accomplish this using the
# :meth:`~gwpy.timeseries.TimeSeries.taper` method.

signal = signal.taper()

# Since the time samples overlap, we can inject this into our noise data
# using :meth:`~gwpy.types.series.Series.inject`:

data = noise.inject(signal)

# Finally, we can visualize the full process in the time domain:

from gwpy.plot import Plot
plot = Plot(noise, signal, data, separate=True, sharex=True, sharey=True)
plot.gca().set_epoch(0)
plot.show()

# We can clearly see that the loud GW150914-like signal has been layered
# on top of Gaussian noise with the correct amplitude and phase evolution.
