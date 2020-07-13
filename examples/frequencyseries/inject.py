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

"""Inject a known signal into a `FrequencySeries`

It can often be useful to add some known signal to inherently random
or noisy data. For example, one might want to investigate what
would happen if a binary black hole merger signal occured at or near
the time of a glitch. In LIGO data analysis, this procedure is referred
to as an _injection_.

In the example below we will create a stream of random, white Gaussian
noise, then inject a loud, steady sinuosoid. We will do this in the
frequency domain because it is much easier to model a sinusoid there.
"""

__author__ = "Alex Urban <alexander.urban@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we prepare one second of Gaussian noise:

from numpy import random
from gwpy.timeseries import TimeSeries
noise = TimeSeries(random.normal(scale=.1, size=1024), sample_rate=1024)

# To inject a signal in the frequency domain, we need to take an FFT:

noisefd = noise.fft()

# We can now easily inject a loud sinusoid of unit amplitude at, say,
# 30 Hz. To do this, we use :meth:`~gwpy.types.series.Series.inject`.

import numpy
from gwpy.frequencyseries import FrequencySeries
signal = FrequencySeries(numpy.array([1.]), f0=30, df=noisefd.df)
injfd = noisefd.inject(signal)

# We can then visualize the data before and after injection in the frequency
# domain:

from gwpy.plot import Plot
plot = Plot(numpy.abs(noisefd), numpy.abs(injfd), separate=True,
            sharex=True, sharey=True, xscale='log', yscale='log')
plot.show()

# Finally, for completeness we can visualize the effect before and after
# injection back in the time domain:

inj = injfd.ifft()
plot = Plot(noise, inj, separate=True, sharex=True, sharey=True,
            figsize=(12, 6))
plot.show()

# We can see why sinusoids are easier to inject in the frequency domain:
# they only require adding at a single frequency.
