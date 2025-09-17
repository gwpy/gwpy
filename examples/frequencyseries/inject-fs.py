# Copyright (c) 2018-2020 Louisiana State University
#               2020-2025 Cardiff University
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

"""
.. currentmodule:: gwpy.timeseries
.. sectionauthor:: Alex Urban <alexander.urban@ligo.org>

Inject a signal into a `FrequencySeries`
########################################

It can often be useful to add some known signal to inherently random
or noisy data. For example, one might want to investigate what
would happen if a binary black hole merger signal occured at or near
the time of a glitch. In LIGO data analysis, this procedure is referred
to as an *injection*.

In the example below we will create a stream of random, white Gaussian
noise, then inject a loud, steady sinuosoid. We will do this in the
frequency domain because it is much easier to model a sinusoid there.
"""

# %%
# Generate random data
# --------------------
# First, we prepare one second of Gaussian noise using
# `numpy.random.Generator.normal`:

from numpy.random import default_rng

from gwpy.timeseries import TimeSeries

rng = default_rng()
noise = TimeSeries(rng.normal(scale=.1, size=1024), sample_rate=1024)

# %%
# To inject a signal in the frequency domain, we need to take an FFT:

noisefd = noise.fft()

# %%
# Inject noise
# ------------
# We can now easily inject a loud sinusoid of unit amplitude at, say,
# 30 Hz. To do this, we use
# :meth:`~gwpy.frequencyseries.FrequencySeries.inject`:

import numpy

from gwpy.frequencyseries import FrequencySeries

signal = FrequencySeries(numpy.array([1.]), f0=30, df=noisefd.df)
injfd = noisefd.inject(signal)

# %%
# Visualisation
# -------------
# We can then visualize the data before and after injection in the frequency
# domain:

from gwpy.plot import Plot

plot = Plot(
    numpy.abs(noisefd),
    numpy.abs(injfd),
    xscale="log",
    yscale="log",
)
plot.show()

# %%
# Finally, for completeness we can visualize the effect before and after
# injection back in the time domain:

inj = injfd.ifft()
plot = Plot(
    noise,
    inj,
    separate=True,
    sharex=True,
    sharey=True,
    figsize=(12, 6),
)
plot.show()
