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

# To demonstrate this, we can download some data associated with the
# gravitational-wave event GW510914:

from gwpy.timeseries import TimeSeries
lho = TimeSeries.fetch_open_data('H1', 1126259458, 1126259467, verbose=True)

# and can :meth:`~TimeSeries.highpass` and :meth:`~TimeSeries.whiten`
# the remove low-frequency noise and try and enhance low-amplitude signals
# across the middle of the frequency band:

hp = lho.highpass(20)
white = hp.whiten(4, 2).crop(1126259460, 1126259465)

# .. note::
#
#    We chose to :meth:`~TimeSeries.crop` out the leading and trailing 2
#    seconds of the the whitened data series here to remove any filtering
#    artefacts that may have been introduced.

# Now we can call the `~TimeSeries.spectrogram2` method of `gwdata` to
# calculate our over-dense `~gwpy.spectrogram.Spectrogram`, using a
# 1/16-second FFT length and high overlap:

specgram = white.spectrogram2(fftlength=1/16., overlap=15/256.) ** (1/2.)
specgram = specgram.crop_frequencies(20)  # drop everything below highpass

# Finally, we make a plot:

plot = specgram.plot(norm='log', cmap='viridis', yscale='log')
ax = plot.gca()
ax.set_title('LIGO-Hanford strain data around GW150914')
ax.set_xlim(1126259462, 1126259463)
ax.colorbar(label=r'Strain ASD [1/$\sqrt{\mathrm{Hz}}$]')
plot.show()

# Here we can see the trace of a high-mass binary black hole system,
# referred to as GW150914.
# For more details on this signal, please see
# `Abbott et al. (2016) <https://doi.org/10.1103/PhysRevLett.116.061102>`_
# (the joint LSC-Virgo publication announcing this detection).
