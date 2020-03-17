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

"""Plotting a `Spectrogram`

One of the most useful methods of visualising gravitational-wave data is to
use a spectrogram, highlighting the frequency-domain content of some data
over a number of time steps.

For this example we can use the public data around the GW150914 detection.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeries` and call
# :meth:`TimeSeries.fetch_open_data` the download the strain
# data for the LIGO-Hanford interferometer
from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data(
    'H1', 'Sep 14 2015 09:45', 'Sep 14 2015 09:55')

# Next, we can calculate a `~gwpy.spectrogram.Spectrogram` using the
# :meth:`spectrogram` method of the `TimeSeries` over a 2-second stride
# with a 1-second FFT and # .5-second overlap (50%):
specgram = data.spectrogram(2, fftlength=1, overlap=.5) ** (1/2.)

# .. note::
#    :meth:`TimeSeries.spectrogram` returns a Power Spectral Density (PSD)
#    `~gwpy.spectrogram.Spectrogram` by default, so we use the ``** (1/2.)``
#    to convert this into a (more familiar) Amplitude Spectral Density.

# Finally, we can make a plot using the
# :meth:`~gwpy.spectrogram.Spectrogram.plot` method
plot = specgram.imshow(norm='log', vmin=5e-24, vmax=1e-19)
ax = plot.gca()
ax.set_yscale('log')
ax.set_ylim(10, 2000)
ax.colorbar(
    label=r'Gravitational-wave amplitude [strain/$\sqrt{\mathrm{Hz}}$]')
plot.show()

# This shows the relative stability of the interferometer sensitivity over
# the ten-minute span. Despite there being a gravitational-wave signal in the
# data, the resolution (and dynamic range) of the spectrogram make it
# impossible to see. The :ref:`next example <gwpy-example-spectrogram-ratio>`
# shows you how to normalise a `~gwpy.spectrogram.Spectrogram` to better
# see features in the most sensitive frequency band.
