#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

"""Calculating the SNR associated with a given astrophysical signal model

The example :ref:`gwpy-example-signal-gw150914` showed us we can visually
extract a signal from the noise using basic signal-processing techniques.

However, an actual astrophysical search algorithm detects signals by
calculating the signal-to-noise ratio (SNR) of data for each in a large bank
of signal models, known as templates.

Using |pycbc|_ (the actual search code), we can do that.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, as always, we fetch some of the public data from the LIGO Open
# Science Center:

from gwpy.timeseries import TimeSeries
data = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)

# and condition it by applying a highpass filter at 15 Hz
high = data.highpass(15)

# This is important to remove noise at lower frequencies that isn't
# accurately calibrated, and swamps smaller noises at higher frequencies.

# For this example, we want to calculate the SNR over a 4 second segment, so
# we calculate a Power Spectral Density with a 4 second FFT length (using all
# of the data), then :meth:`~TimeSeries.crop` the data:

psd = high.psd(4, 2)
zoom = high.crop(1126259460, 1126259464)

# In order to calculate signal-to-noise ratio, we need a signal model
# against which to compare our data.
# For this we import :func:`pycbc.waveform.get_fd_waveform
# <pycbc.waveform.waveform.get_fd_waveform>` and generate a template as a
# `pycbc.types.FrequencySeries <pycbc.types.frequencyseries.FrequencySeries>`:

from pycbc.waveform import get_fd_waveform
hp, _ = get_fd_waveform(approximant="IMRPhenomD", mass1=40, mass2=32,
                        f_lower=20, f_final=2048, delta_f=psd.df.value)

# At this point we are ready to calculate the SNR, so we import the
# :func:`pycbc.filter.matched_filter
# <pycbc.filter.matchedfilter.matched_filter>` method, and pass it
# our template, the data, and the PSD:


from pycbc.filter import matched_filter
snr = matched_filter(hp, zoom.to_pycbc(), psd=psd.to_pycbc(),
                     low_frequency_cutoff=15)
snrts = TimeSeries.from_pycbc(snr).abs()

# .. note::
#
#    Here we have used the :meth:`~TimeSeries.to_pycbc` methods of the
#    `~gwpy.timeseries.TimeSeries` and `~gwpy.frequencyseries.FrequencySeries`
#    objects to convert from GWpy objects to something that PyCBC functions
#    can understand, and then used the :meth:`~TimeSeries.from_pycbc` method
#    to convert back to a GWpy object.

# We can plot the SNR `TimeSeries` around the region of interest:
plot = snrts.plot()
ax = plot.gca()
ax.set_xlim(1126259461, 1126259463)
ax.set_epoch(1126259462.427)
ax.set_ylabel('Signal-to-noise ratio (SNR)')
ax.set_title('LIGO-Hanford signal-correlation for GW150914')
plot.show()

# We can clearly see a large spike (above 17!) at the time of the GW150914
# signal!
# This is, in principle, how the full, blind, CBC search is performed, using
# all of the available data, and a bank of tens of thousand of signal models.
