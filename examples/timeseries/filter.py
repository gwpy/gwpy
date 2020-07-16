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

"""Filtering a `TimeSeries` with a ZPK filter

Several data streams read from the LIGO detectors are whitened before being
recorded to prevent numerical errors when using single-precision data
storage.
In this example we read such `channel <gwpy.detector.Channel>` and undo the
whitening to show the physical content of these data.

"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeries` and :meth:`~TimeSeries.get` the data:
from gwpy.timeseries import TimeSeries
white = TimeSeries.get(
    'L1:OAF-CAL_DARM_DQ', 'March 2 2015 12:00', 'March 2 2015 12:30')

# Now, we can re-calibrate these data into displacement units by first applying
# a `highpass <TimeSeries.highpass>` filter to remove the low-frequency noise,
# and then applying our de-whitening filter in `ZPK <TimeSeries.zpk>` format
# with five zeros at 100 Hz and five poles at 1 Hz (giving an overall DC
# gain of 10 :sup:`-10`:
hp = white.highpass(4)
displacement = hp.zpk([100]*5, [1]*5, 1e-10)

# We can visualise the impact of the whitening by calculating the ASD
# `~gwpy.frequencyseries.FrequencySeries` before and after the filter,

whiteasd = white.asd(8, 4)
dispasd = displacement.asd(8, 4)

# and plotting:

from gwpy.plot import Plot
plot = Plot(whiteasd, dispasd, separate=True, sharex=True,
            xscale='log', yscale='log')

# Here we have passed the two
# `spectra <gwpy.frequencyseries.FrequencySeries>` in order,
# then `separate=True` to display them on separate Axes, `sharex=True` to tie
# the `~matplotlib.axis.XAxis` of each of the `~gwpy.plot.Axes`
# together.
#
# Finally, we prettify our plot with some limits, and some labels:
plot.text(0.95, 0.05, 'Preliminary', fontsize=40, color='gray',  # hide
          ha='right', rotation=45, va='bottom', alpha=0.5)  # hide
plot.axes[0].set_ylabel('ASD [whitened]')
plot.axes[1].set_ylabel(r'ASD [m/$\sqrt{\mathrm{Hz}}$]')
plot.axes[1].set_xlabel('Frequency [Hz]')
plot.axes[1].set_ylim(1e-20, 1e-15)
plot.axes[1].set_xlim(5, 4000)
plot.show()
