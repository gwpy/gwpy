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

.. warning::

   This example requires LIGO.ORG credentials for data access.

Most data recorded from a gravitational-wave interferometer carry information
across a wide band of frequencies, typically up to a few kiloHertz, but
often it is the case that the low-frequency amplitude dwarfs that of the
high-frequency content, making discerning high-frequency features difficult.

We employ a technique called 'whitening' to normalize the power at all
frequencies so that excess power at any frequency is more obvious.

We demonstrate below with an auxiliary signal recording transmitted power
in one of the interferometer arms, which recorded two large glitches with
a frequency of around 5-50Hz.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.timeseries'

# First, we import the `TimeSeries` and :meth:`~TimeSeries.get` the data:
from gwpy.timeseries import TimeSeries
data = TimeSeries.get('H1:ASC-Y_TR_A_NSUM_OUT_DQ', 1123084671, 1123084703)

# Now, we can `~TimeSeries.whiten` the data to enhance the higher-frequency
# content
white = data.whiten(4, 2)

# and can `~TimeSeries.plot` both the original and whitened data
from gwpy.plot import Plot
plot = Plot(data, white, separate=True, sharex=True)
plot.axes[0].set_ylabel('Y-arm power [counts]', fontsize=16)
plot.axes[1].set_ylabel('Whitened amplitude', fontsize=16)
plot.show()

# Here we see two large spikes that are completely undetected in the raw
# `TimeSeries`, but are very obvious in the whitened data. We can also see
# tapering effects at the boundaries as the whitening filter settles in,
# meaning that the first and last ~second of data are corrupted and should
# be discarded before further processing.
