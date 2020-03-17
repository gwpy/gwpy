#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Plotting a spectrogram of all open data for 1 day

In order to study interferometer performance, it is common in LIGO to plot
all of the data for a day, in order to determine trends, and see data-quality
issues.

This is done for the LIGO-Virgo detector network, with
`up-to-date plots <https://www.gw-openscience.org/detector_status/>`__
available from |GWOSC|.

This example demonstrates how to download data segments from GWOSC, then
use those to build a day-timescale spectrogram plot of LIGO-Hanford strain
data.
"""

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# .. currentmodule:: gwpy.segments
#
# Getting the segments
# --------------------
#
# First, we need to fetch the Open Data timeline segments from GWOSC.
# To do that we can call the :meth:`DataQualityFlag.fetch_open_data` method
# using ``'H1_DATA'`` as the flag (for an explanation of what this means,
# read up on `The S6 Data Release <https://www.gw-openscience.org/S6/>`__).

from gwpy.segments import DataQualityFlag
h1segs = DataQualityFlag.fetch_open_data('H1_DATA',
                                         'Sep 16 2010', 'Sep 17 2010')

# For sanity, lets plot these segments:

splot = h1segs.plot(figsize=[12, 3])
splot.show()
splot.close()  # hide

# We see that the LIGO Hanford Observatory detector was operating for the
# majority of the day, with a few outages of ~30 minutes or so.

# We can use the :func:`abs` function to display the total amount of time
# spent taking data:

print(abs(h1segs.active))

# .. currentmodule:: gwpy.timeseries
#
# Working with strain data
# ------------------------
#
# Now, we can loop through the active segments of ``'H1_DATA'`` and fetch the
# strain `TimeSeries` for each segment, calculating a
# :class:`~gwpy.spectrogram.Spectrogram` for each segment.

from gwpy.timeseries import TimeSeries
spectrograms = []
for start, end in h1segs.active:
    h1strain = TimeSeries.fetch_open_data('H1', start, end, verbose=True)
    specgram = h1strain.spectrogram(30, fftlength=4) ** (1/2.)
    spectrograms.append(specgram)

# Finally, we can build a :meth:`~gwpy.spectrogram.Spectrogram.plot`:

from gwpy.plot import Plot
plot = Plot(figsize=(12, 6))
ax = plot.gca()
for specgram in spectrograms:
    ax.imshow(specgram)
ax.set_xscale('auto-gps', epoch='Sep 16 2010')
ax.set_xlim('Sep 16 2010', 'Sep 17 2010')
ax.set_ylim(40, 2000)
ax.set_yscale('log')
ax.set_ylabel('Frequency [Hz]')
ax.set_title('LIGO-Hanford strain data')
ax.colorbar(cmap='viridis', norm='log', clim=(1e-23, 1e-19),
            label=r'Strain noise [1/$\sqrt{\mathrm{Hz}}$]')
plot.add_segments_bar(h1segs)
plot.show()
