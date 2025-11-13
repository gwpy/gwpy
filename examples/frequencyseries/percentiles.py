# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>

"""
.. sectionauthor:: Duncan Macleod <duncan.macleod@ligo.org>
.. currentmodule:: gwpy.timeseries

Plotting an averaged ASD with percentiles.
##########################################

As we have seen in :doc:`hoff`, the Amplitude Spectral Density (ASD) is a key
indicator of frequency-domain sensitivity for a gravitational-wave detector.

However, the ASD doesn't allow you to see how much the sensitivity varies
over time.
One tool for that is the :doc:`spectrogram </spectrogram/index>`, while another
is simply to show percentiles of a number of ASD measurements.

In this example we calculate the median ASD over 2048-seconds surrounding
the GW178017 event, and also the 5th and 95th percentiles of the ASD, and
plot them on a single figure.
"""

# %%
# Data access
# -----------
# First, as always, we get the data using :meth:`TimeSeries.get`:

from gwpy.timeseries import TimeSeries
hoft = TimeSeries.get("H1", 1187007040, 1187009088)

# %%
# Calculate spectrogram
# ---------------------
# Next we calculate a :class:`~gwpy.spectrogram.Spectrogram` by calculating
# a number of ASDs, using the :meth:`~gwpy.timeseries.TimeSeries.spectrogram2`
# method:

sg = hoft.spectrogram2(fftlength=4, overlap=2, window="hann") ** (1 / 2.)

# %%
# From this we can trivially extract the median, 5th and 95th percentiles:

median = sg.percentile(50)
low = sg.percentile(5)
high = sg.percentile(95)

# %%
# Visualisation
# -------------
# Finally, we can make plot, using :meth:`~gwpy.plot.Axes.plot_mmm` to
# display the 5th and 95th percentiles as a shaded region around the median:

from gwpy.plot import Plot
plot = Plot()
ax = plot.add_subplot(
    xscale="log",
    xlim=(10, 1500),
    xlabel="Frequency [Hz]",
    yscale="log",
    ylim=(3e-24, 2e-20),
    ylabel=r"Strain noise [1/$\sqrt{\mathrm{Hz}}$]",
)
ax.plot_mmm(median, low, high, color="gwpy:ligo-hanford")
ax.set_title("LIGO-Hanford strain noise variation around GW170817")
plot.show()

# %%
# Now we can see that the ASD varies by factors of a few across most of the
# frequency band, with notable exceptions, e.g. around the 60-Hz power line
# harmonics (60 Hz, 120 Hz, 180 Hz, ...) where the noise is very stable.
