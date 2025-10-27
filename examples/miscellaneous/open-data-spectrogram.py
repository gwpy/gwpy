# Copyright (c) 2019-2025 Cardiff University
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
.. sectionauthor:: Duncan Macleod <duncan.macleod@ligo.org>
.. currentmodule:: gwpy.timeseries

Plotting a spectrogram of all open data for many hours
######################################################

In order to study interferometer performance, it is common in LIGO to plot
all of the data for a day, in order to determine trends, and see data-quality
issues.

This is done for the LIGO-Virgo detector network, with
`up-to-date plots <https://gwosc.org/detector_status/>`__
available from |GWOSC|.

This example demonstrates how to download data segments from GWOSC, then
use those to build a multi-hour spectrogram plot of LIGO-Livingston strain
data.
"""

# %%
# .. currentmodule:: gwpy.segments
#
# Getting the segments
# --------------------
#
# First, we need to fetch the Open Data timeline segments from GWOSC.
# To do that we can call the :meth:`DataQualityFlag.fetch_open_data` method
# using ``'H1_DATA'`` as the flag (for an explanation of what this means,
# read up on `The S6 Data Release <https://gwosc.org/S6/>`__).

from gwpy.segments import DataQualityFlag
l1segs = DataQualityFlag.fetch_open_data(
    "L1_DATA",
    "Aug 17 2017 08:00",
    "Aug 17 2017 16:00",
)

# %%
# For sanity, lets plot these segments:

splot = l1segs.plot(
    figsize=[12, 3],
    epoch="August 17 2017",
)
splot.show()
splot.close()  # hide

# %%
# We see that the LIGO Hanford Observatory detector was operating for the
# majority of the day, with a few outages of ~30 minutes or so.

# %%
# We can use the :func:`abs` function to display the total amount of time
# spent taking data:

print(abs(l1segs.active))

# %%
# .. currentmodule:: gwpy.timeseries
#
# Working with strain data
# ------------------------
#
# Now, we can loop through the active segments of ``'L1_DATA'`` and fetch the
# strain `TimeSeries` for each segment, calculating a
# :class:`~gwpy.spectrogram.Spectrogram` for each segment.

from gwpy.timeseries import TimeSeries
spectrograms = []
for start, end in l1segs.active:
    l1strain = TimeSeries.get(
        "L1",
        start,
        end,
    )
    specgram = l1strain.spectrogram(30, fftlength=4) ** (1/2.)
    spectrograms.append(specgram)

# %%
# Finally, we can build a :meth:`~gwpy.spectrogram.Spectrogram.plot`:

# Create an empty plot with a single set of Axes
from gwpy.plot import Plot
plot = Plot(figsize=(12, 6))
ax = plot.gca()
# add each spectrogram to the Axes
for specgram in spectrograms:
    ax.imshow(specgram)
# finalise the plot metadata
ax.set_xscale("auto-gps", epoch="Aug 17 2017")
ax.set_ylim(20, 2000)
ax.set_yscale("log")
ax.set_ylabel("Frequency [Hz]")
ax.set_title("LIGO-Livingston strain data")
ax.colorbar(
    cmap="viridis",
    norm="log",
    clim=(5e-24, 1e-21),
    label=r"Strain noise [1/$\sqrt{\mathrm{Hz}}$]",
)
# add the segments as a 'state' indicator along the bottom
plot.add_segments_bar(l1segs)
plot.show()
