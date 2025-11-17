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
# along with GWpy.  If not, see <http://www.gnu.org/licenses/>.

"""
.. sectionauthor:: Duncan Macleod <duncan.macleod@ligo.org>
.. currentmodule:: gwpy.timeseries

Whitening a `TimeSeries`
########################

Most data recorded from a gravitational-wave interferometer carry information
across a wide band of frequencies, typically up to a few kiloHertz, but
often it is the case that the low-frequency amplitude dwarfs that of the
high-frequency content, making discerning high-frequency features difficult.
This is especially true of the LIGO differential arm strain measurement,
which encodes any gravitational wave signals that are present.

We employ a technique called 'whitening' to normalize the power at all
frequencies so that excess power at any frequency is more obvious.

We demonstrate below the LIGO-Livingston gravitational-wave strain
measurement signal around |GW200129|_, the loudest signal as yet
detected by LIGO.
"""

# %%
# Data access
# -----------
# First, we use the :doc:`gwosc <gwosc:index>` Python client to get the GPS
# time of the event:

from gwosc.datasets import event_gps
gps = event_gps("GW200129_065458")

# %%
# Then we can import the `TimeSeries` object and fetch the strain data using
# :meth:`TimeSeries.get` in a window around that event:

from gwpy.timeseries import TimeSeries
data = TimeSeries.get("L1", int(gps) - 64, int(gps) + 64)

# %%
# To demonstrate the relative power across the frequency band, we can
# quickly estimate an Amplitude Spectral Density (ASD) for these data:

asd = data.asd(fftlength=8)
plot = asd.plot(
    xlim=(8, 1000),
    ylabel="Strain ASD [$1/\\sqrt{Hz}$]",
)
plot.show()

# %%
# Whitening
# ---------
# The ASD clearly shows the dominance in amplitude of the lowest frequency
# components of the data, where the seismic noise around the observatory
# is most impactful.
# We can now :meth:`~TimeSeries.whiten` the data to to normalise the
# amplitudes across the frequency range:

white = data.whiten(fftlength=8)

# %%
# and can `~TimeSeries.plot` both the original and whitened data around the
# event time:

from gwpy.plot import Plot
plot = Plot(
    data,
    white,
    separate=True,
    sharex=True,
    epoch=gps,
    xlim=(gps - 1, gps + 1),
)
plot.axes[0].set_ylabel("Strain amplitude", fontsize=16)
plot.axes[1].set_ylabel("Whitened strain amplitude", fontsize=16)
plot.show()

# %%
# The top figure is dominated by the low-frequency noise, whereas the
# whitened data below highlights a few spikes in the data at higher
# frequencies.
#
# We can zoom in very close around the event time:

plot = white.crop(gps - .1, gps + .1).plot(
    ylabel="Whitened strain amplitude",
)
plot.axes[0].set_epoch(gps)
plot.show()

# %%
# Here, centred around time 0.03 is the clear signature of a binary black hole
# merger, |GW200129|_.
# This signal is completely hidden in the unfiltered data, but the simple act
# of whitening has exposed the loudest gravitational-wave event ever detected!
