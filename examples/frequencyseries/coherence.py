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
.. currentmodule:: gwpy.timeseries
.. sectionauthor:: Duncan Macleod <duncan.macleod@ligo.org>

Calculate the coherence between two channels
############################################

The `coherence <http://en.wikipedia.org/wiki/Coherence_(physics)>`_ between
two channels is a measure of the frequency-domain correlation between their
time-series data.

In LIGO, the coherence is a crucial indicator of how noise sources couple into
the main differential arm-length readout.
Here we use use the :meth:`TimeSeries.coherence` method to highlight coupling
of the mains power supply into the strain output of the LIGO-Hanford
interferometer.

These data are available as part of the |GWOSC_O3_AUX_RELEASE|_.
"""

# %%
# Data access
# -----------
# First, we import the `TimeSeries`

from gwpy.timeseries import TimeSeries

# %%
# and then :meth:`~TimeSeries.get` the LIGO-Hanford strain data,
# and the mains power monitor data, for a 600-second window near the
# end of the |GWOSC_O3_AUX_RELEASE|_:

start = 1389410000
end = 1389410600
strain = TimeSeries.get("H1", start, end)
mains = TimeSeries.get(
    "H1:PEM-EY_MAINSMON_EBAY_1_DQ",
    start,
    end,
    frametype="H1_AUX_AR1",
)

# %%
#
# .. admonition:: Data are accessed separately
#     :class: note
#
#     We could have used `TimeSeriesDict.get` to access both channels in a
#     single call - that method would first try to find a single dataset that
#     contain both of them, then automatically fall back to separate calls if
#     that fails.
#
#     But, since we know that these channels are in different datsets, we
#     access them separately here for clarity and speed.
#
# Calculating coherence
# ---------------------
# We can then calculate the :meth:`~TimeSeries.coherence` of one
# `TimeSeries` with respect to the other, using an 8-second Fourier
# transform length, with a 4-second (50%) overlap:

coh = strain.coherence(mains, fftlength=8, overlap=4)

# %%
#
# The output of this method is a :class:`~gwpy.frequencyseries.FrequencySeries`
# containing the coherence values for each frequency bin.
#
# Visualisation
# -------------
# We can now :meth:`~gwpy.frequencyseries.FrequencySeries.plot` the coherence:

plot = coh.plot(
    xlabel="Frequency [Hz]",
    xscale="log",
    ylabel="Coherence",
    yscale="linear",
    ylim=(0, 1),
)
plot.show()

# %%
# Here we can see strong coherence at 60 Hz and its harmonics, indicating
# that the mains power supply is coupling into the differential arm length
# control loop.
