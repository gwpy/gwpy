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

Calculating the time-dependent coherence between two channels
#############################################################

The standard coherence calculation outputs a frequency series
(`~gwpy.frequencyseries.FrequencySeries`) giving a time-averaged measure
of coherence. See :ref:`sphx_glr_examples_frequencyseries_coherence.py` for an
example.

The `TimeSeries` method :meth:`~TimeSeries.coherence_spectrogram` performs the
same coherence calculation every ``stride``, giving a time-varying coherence
measure.

These data are available as part of the |GWOSC_O3_AUX_RELEASE|_.
"""

# %%
# First, we import the `TimeSeries`
from gwpy.timeseries import TimeSeries

# %%
# and then :meth:`~TimeSeries.get` the LIGO-Hanford strain data,
# and the mains power monitor data, for a 20 minute (1200 second
# window near the end of the |GWOSC_O3_AUX_RELEASE|_:

start = 1389410000
end = 1389411200
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
# `TimeSeries` with respect to the other, using an 2-second Fourier
# transform length, with a 1-second (50%) overlap, and a stride of 10:
coh = strain.coherence_spectrogram(mains, 20, fftlength=2, overlap=1)

# %%
# Finally, we can :meth:`~gwpy.spectrogram.Spectrogram.plot` the
# resulting data
plot = coh.plot()
ax = plot.gca()
ax.set_ylabel("Frequency [Hz]")
ax.set_yscale("log")
ax.set_ylim(10, 2000)
ax.set_title("Coherence between Power mains and LIGO-Hanford strain data")
ax.grid(visible=True, which="both", axis="both")
ax.colorbar(label="Coherence", clim=[0, 1], cmap="plasma")
plot.show()

# %%
# This shows the time-dependent coherence between the two channels.
# The 60 Hz power mains frequency and its harmonics are clearly visible,
# but the coherence is variable over time, demonstrating the varying
# fidelity of the incoming mains signal.
