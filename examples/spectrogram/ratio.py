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

Plotting a normalised `~gwpy.spectrogram.Spectrogram`
#####################################################

The :ref:`sphx_glr_examples_spectrogram_plot.py` example showed how to
generate and display a `~gwpy.spectrogram.Spectrogram` of the LIGO-Hanford
strain data around the |GW150914|_ event.

However, because of the shape of the LIGO sensitivity curve, picking out
features in the most sensitive frequency band (a few hundred Hertz) is
very hard.

We can normalise our `~gwpy.spectrogram.Spectrogram` to highligh those
features.
"""

# %%
# Again, we import the `TimeSeries` and call
# :meth:`TimeSeries.get` the download the strain
# data for the LIGO-Hanford interferometer
from gwpy.timeseries import TimeSeries
data = TimeSeries.get(
    "H1",
    "Sep 14 2015 09:45",
    "Sep 14 2015 09:55",
)

# %%
# Next, we can calculate a `~gwpy.spectrogram.Spectrogram` using the
# :meth:`spectrogram` method of the `TimeSeries` over a 2-second stride
# with a 1-second FFT and # .5-second overlap (50%):
specgram = data.spectrogram(2, fftlength=1, overlap=.5) ** (1/2.)

# %%
# and can normalise it against the overall median ASD by calling the
# :meth:`~gwpy.spectrogram.Spectrogram.ratio` method:

normalised = specgram.ratio("median")

# %%
# Finally, we can make a plot using the
# :meth:`~gwpy.spectrogram.Spectrogram.plot` method
plot = normalised.plot(norm="log", vmin=.1, vmax=10, cmap="Spectral_r")
ax = plot.gca()
ax.set_yscale("log")
ax.set_ylim(10, 2000)
ax.colorbar(label="Relative amplitude")
plot.show()

# %%
# Even with a normalised spectrogram, the resolution is such that a signal
# as short as that of GW150914 is impossible to see.
# See :ref:`sphx_glr_examples_spectrogram_spectrogram2.py` for an example that
# uses a high-resolution spectrogram method to zoom in around the exact time of
# the signal.
