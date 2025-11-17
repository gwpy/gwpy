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
.. currentmodule:: gwpy.frequencyseries

Calculate and plot a `FrequencySeries`
######################################

One of the principal means of estimating the sensitivity of a
gravitational-wave detector is to esimate it's amplitude spectral density
(ASD).
The ASD is a measurement of how a signal's amplitude varies across
different frequencies.

The ASD can be estimated directly from a `~gwpy.timeseries.TimeSeries`
using the :meth:`~gwpy.timeseries.TimeSeries.asd` method.
"""

# %%
# Data access
# -----------
# For this example we choose to estimate the ASD around |GW200115|_,
# one of the first observations of a neutron star-black hole binary.
# We can use the :doc:`gwosc <gwosc:index>` Python package to query
# for the relevant GPS time:

from gwosc.datasets import event_gps

gps = event_gps("GW200115")

# %%
# In order to generate a `FrequencySeries` we need to import the
# `~gwpy.timeseries.TimeSeries` and use
# :meth:`~gwpy.timeseries.TimeSeries.fetch_open_data` to download the strain
# records:

from gwpy.timeseries import TimeSeries

lho = TimeSeries.get("H1", gps - 16, gps + 16)
llo = TimeSeries.get("L1", gps - 16, gps + 16)

# %%
# Calculate the ASDs
# ------------------
# We can then call the :meth:`~gwpy.timeseries.TimeSeries.asd` method to
# calculated the amplitude spectral density for each
# `~gwpy.timeseries.TimeSeries`:

lhoasd = lho.asd(4, 2)
lloasd = llo.asd(4, 2)

# %%
# Visualisation
# -------------
# We can then :meth:`~FrequencySeries.plot` the spectra using the 'standard'
# colour scheme:

plot = lhoasd.plot(label="LIGO-Hanford", color="gwpy:ligo-hanford")
ax = plot.gca()
ax.plot(lloasd, label="LIGO-Livingston", color="gwpy:ligo-livingston")
ax.set_xlim(16, 1600)
ax.set_ylim(1e-24, 1e-21)
ax.set_ylabel(r"Strain ASD [1/$\sqrt{\mathrm{Hz}}]$")
ax.legend(frameon=False, bbox_to_anchor=(1., 1.), loc="lower right", ncol=2)
plot.show()
