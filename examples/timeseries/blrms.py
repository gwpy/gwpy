# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2025)
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

Comparing seismic trends between LIGO sites
###########################################

On Jan 16 2020 there was a series of earthquakes, that
should have had an impact on LIGO operations, I'd like to find out.
"""

# %%
# Data access
# -----------
# We choose to look at the 0.03Hz-1Hz ground motion band-limited RMS channel
# (1-second average trends) for each interferometer.
# We use a
# `format string <https://docs.python.org/3/tutorial/inputoutput.html#the-string-format-method>`_
# so we can substitute the interferometer prefix without duplicating the channel name:

channel = "{ifo}:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M"
lhochan = channel.format(ifo="H1")
llochan = channel.format(ifo="L1")

# %%
# To access the data, we can use :meth:`~TimeSeriesDict.get`, and give start and end
# datetimes to fetch 6 hours of data for each interferometer:

from gwpy.timeseries import TimeSeriesDict
data = TimeSeriesDict.get(
    [lhochan, llochan],
    "Jan 16 2020 8:00",
    "Jan 16 2020 14:00",
    host="nds.gwosc.org",
)

# %%
# Visualisation
# -------------
# Now that we have the data, we can easily `~TimeSeriesDict.plot` them:

plot = data[lhochan].plot(
    color="gwpy:ligo-hanford",
    label="LIGO-Hanford",
    yscale="log",
    ylabel=r"$1-3$\,Hz motion [nm/s]",
)
ax = plot.gca()
ax.plot(data[llochan], color="gwpy:ligo-livingston", label="LIGO-Livingston")
ax.set_title("Impact of earthquakes on LIGO")
ax.legend()
plot.show()

# %%
# As we can see, the earthquake had a huge impact on the LIGO observatories,
# severly impairing operations for several hours.
