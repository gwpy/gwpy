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

Accessing and visualising public GW detector data
#################################################

Data from the current generation gravitational wave detectors are published
by |GWOSCl| and freely available to the public.
In this example we demonstrate how to identify times of a published
GW detection event, and to download and visualise detector data.
"""

# %%
# Firstly, we can use the :doc:`gwosc <gwosc:index>` Python package to query
# for the time of the first gravitational-wave detection |GW150914|_:

from gwosc.datasets import event_gps
gps = event_gps("GW150914")

# %%
# GWpy's `TimeSeries` class provides an interface to the public |GWOSC|_
# data in the :meth:`~TimeSeries.get` method; to use it we
# need to first import the `TimeSeries` object:

from gwpy.timeseries import TimeSeries

# %%
# then call the :meth:`~TimeSeries.get` method, passing it the
# prefix for the interferometer we want (`'L1'` here for LIGO-Livingston),
# and the GPS start and stop times of our query (based around the GPS time
# for GW150914):

data = TimeSeries.get("L1", gps-5, gps+5)

# %%
# and then we can make a plot:

plot = data.plot(
    title="LIGO Livingston Observatory data for GW150914",
    ylabel="Strain amplitude",
    color="gwpy:ligo-livingston",
    epoch=gps,
)
plot.show()

# %%
# We can't see anything that looks like a gravitational wave signal in these
# data, the amplitude is dominated by low-frequency detector noise.
# Further filtering is required to be able to identify the GW150914 event
# here, see :ref:`sphx_glr_examples_signal_gw150914.py` for a more in-depth example of
# extracting signals from noise.
