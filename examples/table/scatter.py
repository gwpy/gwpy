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
.. currentmodule:: gwpy.table

Plotting an `EventTable` in a scatter
#####################################

We can use GWpy's `EventTable` to download the catalogue of gravitational-wave
detections, and create a scatter plot to investigate the mass distribution
of events.
"""

# %%
# First, we can download the ``'GWTC-1-confident'`` catalogue using
# :meth:`EventTable.fetch_open_data`:

from gwpy.table import EventTable
events = EventTable.fetch_open_data(
    "GWTC",
    columns=(
        "mass_1_source",
        "mass_2_source",
        "luminosity_distance",
        "network_matched_filter_snr",
    ),
)

# %%
# We can now make a scatter plot by specifying the x- and y-axis columns,
# and (optionally) the colour:

plot = events.scatter(
    "mass_1_source", "mass_2_source",
    color="network_matched_filter_snr",
)
plot.colorbar(label="Signal-to-noise ratio (SNR)")
plot.show()

# %%
# We can similarly plot how the total event mass is distributed with
# distance.  First we have to build the total mass (``'mtotal'``) column
# from the component masses:

events.add_column(
    events["mass_1_source"] + events["mass_2_source"],
    name="mtotal",
)

# %%
# and now can make a new scatter plot:

plot = events.scatter("luminosity_distance", "mtotal")
plot.show()
