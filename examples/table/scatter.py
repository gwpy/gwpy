#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Plotting an `EventTable` in a scatter

We can use GWpy's `EventTable` to download the catalogue of gravitational-wave
detections, and create a scatter plot to investigate the mass distribution
of events.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.table'

# First, we can download the ``'GWTC-1-confident'`` catalogue using
# :meth:`EventTable.fetch_open_data`:

from gwpy.table import EventTable
events = EventTable.fetch_open_data(
    "GWTC-1-confident",
    columns=(
        "mass_1_source",
        "mass_2_source",
        "chirp_mass_source",
        "luminosity_distance"
    ),
)

# We can now make a scatter plot by specifying the x- and y-axis columns,
# and (optionally) the colour:

plot = events.scatter(
    "mass_1_source", "mass_2_source",
    color="chirp_mass_source"
)
plot.colorbar(label="Chirp_mass [{}]".format(r"M$_{\odot}$"))
plot.show()

# We can similarly plot how the total event mass is distributed with
# distance.  First we have to build the total mass (``'mtotal'``) column
# from the component masses:

events.add_column(
    events["mass_1_source"] + events["mass_2_source"],
    name="mtotal"
)

# and now can make a new scatter plot:

plot = events.scatter("luminosity_distance", "mtotal")
plot.show()
