#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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
    columns=("mass1", "mass2", "E_rad", "distance"),
)

# We can now make a scatter plot by specifying the x- and y-axis columns,
# and (optionally) the colour:

plot = events.scatter("mass1", "mass2", color="E_rad")
plot.colorbar(label="E_rad [{}]".format(r"M$_{\odot}$ c$^{2}$"))
plot.show()

# We can similarly plot how the total event mass is distributed with
# distance.  First we have to build the total mass (``'mtotal'``) column
# from the component masses:

events.add_column(events["mass1"] + events["mass2"], name="mtotal")

# and now can make a new scatter plot:

plot = events.scatter("distance", "mtotal")
plot.show()
