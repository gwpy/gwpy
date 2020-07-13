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

"""Plotting an `EventTable` in a histogram

I would like to study the distribution of the GW events detected to date.
"""

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__currentmodule__ = 'gwpy.table'

# First, we can download the ``'GWTC-1-confident'`` catalogue using
# :meth:`EventTable.fetch_open_data`:

from gwpy.table import EventTable
events = EventTable.fetch_open_data(
    "GWTC-1-confident",
    columns=("mass_1_source", "mass_2_source"),
)
events.add_column(
    events["mass_1_source"] + events["mass_2_source"],
    name="mtotal"
)

# and can generate a new `~gwpy.plot.Plot` using the
# :meth:`~EventTable.hist` method:

plot = events.hist('mtotal', bins=10, range=(0, 100), histtype='stepfilled')
ax = plot.gca()
ax.set_xlabel(r"Total mass [M$_{\odot}$]")
ax.set_ylabel("Number of events")
ax.set_title("GWTC-1-confident")
plot.show()
