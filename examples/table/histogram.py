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

Plotting an `EventTable` in a histogram
#######################################

I would like to study the distribution of the GW events detected to date.
"""

# %%
# First, we can download the ``'GWTC-1-confident'`` catalogue using
# :meth:`EventTable.fetch_open_data`:

from gwpy.table import EventTable
events = EventTable.fetch_open_data(
    "GWTC",
    columns=("mass_1_source", "mass_2_source"),
)
events.add_column(
    events["mass_1_source"] + events["mass_2_source"],
    name="mtotal",
)

# %%
# and can generate a new `~gwpy.plot.Plot` using the
# :meth:`~EventTable.hist` method:

plot = events.hist("mtotal", bins=20, range=(0, 100), histtype="stepfilled")
ax = plot.gca()
ax.set_xlabel(r"Total mass [M$_{\odot}$]")
ax.set_ylabel("Number of events")
ax.set_title("GWTC events")
plot.show()
