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

Plotting segments for a `StateVector`
#####################################

Confident detection of gravitational-wave signals is critically dependent
on understanding the quality of the data searched.
Alongside the strain *h(t)* data, |GWOSC|_ also
releases a *Data Quality* :ref:`state vector <gwpy-statevector>`.
We can use this to check on the quality of the data from the LIGO Livingston
detector around |GW170817|_.
"""

# %%
# First, we can import the `StateVector` class:
from gwpy.timeseries import StateVector

# %%
# and download the state information surrounding GW170817:
data = StateVector.get(
    "L1",
    1187008882 - 100,
    1187008882 + 100,
)

# %%
# Finally, we make a :meth:`~StateVector.plot`, passing `insetlabels=True` to
# display the bit names inside the axes:
print(type(data))
plot = data.plot(insetlabels=True)
ax = plot.gca()
ax.set_xscale("seconds", epoch=1187008882)
ax.axvline(1187008882, color="orange", linestyle="--")
ax.set_title("LIGO-Livingston data quality around GW170817")
plot.show()

# %%
# This plot shows that for a short time exactly overlapping with GW170817
# there was a data quality issue recorded that would negatively impact a
# search for generic gravitational-wave transients (bursts).
# For more details on this *glitch*, and on how it was excised, please see
# the science summary for |GW170817|_.
