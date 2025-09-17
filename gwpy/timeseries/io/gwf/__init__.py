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

"""Input/output routines for gravitational-wave frame (GWF) format files.

The API for GWF I/O integration is defined by the core module, which
specifies that each backend module provides two functions:

`read`
    A function that takes in a source reference (file path) and a list of
    channels to read, and returns a `TimeSeriesDict` of data.

`write`
    A function that takes in a `TimeSeriesDict` of data and a target reference
    (file path) and writes the data to that file.

See `gwpy.timeseries.io.gwf.core.read_timeseriesdict` and
`gwpy.timeseries.io.gwf.core.write_timeseriesdict` for full function
signatures and docstrings.

A backend module can then register itself to act as a backend for the
`format='gwf'` series reader by calling the
`gwpy.timeseries.io.gwf.core.register_gwf_backend` function.
"""

# basic routines
from . import core

# backends
for backend_mod_name in (
    "frameCPP",
    "framel",
    "lalframe",
):
    core.register_gwf_backend(backend_mod_name)

BACKENDS = core.BACKENDS
