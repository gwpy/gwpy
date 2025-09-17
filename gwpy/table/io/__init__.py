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

"""Input/output methods for tabular data."""

# apply useful decorators to the I/O formats provided by Astropy
# _before_ we add any of our own
from .. import EventTable
from . import reg
reg.wrap_unified_io_readers(EventTable)

# other readers are defined in their own modules, and are responsible
# for applying the decorators themselves.
from . import (  # noqa: E402
    cwb,  # cWB ROOT and ASCII formats
    gravityspy,  # Gravity Spy Triggers
    gstlal,  # GstLAL LIGO_LW XML format
    gwf,  # GWF FrEvents (e.g. MBTA)
    gwosc,  # GWOSC
    hacr,  # Hierarchichal Algorithm for Curves and Ridges
    ligolw,  # LIGO_LW XML format
    omega,  # Omega ASCII format
    omicron,  # Omicron ROOT format
    pycbc,  # PyCBC (Live) HDF5
    root,  # generic ROOT stuff
    snax,  # SNAX HDF5 features
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
