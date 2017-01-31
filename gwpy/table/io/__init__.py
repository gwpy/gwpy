# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Input/output methods for tabular data.
"""

from . import (  # pylint: disable=unused-import
    ligolw,  # glue.ligolw XML format
    root,  # generic ROOT stuff
    omicron,  # Omicron ROOT format
    omega,  # Omega ASCII format
    cwb,  # cWB ROOT and ASCII formats
    pycbc,  # PyCBC (Live) HDF5
    hacr,  # Hierarchichal Algorithm for Curves and Ridges
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
