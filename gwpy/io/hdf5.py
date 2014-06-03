# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Write GWpy objects to HDF5 files
"""

import h5py

from gwpy import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def open_hdf5(filename):
    """Wrapper to open a :class:`h5py.File` from disk, gracefully
    handling a few corner cases
    """
    if isinstance(filename, (h5py.Group, h5py.Dataset)):
        return filename
    elif isinstance(filename, file):
        return h5py.File(filename.name, 'r')
    else:
        return h5py.File(filename, 'r')


def identify_hdf5(*args, **kwargs):
    """Identify an input file as LOSC HDF based on its filename
    """
    filename = args[3]
    if isinstance(filename, file):
        filename = filename.name
    if (isinstance(filename, (unicode, str)) and
            filename.endswith(('hdf', 'hdf5'))):
        return True
    elif isinstance(filename, (h5py.Group, h5py.Dataset)):
        return True
    else:
        return False
