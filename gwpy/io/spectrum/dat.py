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

"""Read a Spectrum from a dat file

These files must be in two-colum (frequency, amplitude) format
"""

from six import string_types

import numpy
from astropy.io import registry

from ...spectrum.core import Spectrum
from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

def read_dat(filepath, fcol=0, ampcol=1, **kwargs):
    """Read a `Spectrum` from a txt file
    """
    frequency, amplitude = numpy.loadtxt(filepath, usecols=[fcol, ampcol],
                                         unpack=True)
    return Spectrum(amplitude, frequencies=frequency, **kwargs)


def identify_dat(*args, **kwargs):
    """Identify the given file as a dat file, rather than anything else

    Returns
    -------
    True
        if the filename endswith .txt or .dat
    False
        otherwise
    """
    filename = args[1][0]
    if not isinstance(filename, string_types):
        filename = filename.name
    if filename.endswith('txt') or filename.endswith('dat'):
        return True
    return False


# register this file-reader with the Spectrum class
registry.register_reader('dat', Spectrum, read_dat, force=True)
registry.register_identifier('dat', Spectrum, identify_dat)
