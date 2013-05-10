# Copyright (C) 2012 Duncan Macleod
#
# This program is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation; either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

"""Module to calculate CBC parameters
"""

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = 0.0
__date__ = None


def reduced_mass(mass1, mass2):
    """@returns the reduced mass of the given binary
    """
    return mass1 * mass2  / float(mass1 + mass2)


def chirp_mass(mass1, mass2):
    """@returns the chirp mass of the given binary
    """
    total_mass = float(mass1 + mass2)
    reduced_mass = mass1 * mass2 / total_mass
    return reduced_mass**(3/5.) * total_mass**(2/5.)
