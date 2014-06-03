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

"""Utilities for `Spectrum calculation.
"""

from astropy import units

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


def safe_import(module, method):
    """Import the given module, with a more useful `ImportError` message.
    """
    try:
        return __import__(module, fromlist=[''])
    except ImportError:
        raise ImportError("The '%s' Spectrum method requires the %s module."
                          % (method, module))


def scale_timeseries_units(tsunit, scaling='density'):
    """Scale the unit of a `TimeSeries` to match that of a `Spectrum`.

    Parameters
    ----------
    tsunit : `~astropy.units.UnitBase`
        input unit from `TimeSeries`
    scaling : `str`
        type of spectrum, either 'density' for a PSD, or 'spectrum' for a
        power spectrum.

    Returns
    -------
    spectrumunit : `~astropy.units.Unit`
        unit to be applied to the resulting `Spectrum`.
    """
    # set units
    if scaling == 'density':
        baseunit = units.Hertz
    elif scaling == 'spectrum':
        baseunit = units.dimensionless_unscaled
    else:
        raise ValueError("Unknown scaling: %r" % scaling)
    if tsunit:
        specunit = tsunit ** 2 / baseunit
    else:
        specunit = 1 / baseunit
    return specunit
