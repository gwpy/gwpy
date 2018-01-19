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

"""Utilies for interacting with the LIGO Algorithm Library.
"""

from __future__ import absolute_import

from collections import OrderedDict

from six import string_types

import numpy

from astropy import units

import lal

from ..time import to_gps
from ..detector import units as gwpy_units  # pylint: disable=unused-import

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# LAL type enum
try:
    LAL_TYPE_STR = {lal.LAL_I2_TYPE_CODE: 'INT2',
                    lal.LAL_I4_TYPE_CODE: 'INT4',
                    lal.LAL_I8_TYPE_CODE: 'INT8',
                    lal.LAL_U2_TYPE_CODE: 'UINT2',
                    lal.LAL_U4_TYPE_CODE: 'UINT4',
                    lal.LAL_U8_TYPE_CODE: 'UINT8',
                    lal.LAL_S_TYPE_CODE: 'REAL4',
                    lal.LAL_D_TYPE_CODE: 'REAL8',
                    lal.LAL_C_TYPE_CODE: 'COMPLEX8',
                    lal.LAL_Z_TYPE_CODE: 'COMPLEX16'}
except AttributeError:
    LAL_TYPE_STR = {lal.I2_TYPE_CODE: 'INT2',
                    lal.I4_TYPE_CODE: 'INT4',
                    lal.I8_TYPE_CODE: 'INT8',
                    lal.U2_TYPE_CODE: 'UINT2',
                    lal.U4_TYPE_CODE: 'UINT4',
                    lal.U8_TYPE_CODE: 'UINT8',
                    lal.S_TYPE_CODE: 'REAL4',
                    lal.D_TYPE_CODE: 'REAL8',
                    lal.C_TYPE_CODE: 'COMPLEX8',
                    lal.Z_TYPE_CODE: 'COMPLEX16'}

LAL_TYPE_FROM_STR = dict((v, k) for k, v in LAL_TYPE_STR.items())

# map numpy dtypes to LAL type codes
try:
    LAL_TYPE_FROM_NUMPY = {numpy.int16: lal.LAL_I2_TYPE_CODE,
                           numpy.int32: lal.LAL_I4_TYPE_CODE,
                           numpy.int64: lal.LAL_I8_TYPE_CODE,
                           numpy.uint16: lal.LAL_U2_TYPE_CODE,
                           numpy.uint32: lal.LAL_U4_TYPE_CODE,
                           numpy.uint64: lal.LAL_U8_TYPE_CODE,
                           numpy.float32: lal.LAL_S_TYPE_CODE,
                           numpy.float64: lal.LAL_D_TYPE_CODE,
                           numpy.complex64: lal.LAL_C_TYPE_CODE,
                           numpy.complex128: lal.LAL_Z_TYPE_CODE}
except AttributeError:
    LAL_TYPE_FROM_NUMPY = {numpy.int16: lal.I2_TYPE_CODE,
                           numpy.int32: lal.I4_TYPE_CODE,
                           numpy.int64: lal.I8_TYPE_CODE,
                           numpy.uint16: lal.U2_TYPE_CODE,
                           numpy.uint32: lal.U4_TYPE_CODE,
                           numpy.uint64: lal.U8_TYPE_CODE,
                           numpy.float32: lal.S_TYPE_CODE,
                           numpy.float64: lal.D_TYPE_CODE,
                           numpy.complex64: lal.C_TYPE_CODE,
                           numpy.complex128: lal.Z_TYPE_CODE}

LAL_TYPE_STR_FROM_NUMPY = dict((key, LAL_TYPE_STR[value]) for (key, value) in
                               LAL_TYPE_FROM_NUMPY.items())

try:
    LAL_UNIT_INDEX = [lal.lalMeterUnit,
                      lal.lalKiloGramUnit,
                      lal.lalSecondUnit,
                      lal.lalAmpereUnit,
                      lal.lalKelvinUnit,
                      lal.lalStrainUnit,
                      lal.lalADCCountUnit]
except AttributeError:
    LAL_UNIT_INDEX = [lal.MeterUnit,
                      lal.KiloGramUnit,
                      lal.SecondUnit,
                      lal.AmpereUnit,
                      lal.KelvinUnit,
                      lal.StrainUnit,
                      lal.ADCCountUnit]
    lal_unit_to_str = str
    LAL_UNIT_FROM_ASTROPY = dict((units.Unit(lal_unit_to_str(u)), u) for
                                 u in LAL_UNIT_INDEX)
else:
    lal_unit_to_str = lal.UnitToString
    LAL_UNIT_FROM_ASTROPY = dict((units.Unit(lal_unit_to_str(u)), u) for
                                 u in LAL_UNIT_INDEX)


def to_lal_unit(aunit):
    """Convert the input unit into a `LALUnit`

    For example::

       >>> u = to_lal_unit('m**2 / kg ** 4')
       >>> print(u)
       m^2 kg^-4

    Parameters
    ----------
    aunit : `~astropy.units.Unit`, `str`
        the input unit

    Returns
    -------
    unit : `LALUnit`
        the LALUnit representation of the input

    Raises
    ------
    ValueError
        if LAL doesn't understand the base units for the input
    """
    if isinstance(aunit, string_types):
        aunit = units.Unit(aunit)
    aunit = aunit.decompose()
    lunit = lal.Unit()
    for base, power in zip(aunit.bases, aunit.powers):
        # try this base
        try:
            lalbase = LAL_UNIT_FROM_ASTROPY[base]
        except KeyError:
            lalbase = None
            # otherwise loop through the equivalent bases
            for eqbase in base.find_equivalent_units():
                try:
                    lalbase = LAL_UNIT_FROM_ASTROPY[eqbase]
                except KeyError:
                    continue
        # if we didn't find anything, raise an exception
        if lalbase is None:
            raise ValueError("LAL has no unit corresponding to %r" % base)
        lunit *= lalbase ** power
    return lunit


def from_lal_unit(lunit):
    """Convert a LALUnit` into a `~astropy.units.Unit`

    Parameters
    ----------
    lunit : `lal.Unit`
        the input unit

    Returns
    -------
    unit : `~astropy.units.Unit`
        the Astropy representation of the input

    Raises
    ------
    TypeError
        if ``lunit`` cannot be converted to `lal.Unit`
    ValueError
        if Astropy doesn't understand the base units for the input
    """
    try:
        lunit = lal.Unit(lunit)
    except RuntimeError:
        raise TypeError("Cannot convert %r to lal.Unit" % lunit)
    aunit = units.Unit("")
    for power, lalbase in zip(lunit.unitNumerator, LAL_UNIT_INDEX):
        # if not used, continue
        if not power:
            continue
        # convert to astropy unit
        try:
            u = units.Unit(lal_unit_to_str(lalbase))
        except ValueError:
            raise ValueError("Astropy has no unit corresponding to %r"
                             % lalbase)
        aunit *= u ** power
    return aunit


def to_lal_ligotimegps(gps):
    """Convert the given GPS time to a `lal.LIGOTimeGPS` object

    Parameters
    ----------
    gps : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        input GPS time, can be anything parsable by :meth:`~gwpy.time.to_gps`

    Returns
    -------
    ligotimegps : `lal.LIGOTimeGPS`
        a SWIG-LAL `~lal.LIGOTimeGPS` representation of the given GPS time
    """
    gps = to_gps(gps)
    return lal.LIGOTimeGPS(gps.gpsSeconds, gps.gpsNanoSeconds)


# -- detectors ----------------------------------------------------------------

LAL_DETECTORS = OrderedDict((ifo.frDetector.prefix, ifo.frDetector) for ifo in
                            lal.CachedDetectors)
