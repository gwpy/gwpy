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

from six import string_types

import numpy

from astropy import units

import lal

from .. import version
from .. import timeseries

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

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

LAL_TYPE_FROM_STR = dict((v, k) for k, v in LAL_TYPE_STR.iteritems())

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
                               LAL_TYPE_FROM_NUMPY.iteritems())

try:
    LAL_UNIT_INDEX = [lal.lalMeterUnit,
                      lal.lalKiloGramUnit,
                      lal.lalSecondUnit,
                      lal.lalAmpereUnit,
                      lal.lalKelvinUnit,
                      lal.lalStrainUnit,
                      lal.lalADCCountUnit]
    LAL_UNIT_FROM_ASTROPY = dict((units.Unit(lal.UnitToString(u)), u) for
                                 u in LAL_UNIT_INDEX)
except AttributeError:
    LAL_UNIT_INDEX = [lal.MeterUnit,
                      lal.KiloGramUnit,
                      lal.SecondUnit,
                      lal.AmpereUnit,
                      lal.KelvinUnit,
                      lal.StrainUnit,
                      lal.ADCCountUnit]
    LAL_UNIT_FROM_ASTROPY = dict((units.Unit(str(u)), u) for
                                 u in LAL_UNIT_INDEX)


def to_lal_unit(aunit):
    """Convert the input unit into a :lalsuite:`LALUnit`

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
    unit : :lalsuite:`LALUnit`
        the LALUnit representation of the input

    Raises
    ------
    ValueError
        if LAL doesn't understand the base units for the input
    """
    if isinstance(aunit, string_types):
        aunit = units.Unit(aunit)
    lunit = lal.Unit()
    for base, power in zip(aunit.bases, aunit.powers):
        lalbase = None
        for eqbase in base.find_equivalent_units():
            try:
                lalbase = LAL_UNIT_FROM_ASTROPY[eqbase]
            except KeyError:
                continue
            else:
                lunit *= lalbase ** power
                break
        if lalbase is None:
            raise ValueError("LAL has no unit corresponding to %r" % base)
    return lunit


def from_lal_unit(lunit):
    """Convert a :lalsuite`LALUnit` into a `~astropy.units.Unit`

    Parameters
    ----------
    aunit : :lalsuite:`LALUnit`
        the input unit

    Returns
    -------
    unit : `~astropy.units.Unit`
        the Astropy representation of the input

    Raises
    ------
    ValueError
        if Astropy doesn't understand the base units for the input
    """
    aunit = units.Unit('')
    for power, lalbase in zip(lunit.unitNumerator, LAL_UNIT_INDEX):
        astrobase = None
        for key, val in LAL_UNIT_FROM_ASTROPY:
            if val == lalbase:
                astrobase = key
        if astrobase is None:
            raise ValueError("Astropy has no unit corresponding to %r"
                             % lalbase)
        aunit *= astrobase ** power
    return aunit
