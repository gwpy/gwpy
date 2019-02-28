# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

This module requires lal >= 6.14.0
"""

from __future__ import absolute_import

import operator
from collections import OrderedDict

from six import string_types
from six.moves import reduce

import numpy

from astropy import units

import lal

from ..time import to_gps
# import gwpy.detector.units to register other units now
from ..detector import units as gwpy_units  # noqa: F401

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# -- type matching ------------------------------------------------------------

# LAL type enum
LAL_TYPE_STR = {
    lal.I2_TYPE_CODE: 'INT2',
    lal.I4_TYPE_CODE: 'INT4',
    lal.I8_TYPE_CODE: 'INT8',
    lal.U2_TYPE_CODE: 'UINT2',
    lal.U4_TYPE_CODE: 'UINT4',
    lal.U8_TYPE_CODE: 'UINT8',
    lal.S_TYPE_CODE: 'REAL4',
    lal.D_TYPE_CODE: 'REAL8',
    lal.C_TYPE_CODE: 'COMPLEX8',
    lal.Z_TYPE_CODE: 'COMPLEX16',
}

LAL_TYPE_FROM_STR = {v: k for k, v in LAL_TYPE_STR.items()}

LAL_TYPE_FROM_NUMPY = {
    numpy.int16: lal.I2_TYPE_CODE,
    numpy.int32: lal.I4_TYPE_CODE,
    numpy.int64: lal.I8_TYPE_CODE,
    numpy.uint16: lal.U2_TYPE_CODE,
    numpy.uint32: lal.U4_TYPE_CODE,
    numpy.uint64: lal.U8_TYPE_CODE,
    numpy.float32: lal.S_TYPE_CODE,
    numpy.float64: lal.D_TYPE_CODE,
    numpy.complex64: lal.C_TYPE_CODE,
    numpy.complex128: lal.Z_TYPE_CODE,
}

LAL_TYPE_STR_FROM_NUMPY = {k: LAL_TYPE_STR[v] for
                           (k, v) in LAL_TYPE_FROM_NUMPY.items()}


def to_lal_type_str(pytype):
    """Convert the input python type to a LAL type string

    Examples
    --------
    To convert a python type:

    >>> from gwpy.utils.lal import to_lal_type_str
    >>> to_lal_type_str(float)
    'REAL8'

    To convert a `numpy.dtype`:

    >>> import numpy
    >>> to_lal_type_str(numpy.dtype('uint32'))
    'UINT4'

    To convert a LAL type code:

    >>> to_lal_type_str(11)
    'REAL8'

    Raises
    ------
    KeyError
        if the input doesn't map to a LAL type string
    """
    # noop
    if pytype in LAL_TYPE_FROM_STR:
        return pytype

    # convert type code
    if pytype in LAL_TYPE_STR:
        return LAL_TYPE_STR[pytype]

    # convert python type
    try:
        dtype = numpy.dtype(pytype)
        return LAL_TYPE_STR_FROM_NUMPY[dtype.type]
    except (TypeError, KeyError):
        raise ValueError("Failed to map {!r} to LAL type string")


def find_typed_function(pytype, prefix, suffix, module=lal):
    """Returns the lal method for the correct type

    Parameters
    ----------
    pytype : `type`, `numpy.dtype`
        the python type, or dtype, to map

    prefix : `str`
        the function name prefix (before the type tag)

    suffix : `str`
        the function name suffix (after the type tag)

    Raises
    ------
    AttributeError
        if the function is not found

    Examples
    --------
    >>> from gwpy.utils.lal import find_typed_function
    >>> find_typed_function(float, 'Create', 'Sequence')
    <built-in function CreateREAL8Sequence>
    """
    laltype = to_lal_type_str(pytype)
    return getattr(module, '{0}{1}{2}'.format(prefix, laltype, suffix))


# -- units --------------------------------------------------------------------

LAL_UNIT_INDEX = [
    lal.MeterUnit,
    lal.KiloGramUnit,
    lal.SecondUnit,
    lal.AmpereUnit,
    lal.KelvinUnit,
    lal.StrainUnit,
    lal.ADCCountUnit,
]
LAL_UNIT_FROM_ASTROPY = {units.Unit(str(u)): u for u in LAL_UNIT_INDEX}


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
    return reduce(operator.mul, (
        units.Unit(str(LAL_UNIT_INDEX[i])) ** exp for
        i, exp in enumerate(lunit.unitNumerator)))


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
