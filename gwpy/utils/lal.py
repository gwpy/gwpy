# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

import operator
import re
from collections import OrderedDict
from fractions import Fraction
from functools import reduce

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
LAL_NUMPY_FROM_TYPE_STR = {v: k for k, v in LAL_TYPE_STR_FROM_NUMPY.items()}

LAL_TYPE_REGEX = re.compile(r'(U?INT|REAL|COMPLEX)\d+')


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


def from_lal_type(laltype):
    """Convert the data type of a LAL instance or type into a numpy data type.

    Parameters
    ----------
    laltype : `SwigPyObject` or `type`
        the input LAL instance or type

    Returns
    -------
    dtype : `type`
        the numpy data type, such as `numpy.uint32`, `numpy.float64`, etc.

    Raises
    ------
    ValueError
        if the numpy data type cannot be inferred from the LAL object
    """
    if not isinstance(laltype, type):
        laltype = type(laltype)
    name = laltype.__name__
    match = LAL_TYPE_REGEX.match(name)
    if not match or match[0] not in LAL_NUMPY_FROM_TYPE_STR:
        raise ValueError(f'{name!r} has no known numpy type equivalent')
    return LAL_NUMPY_FROM_TYPE_STR[match[0]]


# -- units --------------------------------------------------------------------

LAL_UNIT_INDEX = [  # the order corresponds to how LAL stores compound units
    units.meter,
    units.kilogram,
    units.second,
    units.ampere,
    units.Kelvin,
    units.Unit('strain'),
    units.count,
]


def to_lal_unit(aunit):
    """Convert the input unit into a `lal.Unit` and a scaling factor.

    Parameters
    ----------
    aunit : `~astropy.units.Unit`, `str`
        the input unit

    Returns
    -------
    unit : `lal.Unit`
        the LAL representation of the base unit.

    scale : `float`
        the linear scaling factor that should be applied to any associated
        data, see _Notes_ below.

    Notes
    -----
    Astropy supports 'scaled' units of the form ``<N> <u>``
    ere ``<N>`` is a `float` and ``<u>`` the base `astropy.units.Unit`,
    e.g. ``'123 m'``, e.g:

    >>> from astropy.units import Quantity
    >>> x = Quantity(4, '123m')
    >>> print(x)
    4.0 123 m
    >>> print(x.decompose())
    492.0 m

    LAL doesn't support scaled units in this way, so this function simply
    returns the scaling factor of the unit so that it may be applied
    manually to any associated data.

    Examples
    --------
    >>> print(to_lal_unit('m**2 / kg ** 4'))
    (m^2 kg^-4, 1.0)
    >>> print(to_lal_unit('123 m'))
    (m, 123.0)

    Raises
    ------
    ValueError
        if LAL doesn't understand the base units for the input
    """
    # format incoming unit
    if isinstance(aunit, str):
        aunit = units.Unit(aunit)
    aunit = aunit.decompose()

    # handle scaled units
    pow10 = numpy.log10(aunit.scale)
    if pow10 and pow10.is_integer():
        lunit = lal.Unit(f"10^{int(pow10)}")
        scale = 1
    else:
        lunit = lal.Unit()
        scale = aunit.scale

    # decompose unit into LAL base units
    for base, power in zip(aunit.bases, aunit.powers):
        try:  # try this base
            i = LAL_UNIT_INDEX.index(base)
        except ValueError as exc:
            exc.args = (
                f"LAL has no unit corresponding to '{base}'",
            )
            raise
        frac = Fraction(power)
        lunit.unitNumerator[i] = frac.numerator
        lunit.unitDenominatorMinusOne[i] = frac.denominator - 1

    return lunit, scale


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
    return reduce(
        operator.mul,
        (
            LAL_UNIT_INDEX[i] ** Fraction(int(num), int(den + 1))
            for i, (num, den) in enumerate(zip(
                lunit.unitNumerator,
                lunit.unitDenominatorMinusOne,
            ))
        ),
    ) * 10**lunit.powerOfTen


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
