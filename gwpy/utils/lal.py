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

"""Utilities for interacting with the LIGO Algorithm Library.

This module requires lal >= 6.14.0.
"""

from __future__ import annotations

import operator
import re
from fractions import Fraction
from functools import reduce
from typing import (
    TYPE_CHECKING,
    overload,
)

import lal
import numpy
from astropy import units

# Import gwpy.detector.units to register other units now
from ..detector import units as gwpy_units  # noqa: F401
from ..time import to_gps

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType
    from typing import Literal

    from numpy.typing import DTypeLike

    from ..time import SupportsToGps

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# -- type matching -------------------

#: Mapping of LAL type `int` to type `str`
LAL_TYPE_STR: dict[int, str] = {
    lal.I2_TYPE_CODE: "INT2",
    lal.I4_TYPE_CODE: "INT4",
    lal.I8_TYPE_CODE: "INT8",
    lal.U2_TYPE_CODE: "UINT2",
    lal.U4_TYPE_CODE: "UINT4",
    lal.U8_TYPE_CODE: "UINT8",
    lal.S_TYPE_CODE: "REAL4",
    lal.D_TYPE_CODE: "REAL8",
    lal.C_TYPE_CODE: "COMPLEX8",
    lal.Z_TYPE_CODE: "COMPLEX16",
}

#: Mapping of LAL type `str` to type `int`
LAL_TYPE_FROM_STR: dict[str, int] = {v: k for k, v in LAL_TYPE_STR.items()}

#: Mapping of `numpy` types to LAL type `int`
LAL_TYPE_FROM_NUMPY: dict[type, int] = {
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

#: Mapping of `numpy` types to LAL type `str`
LAL_TYPE_STR_FROM_NUMPY: dict[type, str] = {
    k: LAL_TYPE_STR[v] for k, v in LAL_TYPE_FROM_NUMPY.items()
}
#: Mapping of LAL type `str` to `numpy` types
LAL_NUMPY_FROM_TYPE_STR: dict[str, type] = {
    v: k for k, v in LAL_TYPE_STR_FROM_NUMPY.items()
}

#: Regular expression to match LAL type strings
LAL_TYPE_REGEX: re.Pattern[str] = re.compile(r"(U?INT|REAL|COMPLEX)\d+")


def to_lal_type_str(pytype: type | DTypeLike | str | int) -> str:
    """Convert the input python type to a LAL type string.

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
    ValueError
        If the input doesn't map to a LAL type string.
    """
    # noop
    if pytype in LAL_TYPE_FROM_STR:
        return pytype  # type: ignore[return-value]

    # convert type code
    if pytype in LAL_TYPE_STR:
        return LAL_TYPE_STR[pytype]  # type: ignore[index]

    # convert python type
    try:
        dtp: type = numpy.dtype(pytype).type  # type: ignore[arg-type]
        return LAL_TYPE_STR_FROM_NUMPY[dtp]
    except (
        TypeError,  # failed to convert input to dtype
        KeyError,  # dtype didn't match
    ) as exc:
        msg = f"Failed to map '{pytype}' to LAL type string"
        raise ValueError(msg) from exc


# -- Type aliases for common types

#: Type alias for LAL FFT plan types
LALFFTPlanType = (
    lal.REAL4FFTPlan
    | lal.REAL8FFTPlan
    | lal.COMPLEX8FFTPlan
    | lal.COMPLEX16FFTPlan
)
#: Type alias for LAL FrequencySeries types
LALFrequencySeriesType = (
    lal.INT2FrequencySeries
    | lal.INT4FrequencySeries
    | lal.INT8FrequencySeries
    | lal.UINT2FrequencySeries
    | lal.UINT4FrequencySeries
    | lal.UINT8FrequencySeries
    | lal.REAL4FrequencySeries
    | lal.REAL8FrequencySeries
    | lal.COMPLEX8FrequencySeries
    | lal.COMPLEX16FrequencySeries
)
#: Type alias for LAL TimeSeries types
LALTimeSeriesType = (
    lal.INT2TimeSeries
    | lal.INT4TimeSeries
    | lal.INT8TimeSeries
    | lal.UINT2TimeSeries
    | lal.UINT4TimeSeries
    | lal.UINT8TimeSeries
    | lal.REAL4TimeSeries
    | lal.REAL8TimeSeries
    | lal.COMPLEX8TimeSeries
    | lal.COMPLEX16TimeSeries
)
#: Type alias for LAL Vector types
LALVectorType = (
    lal.INT2Vector
    | lal.INT4Vector
    | lal.INT8Vector
    | lal.UINT2Vector
    | lal.UINT4Vector
    | lal.UINT8Vector
    | lal.REAL4Vector
    | lal.REAL8Vector
    | lal.COMPLEX8Vector
    | lal.COMPLEX16Vector
)
#: Type alias for LAL Window types
LALWindowType = lal.REAL4Window | lal.REAL8Window


@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal[""],
    suffix: Literal[
        "AverageSpectrumMedian",
        "AverageSpectrumMedianMean",
        "AverageSpectrumWelch",
    ],
    module: ModuleType = lal,
) -> Callable[
    [LALFrequencySeriesType, LALTimeSeriesType, int, int,
     LALWindowType, LALFFTPlanType],
    int,
]: ...

@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal["Create"],
    suffix: Literal["FFTPlan"],
    module: ModuleType = lal,
) -> Callable[[int, int, int], LALFFTPlanType]: ...

@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal["Create"],
    suffix: Literal["FrequencySeries"],
    module: ModuleType = lal,
) -> Callable[
    [str, lal.LIGOTimeGPS, float, float, lal.Unit, int],
    LALFrequencySeriesType,
]: ...

@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal["Create"],
    suffix: Literal["Sequence"],
    module: ModuleType = lal,
) -> Callable[[int], LALVectorType]: ...

@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal["Create"],
    suffix: Literal["TimeSeries"],
    module: ModuleType = lal,
) -> Callable[
    [str | None, lal.LIGOTimeGPS, float, float, lal.Unit, int],
    LALTimeSeriesType,
]: ...

@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal["Create"],
    suffix: Literal["WindowFromSequence"],
    module: ModuleType = lal,
) -> Callable[[LALVectorType], LALWindowType]: ...

@overload
def find_typed_function(
    pytype: type | DTypeLike,
    prefix: Literal["CreateNamed"],
    suffix: Literal["Window"],
    module: ModuleType = lal,
) -> Callable[[str, float, int], LALWindowType]: ...

def find_typed_function(
    pytype: type | DTypeLike,
    prefix: str,
    suffix: str,
    module: ModuleType = lal,
) -> Callable:
    """Return the lal method for the correct type.

    Parameters
    ----------
    pytype : `type`, `numpy.dtype`
        The python type, or dtype, to map.

    prefix : `str`
        The function name prefix (before the type tag).

    suffix : `str`
        The function name suffix (after the type tag).

    module : `ModuleType`
        The module in which to search for the function.
        Default is `lal`.

    Raises
    ------
    AttributeError
        If the function is not found.

    Examples
    --------
    >>> from gwpy.utils.lal import find_typed_function
    >>> find_typed_function(float, 'Create', 'Sequence')
    <built-in function CreateREAL8Sequence>
    """
    laltype = to_lal_type_str(pytype)
    return getattr(module, f"{prefix}{laltype}{suffix}")


def from_lal_type(laltype: type) -> type:
    """Convert the data type of a LAL instance or type into a numpy data type.

    Parameters
    ----------
    laltype : `SwigPyObject` or `type`
        The input LAL instance or type.

    Returns
    -------
    npytype : `type`
        The numpy data type, such as `numpy.uint32`, `numpy.float64`, etc.

    Raises
    ------
    ValueError
        If the numpy data type cannot be inferred from the LAL object.

    Examples
    --------
    >>> from_lal_type(lal.REAL8TimeSeries)
    numpy.float64

    This also works with instances of LAL series types:

    >>> series = lal.CreateINT4TimeSeries("test", 0, 0, 1, "m", 10)
    >>> from_lal_type(series)
    numpy.int32
    """
    if not isinstance(laltype, type):
        laltype = type(laltype)
    name = laltype.__name__
    match = LAL_TYPE_REGEX.match(name)
    if not match or match[0] not in LAL_NUMPY_FROM_TYPE_STR:
        msg = f"{name!r} has no known numpy type equivalent"
        raise ValueError(msg)
    return LAL_NUMPY_FROM_TYPE_STR[match[0]]


# -- units ---------------------------

_LAL_UNIT_INDEX: list[units.UnitBase] = [
    # the order corresponds to how LAL stores compound units
    units.meter,
    units.kilogram,
    units.second,
    units.ampere,
    units.Kelvin,
    units.Unit("strain"),
    units.count,
]


def to_lal_unit(
    astropy_unit: units.Unit | str,
) -> tuple[lal.Unit, float | complex]:
    """Convert the input unit into a |lal.Unit|_ and a scaling factor.

    Parameters
    ----------
    astropy_unit : `~astropy.units.Unit`, `str`
        The input unit.

    Returns
    -------
    lal_unit : |lal.Unit|_
        The LAL representation of the base unit.

    scale : `float`, `complex`
        The linear scaling factor that should be applied to any associated
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
        If LAL doesn't understand the base units for the input.
    """
    # format incoming unit
    aunit = units.Unit(astropy_unit).decompose()

    # handle scaled units
    pow10 = numpy.log10(aunit.scale)
    if pow10 and pow10.is_integer():
        lunit = lal.Unit(f"10^{int(pow10)}")
        scale = 1
    else:
        lunit = lal.Unit()
        scale = aunit.scale

    # decompose unit into LAL base units
    for base, power in zip(aunit.bases, aunit.powers, strict=True):
        try:  # try this base
            i = _LAL_UNIT_INDEX.index(base)
        except ValueError as exc:
            exc.args = (f"LAL has no unit corresponding to '{base}'",)
            raise
        frac = Fraction(power)
        lunit.unitNumerator[i] = frac.numerator
        lunit.unitDenominatorMinusOne[i] = frac.denominator - 1

    return lunit, scale


def from_lal_unit(
    lunit: lal.Unit,
) -> units.Unit:
    """Convert a LALUnit` into a `~astropy.units.Unit`.

    Parameters
    ----------
    lunit : |lal.Unit|_
        The input unit.

    Returns
    -------
    unit : `~astropy.units.Unit`
        The Astropy representation of the input.

    Raises
    ------
    TypeError
        If ``lunit`` cannot be converted to |lal.Unit|_.

    ValueError
        If Astropy doesn't understand the base units for the input.
    """
    return reduce(
        operator.mul,
        (
            _LAL_UNIT_INDEX[i] ** Fraction(int(num), int(den + 1))
            for i, (num, den) in enumerate(zip(
                lunit.unitNumerator,
                lunit.unitDenominatorMinusOne,
                strict=True,
            ))
        ),
    ) * 10 ** lunit.powerOfTen


def to_lal_ligotimegps(
    gps: SupportsToGps,
) -> lal.LIGOTimeGPS:
    """Convert the given GPS time to a |lal.LIGOTimeGPS|_ object.

    Parameters
    ----------
    gps : `~gwpy.time.LIGOTimeGPS`, `float`, `str`
        Input GPS time, can be anything parsable by :meth:`~gwpy.time.to_gps`.

    Returns
    -------
    ligotimegps : |lal.LIGOTimeGPS|_
        A SWIG-LAL `~lal.LIGOTimeGPS` representation of the given GPS time.
    """
    gps = to_gps(gps)
    return lal.LIGOTimeGPS(gps.gpsSeconds, gps.gpsNanoSeconds)


# -- detectors -----------------------

#: Mapping of IFO names to |lal.FrDetector|_ objects
LAL_DETECTORS: dict[str, lal.FrDetector] = {
    ifo.frDetector.prefix: ifo.frDetector
    for ifo in lal.CachedDetectors
}
