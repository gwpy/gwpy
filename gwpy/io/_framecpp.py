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

"""GWF I/O utilities for frameCPP

This module is where all constants and variables that require
frameCPP should go, as opposed to functions which can import
the necessary objects at runtime.
"""

from enum import IntEnum

from LDAStools import frameCPP

from ..utils.enum import NumpyTypeEnum

_FrVect = frameCPP.FrVect

# -- detectors ----------------------------------------------------------------

DetectorLocation = IntEnum(
    "DetectorLocation",
    {key[18:]: val for key, val in vars(frameCPP).items() if
     key.startswith("DETECTOR_LOCATION_")},
)


# -- type mapping -------------------------------------------------------------

class FrVectType(IntEnum, NumpyTypeEnum):
    INT8 = _FrVect.FR_VECT_C
    INT16 = _FrVect.FR_VECT_2S
    INT32 = _FrVect.FR_VECT_4S
    INT64 = _FrVect.FR_VECT_8S
    FLOAT32 = _FrVect.FR_VECT_4R
    FLOAT64 = _FrVect.FR_VECT_8R
    COMPLEX64 = _FrVect.FR_VECT_8C
    COMPLEX128 = _FrVect.FR_VECT_16C
    BYTES = _FrVect.FR_VECT_STRING
    UINT8 = _FrVect.FR_VECT_1U
    UINT16 = _FrVect.FR_VECT_2U
    UINT32 = _FrVect.FR_VECT_4U
    UINT64 = _FrVect.FR_VECT_8U


# -- compression types --------------------------------------------------------

try:
    _FrVect.ZERO_SUPPRESS
except AttributeError:  # python-ldas-tools-framecpp < 3.0.0
    class Compression(IntEnum):
        RAW = _FrVect.RAW
        GZIP = _FrVect.GZIP
        DIFF_GZIP = _FrVect.DIFF_GZIP
        ZERO_SUPPRESS_WORD_2 = _FrVect.ZERO_SUPPRESS_WORD_2
        ZERO_SUPPRESS_WORD_4 = _FrVect.ZERO_SUPPRESS_WORD_4
        ZERO_SUPPRESS_WORD_8 = _FrVect.ZERO_SUPPRESS_WORD_8
        ZERO_SUPPRESS_OTHERWISE_GZIP = _FrVect.ZERO_SUPPRESS_OTHERWISE_GZIP
else:
    class Compression(IntEnum):
        RAW = _FrVect.RAW
        BIGENDIAN_RAW = _FrVect.BIGENDIAN_RAW
        LITTLEENDIAN_RAW = _FrVect.LITTLEENDIAN_RAW
        GZIP = _FrVect.GZIP
        BIGENDIAN_GZIP = _FrVect.BIGENDIAN_GZIP
        LITTLEENDIAN_GZIP = _FrVect.LITTLEENDIAN_GZIP
        DIFF_GZIP = _FrVect.DIFF_GZIP
        BIGENDIAN_DIFF_GZIP = _FrVect.BIGENDIAN_DIFF_GZIP
        LITTLEENDIAN_DIFF_GZIP = _FrVect.LITTLEENDIAN_DIFF_GZIP
        ZERO_SUPPRESS = _FrVect.ZERO_SUPPRESS
        BIGENDIAN_ZERO_SUPPRESS = _FrVect.BIGENDIAN_ZERO_SUPPRESS
        LITTLEENDIAN_ZERO_SUPPRESS = _FrVect.LITTLEENDIAN_ZERO_SUPPRESS
        ZERO_SUPPRESS_OTHERWISE_GZIP = _FrVect.ZERO_SUPPRESS_OTHERWISE_GZIP

# compression level is '6' for all GZip compressions, otherwise 0 (none)
DefaultCompressionLevel = IntEnum(
    "DefaultCompressionLevel",
    {k: 6 if "GZIP" in k else 0 for k in Compression.__members__},
)


# -- Proc data types ----------------------------------------------------------

class FrProcDataType(IntEnum):
    UNKNOWN = frameCPP.FrProcData.UNKNOWN_TYPE
    TIME_SERIES = frameCPP.FrProcData.TIME_SERIES
    FREQUENCY_SERIES = frameCPP.FrProcData.FREQUENCY_SERIES
    OTHER_1D_SERIES_DATA = frameCPP.FrProcData.OTHER_1D_SERIES_DATA
    TIME_FREQUENCY = frameCPP.FrProcData.TIME_FREQUENCY
    WAVELETS = frameCPP.FrProcData.WAVELETS
    MULTI_DIMENSIONAL = frameCPP.FrProcData.MULTI_DIMENSIONAL


class FrProcDataSubType(IntEnum):
    UNKNOWN = frameCPP.FrProcData.UNKNOWN_SUB_TYPE
    DFT = frameCPP.FrProcData.DFT
    AMPLITUDE_SPECTRAL_DENSITY = frameCPP.FrProcData.AMPLITUDE_SPECTRAL_DENSITY
    POWER_SPECTRAL_DENSITY = frameCPP.FrProcData.POWER_SPECTRAL_DENSITY
    CROSS_SPECTRAL_DENSITY = frameCPP.FrProcData.CROSS_SPECTRAL_DENSITY
    COHERENCE = frameCPP.FrProcData.COHERENCE
    TRANSFER_FUNCTION = frameCPP.FrProcData.TRANSFER_FUNCTION
