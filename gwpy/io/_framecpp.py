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

# -- detectors ----------------------------------------------------------------

DetectorLocation = IntEnum(
    "DetectorLocation",
    {key[18:]: val for key, val in vars(frameCPP).items() if
     key.startswith("DETECTOR_LOCATION_")},
)


# -- type mapping -------------------------------------------------------------

class FrVectType(IntEnum, NumpyTypeEnum):
    INT8 = frameCPP.FrVect.FR_VECT_C
    INT16 = frameCPP.FrVect.FR_VECT_2S
    INT32 = frameCPP.FrVect.FR_VECT_4S
    INT64 = frameCPP.FrVect.FR_VECT_8S
    FLOAT32 = frameCPP.FrVect.FR_VECT_4R
    FLOAT64 = frameCPP.FrVect.FR_VECT_8R
    COMPLEX64 = frameCPP.FrVect.FR_VECT_8C
    COMPLEX128 = frameCPP.FrVect.FR_VECT_16C
    BYTES = frameCPP.FrVect.FR_VECT_STRING
    UINT8 = frameCPP.FrVect.FR_VECT_1U
    UINT16 = frameCPP.FrVect.FR_VECT_2U
    UINT32 = frameCPP.FrVect.FR_VECT_4U
    UINT64 = frameCPP.FrVect.FR_VECT_8U


# -- compression types --------------------------------------------------------

class Compression(IntEnum):
    RAW = frameCPP.FrVect.RAW
    GZIP = frameCPP.FrVect.GZIP
    DIFF_GZIP = frameCPP.FrVect.DIFF_GZIP
    ZERO_SUPPRESS_WORD_2 = frameCPP.FrVect.ZERO_SUPPRESS_WORD_2
    ZERO_SUPPRESS_WORD_4 = frameCPP.FrVect.ZERO_SUPPRESS_WORD_4
    ZERO_SUPPRESS_WORD_8 = frameCPP.FrVect.ZERO_SUPPRESS_WORD_8
    ZERO_SUPPRESS_OTHERWISE_GZIP = frameCPP.FrVect.ZERO_SUPPRESS_OTHERWISE_GZIP


class DefaultCompressionLevel(IntEnum):
    RAW = 0
    GZIP = 6
    DIFF_GZIP = 6
    ZERO_SUPPRESS_WORD_2 = 0
    ZERO_SUPPRESS_WORD_4 = 0
    ZERO_SUPPRESS_WORD_8 = 0
    ZERO_SUPPRESS_OTHERWISE_GZIP = 6


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
