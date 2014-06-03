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

import numpy

import lal

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

# LAL type enum
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
LAL_TYPE_FROM_STR = dict((v, k) for k, v in LAL_TYPE_STR.iteritems())

# map numpy dtypes to LAL type codes
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
LAL_TYPE_STR_FROM_NUMPY = dict((key, LAL_TYPE_STR[value]) for (key, value) in
                               LAL_TYPE_FROM_NUMPY.iteritems())
