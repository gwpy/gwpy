"""This module provides trivial mappings between numpy data types
and the LAL atomic C-types
"""

from ... import version

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

import numpy

from lal import lal

NUMPY_TYPE = {numpy.int16: lal.LAL_I2_TYPE_CODE,
              numpy.int32: lal.LAL_I4_TYPE_CODE,
              numpy.int64: lal.LAL_I8_TYPE_CODE,
              numpy.uint16: lal.LAL_U2_TYPE_CODE,
              numpy.uint32: lal.LAL_U4_TYPE_CODE,
              numpy.uint64: lal.LAL_U8_TYPE_CODE,
              numpy.float32: lal.LAL_S_TYPE_CODE,
              numpy.float64: lal.LAL_D_TYPE_CODE,
              numpy.complex64: lal.LAL_C_TYPE_CODE,
              numpy.complex128: lal.LAL_Z_TYPE_CODE}
LAL_TYPE = dict((v,k) for k, v in NUMPY_TYPE.iteritems())
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
LAL_STR_TYPE = dict((v,k) for k,v in LAL_TYPE_STR.iteritems())

def lal_type(series):
    type_str = str(type(timeseries)).split("'")[1]
    for key,val in LAL_TYPE_STR.iteritems():
        if re.search(val, type_str):
           return key
    raise TypeError("Cannot determine type for TimeSeries '%s'" % str(t))
