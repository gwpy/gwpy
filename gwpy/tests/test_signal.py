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

"""Unit test for signal module
"""

from compat import unittest

import numpy
from numpy import testing as nptest

from scipy import signal

from astropy import units

from gwpy import signal as gwpy_signal

ONE_HZ = units.Quantity(1, 'Hz')

NOTCH_60HZ = (
    numpy.asarray([ 0.99973536+0.02300468j,  0.99973536-0.02300468j]),
    numpy.asarray([ 0.99954635-0.02299956j,  0.99954635+0.02299956j]),
    0.99981094420429639,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -----------------------------------------------------------------------------

class FilterDesignTestCase(unittest.TestCase):
    """`~unittest.TestCase` for the `gwpy.signal.filter_design` module
    """
    def test_notch_design(self):
        # test simple notch
        zpk = gwpy_signal.notch(60, 16384)
        for a, b in zip(zpk, NOTCH_60HZ):
            nptest.assert_array_almost_equal(a, b)
        # test Quantities
        zpk2 = gwpy_signal.notch(60 * ONE_HZ, 16384 * ONE_HZ)
        for a, b in zip(zpk, zpk2):
            nptest.assert_array_almost_equal(a, b)
