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

"""Unit test for data classes
"""

import os
import os.path
import sys

import numpy
from numpy import testing as nptest

if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

from astropy import units
from astropy.time import Time

from gwpy import version
from gwpy.data import (Array, Series, Array2D)
from gwpy.detector import Channel

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

GPS_EPOCH = 12345
TIME_EPOCH = Time(12345, format='gps', scale='utc')
CHANNEL_NAME = 'X1:TEST-CHANNEL'
CHANNEL = Channel(CHANNEL_NAME)


class CommonTests(object):
    TEST_CLASS = Array

    def setUp(self):
        self.data = numpy.arange(10)
        self.datasq = self.data ** 2

    def test_init(self):
        """Test Array creation
        """
        # test basic empty contructor
        self.assertRaises(TypeError, Array)
        self.assertRaises(IndexError, Array, [])
        # test with some data
        array = self.TEST_CLASS(self.data)
        nptest.assert_array_equal(array.value, self.data)
        self.assertIsNone(array.unit)
        self.assertIsNone(array.name)
        self.assertIsNone(array.epoch)
        self.assertIsNone(array.channel)
        # test metadata
        array = self.TEST_CLASS(self.data, 'Hz', name='TEST',
                                channel=CHANNEL_NAME, epoch=GPS_EPOCH)
        self.assertEqual(array.unit, units.Hz)
        self.assertEqual(array.name, 'TEST')
        self.assertEqual(array.epoch, TIME_EPOCH)
        self.assertEqual(array.channel, CHANNEL)
        return array

    def test_math(self):
        """Test Array math operations
        """
        array = self.TEST_CLASS(self.data, unit='Hz')
        # test basic operations
        arraysq = array ** 2
        nptest.assert_array_equal(arraysq.value, self.datasq)
        self.assertEqual(arraysq.unit, units.Hz ** 2)
        self.assertEqual(arraysq.name, array.name)
        self.assertEqual(arraysq.epoch, array.epoch)
        self.assertEqual(arraysq.channel, array.channel)

    def test_copy(self):
        """Test Array.copy
        """
        array = self.TEST_CLASS(self.data, unit='Hz')
        array2 = array.copy()
        nptest.assert_array_equal(array.value, array2.value)
        self.assertEqual(array.unit, array2.unit)


class ArrayTestCase(CommonTests, unittest.TestCase):
    pass


class SeriesTestCase(CommonTests, unittest.TestCase):
    TEST_CLASS = Series

    def setUp(self):
        super(SeriesTestCase, self).setUp()
        self.index = units.Quantity(numpy.arange(10) * 4, units.m)

    def test_init(self):
        series = super(SeriesTestCase, self).test_init()
        self.assertEqual(series.x0, units.Quantity(0))
        self.assertEqual(series.dx, units.Quantity(1))
        series = self.TEST_CLASS(self.data, 'Hz', dx=4*units.m)
        self.assertEqual(series.x0, 0*units.m)
        self.assertEqual(series.dx, 4*units.m)
        self.assertFalse(hasattr(series, '_xindex'))
        nptest.assert_array_equal(series.xindex, self.index)


if __name__ == '__main__':
    unittest.main()
