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

"""Unit test for spectrogram module
"""

import tempfile

import pytest

import numpy
from numpy import testing as nptest

from astropy import units

from gwpy.spectrogram import Spectrogram

from test_array import Array2DTestCase

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -----------------------------------------------------------------------------

class SpectrogramTestCase(Array2DTestCase):
    """`~unittest.TestCase` for the `~gwpy.spectrogram.Spectrogram` class
    """
    TEST_CLASS = Spectrogram

    def test_epoch(self):
        array = self.create()
        self.assertEquals(array.epoch.gps, array.x0.value)

    def test_ratio(self):
        mean_ = self.TEST_ARRAY.ratio('mean')
        nptest.assert_array_equal(
            mean_.value,
            self.TEST_ARRAY.value / self.TEST_ARRAY.mean(axis=0).value)
        median_ = self.TEST_ARRAY.ratio('median')
        nptest.assert_array_equal(
            median_.value,
            self.TEST_ARRAY.value / self.TEST_ARRAY.median(axis=0).value)

    def test_from_spectra(self):
        min_ = self.TEST_ARRAY.min(axis=0)
        max_ = self.TEST_ARRAY.max(axis=0)
        mean = self.TEST_ARRAY.mean(axis=0)
        # check basic stack works
        new = self.TEST_ARRAY.from_spectra(mean, min_, max_, dt=1)
        self.assertEqual(new.shape, (3, min_.size))
        self.assertEqual(new.name, mean.name)
        self.assertEqual(new.epoch, mean.epoch)
        self.assertEqual(new.f0, mean.f0)
        self.assertEqual(new.df, mean.df)
        self.assertEqual(new.unit, mean.unit)
        self.assertEqual(new.dt, 1 * units.second)
        nptest.assert_array_equal(
            new.value, numpy.vstack((mean.value, min_.value, max_.value)))
        # check kwargs
        new = self.TEST_ARRAY.from_spectra(
            mean, min_, max_,
            dt=2, epoch=0, f0=100, df=.5, unit='meter', name='test')
        self.assertEqual(new.name, 'test')
        self.assertEqual(new.epoch.gps, 0)
        self.assertEqual(new.f0, 100 * units.Hertz)
        self.assertEqual(new.df, 0.5 * units.Hertz)
        self.assertEqual(new.unit, units.meter)
        # check error on timing
        self.assertRaises(ValueError, self.TEST_ARRAY.from_spectra, mean)

    def test_crop_frequencies(self):
        array = self.create(f0=0, df=1)
        # test simple
        array2 = array.crop_frequencies()
        self.assertArraysEqual(array, array2)
        # test normal
        array2 = array.crop_frequencies(2, 5)
        nptest.assert_array_equal(array2.value, array.value[:, 2:5])
        self.assertEqual(array2.f0, 2 * units.Hertz)
        self.assertEqual(array2.df, array.df)
        # test warnings
        with pytest.warns(UserWarning):
            array.crop_frequencies(array.yspan[0]-1, array.yspan[1])
        with pytest.warns(UserWarning):
            array.crop_frequencies(array.yspan[0], array.yspan[1]+1)

    def test_plot(self):
        plot = self.TEST_ARRAY.plot()
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            plot.save(f.name)


if __name__ == '__main__':
    unittest.main()
