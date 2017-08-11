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

from matplotlib import (use, rc_context)
use('agg')  # nopep8

from astropy import units

from gwpy.spectrogram import Spectrogram
from gwpy.plotter import (TimeSeriesPlot, TimeSeriesAxes)

import utils
from test_array import TestArray2D

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -----------------------------------------------------------------------------
#
#     gwpy.spectrogram.core
#
# -----------------------------------------------------------------------------

# -- Spectrogram --------------------------------------------------------------

class TestSpectrogram(TestArray2D):
    """Tests of `gwpy.spectrogram.Spectrogram`
    """
    TEST_CLASS = Spectrogram

    def test_epoch(self, array):
        assert array.epoch.gps == array.x0.value

    def test_value_at(self, array):
        super(TestSpectrogram, self).test_value_at(array)
        print(array)
        v = array.value_at(5000 * units.millisecond,
                           2000 * units.milliHertz)
        assert v == self.data[5][2] * array.unit

    @pytest.mark.parametrize('ratio', ('mean', 'median'))
    def test_ratio(self, array, ratio):
        rat = array.ratio(ratio)
        array_meth = getattr(array, ratio)
        utils.assert_quantity_sub_equal(rat, array / array_meth(axis=0))

    def test_from_spectra(self, array):
        min_ = self.TEST_ARRAY.min(axis=0)
        max_ = self.TEST_ARRAY.max(axis=0)
        mean = self.TEST_ARRAY.mean(axis=0)
        # check basic stack works
        new = self.TEST_ARRAY.from_spectra(mean, min_, max_, dt=1)
        assert new.shape == (3, min_.size)
        assert new.name == mean.name
        assert new.epoch == mean.epoch
        assert new.f0 == mean.f0
        assert new.df == mean.df
        assert new.unit == mean.unit
        assert new.dt == 1 * units.second
        utils.assert_array_equal(
            new.value, numpy.vstack((mean.value, min_.value, max_.value)))
        # check kwargs
        new = self.TEST_ARRAY.from_spectra(
            mean, min_, max_,
            dt=2, epoch=0, f0=100, df=.5, unit='meter', name='test')
        assert new.name == 'test'
        assert new.epoch.gps == 0
        assert new.f0 == 100 * units.Hertz
        assert new.df == 0.5 * units.Hertz
        assert new.unit == units.meter
        # check error on timing
        with pytest.raises(ValueError):
            self.TEST_ARRAY.from_spectra(mean)

    def test_crop_frequencies(self):
        array = self.create(f0=0, df=1)
        # test simple
        array2 = array.crop_frequencies()
        utils.assert_quantity_sub_equal(array, array2)
        # test normal
        array2 = array.crop_frequencies(2, 5)
        utils.assert_array_equal(array2.value, array.value[:, 2:5])
        assert array2.f0 == 2 * units.Hertz
        assert array2.df == array.df
        # test warnings
        with pytest.warns(UserWarning):
            array.crop_frequencies(array.yspan[0]-1, array.yspan[1])
        with pytest.warns(UserWarning):
            array.crop_frequencies(array.yspan[0], array.yspan[1]+1)

    def test_plot(self, array):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot()
            assert isinstance(plot, TimeSeriesPlot)
            assert isinstance(plot.gca(), TimeSeriesAxes)
            assert plot.gca().lines == []
            assert len(plot.gca().collections) == 1
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)
            plot.close()
