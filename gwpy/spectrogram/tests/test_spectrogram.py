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

"""Unit tests for :mod:`gwpy.spectrogram.spectrogram`
"""

from io import BytesIO

import pytest

import numpy

from scipy import signal

from matplotlib import rc_context

from astropy import units

from ...testing import utils
from ...types.tests.test_array2d import TestArray2D as _TestArray2D
from .. import Spectrogram

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class TestSpectrogram(_TestArray2D):
    """Tests of `gwpy.spectrogram.Spectrogram`
    """
    TEST_CLASS = Spectrogram

    def test_new(self):
        super().test_new()

        # check handling of epoch vs t0
        a = self.create(epoch=10)
        b = self.create(t0=10)
        utils.assert_quantity_sub_equal(a, b)
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS(self.data, epoch=1, t0=1)
        assert str(exc.value) == 'give only one of epoch or t0'

        # check times
        times = numpy.arange(self.data.shape[0])
        a = self.create(times=times)
        utils.assert_quantity_equal(a.times, times * units.second)

    def test_epoch(self, array):
        assert array.epoch.gps == array.x0.value

    def test_value_at(self, array):
        super().test_value_at(array)
        v = array.value_at(5000 * units.millisecond,
                           2000 * units.milliHertz)
        assert v == self.data[5][2] * array.unit

    @pytest.mark.parametrize('ratio', ('mean', 'median'))
    def test_ratio(self, array, ratio):
        rat = array.ratio(ratio)
        array_meth = getattr(array, ratio)
        utils.assert_quantity_sub_equal(rat, array / array_meth(axis=0))

        with pytest.raises(ValueError):
            array.ratio('blah')

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
        self.TEST_ARRAY.from_spectra(mean, dt=array.dt)

        # check error on inputs
        with pytest.raises(ValueError):
            self.TEST_ARRAY.from_spectra(mean, mean[1:])
        with pytest.raises(ValueError):
            self.TEST_ARRAY.from_spectra(mean, mean[::2])

    def test_crop_frequencies(self):
        array = self.create(f0=0, df=1)

        # test simple
        array2 = array.crop_frequencies()
        utils.assert_quantity_sub_equal(array, array2)
        assert numpy.may_share_memory(array.value, array2.value)

        # test normal
        array2 = array.crop_frequencies(2, 5)
        utils.assert_array_equal(array2.value, array.value[:, 2:5])
        assert array2.f0 == 2 * units.Hertz
        assert array2.df == array.df

        # test copy
        array2 = array.crop_frequencies(copy=True)
        assert not numpy.may_share_memory(array.value, array2.value)

        # test warnings
        with pytest.warns(UserWarning):
            array.crop_frequencies(array.yspan[0]-1, array.yspan[1])
        with pytest.warns(UserWarning):
            array.crop_frequencies(array.yspan[0], array.yspan[1]+1)

    @pytest.mark.parametrize('method', ('imshow', 'pcolormesh'))
    def test_plot(self, array, method):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot(method=method)
            ax = plot.gca()
            assert len(ax.lines) == 0
            if method == 'imshow':
                assert len(ax.images) == 1
            else:
                assert len(ax.collections) == 1
            assert ax.get_epoch() == array.x0.value
            assert ax.get_xlim() == array.xspan
            assert ax.get_ylim() == array.yspan
            plot.save(BytesIO(), format='png')
            plot.close()

    def test_zpk(self, array):
        zpk = [], [1], 1
        utils.assert_quantity_sub_equal(
            array.zpk(*zpk), array.filter(*zpk, analog=True))

    def test_filter(self):
        array = self.create(t0=0, dt=1/1024., f0=0, df=1)

        # build filter
        zpk = [], [1], 1
        lti = signal.lti(*zpk)
        fresp = numpy.nan_to_num(abs(
            lti.freqresp(w=array.frequencies.value)[1]))

        # test simple filter
        a2 = array.filter(*zpk)
        utils.assert_array_equal(array * fresp, a2)

        # test inplace filtering
        array.filter(lti, inplace=True)
        utils.assert_array_equal(array, a2)

        # test errors
        with pytest.raises(TypeError):
            array.filter(lti, blah=1)

    def test_read_write_hdf5(self):
        array = self.create(name='X1:TEST')
        utils.test_read_write(array, 'hdf5', write_kw={'overwrite': True})

    def test_percentile(self):
        array = self.create(name='Test', unit='m')
        a2 = array.percentile(50)
        utils.assert_quantity_sub_equal(array.median(axis=0), a2,
                                        exclude=('name',))
        assert a2.name == 'Test: 50th percentile'
        assert a2.unit == array.unit
