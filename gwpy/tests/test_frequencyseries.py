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

"""Unit test for frequencyseries module
"""

import tempfile

import pytest

import numpy

from scipy import signal

from matplotlib import (use, rc_context)
use('agg')  # nopep8

from astropy import units

from gwpy.frequencyseries import (FrequencySeries, SpectralVariance)
from gwpy.plotter import (FrequencySeriesPlot, FrequencySeriesAxes)

import utils
from test_array import (TestSeries, TestArray2D)
from compat import unittest

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -----------------------------------------------------------------------------
#
#     gwpy.frequencyseries.core
#
# -----------------------------------------------------------------------------

# -- FrequencySeries ----------------------------------------------------------

class TestFrequencySeries(TestSeries):
    TEST_CLASS = FrequencySeries

    # -- test properties ------------------------

    def test_f0(self, array):
        assert array.f0 is array.x0
        array.f0 = 4
        assert array.f0 == 4 * units.Hz

    def test_df(self, array):
        assert array.df is array.dx
        array.df = 4
        assert array.df == 4 * units.Hz

    def test_frequencies(self, array):
        assert array.frequencies is array.xindex
        utils.assert_quantity_equal(
            array.frequencies, numpy.arange(array.size) * array.df + array.f0)

    # -- test methods ---------------------------

    def test_plot(self, array):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot()
            assert isinstance(plot, FrequencySeriesPlot)
            assert isinstance(plot.gca(), FrequencySeriesAxes)
            l = plot.gca().lines[0]
            utils.assert_array_equal(l.get_xdata(), array.xindex.value)
            utils.assert_array_equal(l.get_ydata(), array.value)
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)

    def test_filter(self, array):
        a2 = array.filter([100], [1], 1e-2)
        assert isinstance(a2, type(array))
        utils.assert_quantity_equal(a2.frequencies, array.frequencies)

        # manually rebuild the filter to test it works
        b, a, = signal.zpk2tf([100], [1], 1e-2)
        fresp = abs(signal.freqs(b, a, array.frequencies.value)[1])
        utils.assert_array_equal(a2.value, fresp * array.value)

    def test_zpk(self, array):
        a2 = array.zpk([100], [1], 1e-2)
        assert isinstance(a2, type(array))
        utils.assert_quantity_equal(a2.frequencies, array.frequencies)

    @utils.skip_missing_dependency('lal')
    def test_to_from_lal(self, array):
        import lal

        array.epoch = 0

        # check that to + from returns the same array
        lalts = array.to_lal()
        a2 = type(array).from_lal(lalts)
        utils.assert_quantity_sub_equal(array, a2, exclude=['name', 'channel'])
        assert a2.name == array.name

        # test copy=False
        a2 = type(array).from_lal(lalts, copy=False)
        assert numpy.shares_memory(a2.value, lalts.data.data)

        # test units
        array.override_unit('undef')
        with pytest.warns(UserWarning):
            lalts = array.to_lal()
        assert lalts.sampleUnits == lal.DimensionlessUnit
        a2 = self.TEST_CLASS.from_lal(lalts)
        assert a2.unit is units.dimensionless_unscaled

    @utils.skip_missing_dependency('pycbc')
    def test_to_from_pycbc(self, array):
        from pycbc.types import FrequencySeries as PyCBCFrequencySeries

        array.epoch = 0

        # test default conversion
        pycbcfs = array.to_pycbc()
        assert isinstance(pycbcfs, PyCBCFrequencySeries)
        utils.assert_array_equal(array.value, pycbcfs.data)
        assert array.f0.value == 0 * units.Hz
        assert array.df.value == pycbcfs.delta_f
        assert array.epoch.gps == pycbcfs.epoch

        # go back and check we get back what we put in in the first place
        a2 = type(array).from_pycbc(pycbcfs)
        utils.assert_quantity_sub_equal(
            array, a2, exclude=['name', 'unit', 'channel'])

        # test copy=False
        a2 = type(array).from_pycbc(array.to_pycbc(copy=False), copy=False)
        assert numpy.shares_memory(array.value, a2.value)

    @pytest.mark.parametrize('format', [
        'txt',
        'csv',
    ])
    def test_read_write(self, array, format):
        utils.test_read_write(
            array, format,
            assert_equal=utils.assert_quantity_sub_equal,
            assert_kw={'exclude': ['name', 'channel', 'unit', 'epoch']})


# -----------------------------------------------------------------------------
#
#     gwpy.frequencyseries.hist
#
# -----------------------------------------------------------------------------

# -- SpectralVaraince ---------------------------------------------------------

class TestSpectralVariance(TestArray2D):
    TEST_CLASS = SpectralVariance

    # -- helpers --------------------------------

    @classmethod
    def setup_class(cls):
        super(TestSpectralVariance, cls).setup_class()
        cls.bins = numpy.linspace(0, 1e5, cls.data.shape[1] + 1, endpoint=True)

    @classmethod
    def create(cls, *args, **kwargs):
        args = list(args)
        args.insert(0, cls.bins)
        return super(TestSpectralVariance, cls).create(*args, **kwargs)

    # -- test properties ------------------------

    def test_y0(self, array):
        assert array.y0 == self.bins[0]
        with pytest.raises(AttributeError):
            array.y0 = 0

    def test_dy(self, array):
        assert array.dy == self.bins[1] - self.bins[0]
        with pytest.raises(AttributeError):
            array.dy = 0

    def test_yunit(self):
        return NotImplemented

    def test_yspan(self):
        return NotImplemented

    def test_yindex(self, array):
        utils.assert_array_equal(array.yindex, array.bins[:-1])

    # -- test methods ---------------------------

    def test_init(self, array):
        utils.assert_array_equal(array.value, self.data)
        utils.assert_array_equal(array.bins.value, self.bins)
        assert array.x0 == 0 * units.Hertz
        assert array.df == 1 * units.Hertz
        assert array.y0 == self.bins[0]
        assert array.dy == self.bins[1] - self.bins[0]

    def test_is_compatible(self, array):
        return super(TestArray2D, self).test_is_compatible(array)

    def test_plot(self, array):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot()
            assert isinstance(plot, FrequencySeriesPlot)
            assert isinstance(plot.gca(), FrequencySeriesAxes)
            assert len(plot.gca().collections) == 1
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)

    def test_value_at(self, array):
        assert array.value_at(5, self.bins[3]) == (
            self.data[5][3] * array.unit)
        assert array.value_at(8 * array.xunit,
                              self.bins[1] * array.yunit) == (
            self.data[8][1] * array.unit)
        with pytest.raises(IndexError):
            array.value_at(1.6, 5.8)


if __name__ == '__main__':
    unittest.main()
