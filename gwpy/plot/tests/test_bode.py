# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Tests for `gwpy.plot.filter`
"""

import numpy
from numpy import testing as nptest

from scipy import signal

from ...frequencyseries import FrequencySeries
from .. import BodePlot
from .utils import FigureTestBase

# design ZPK for BodePlot test
ZPK = [100], [1], 1e-2
FREQUENCIES, MAGNITUDE, PHASE = signal.bode(ZPK, n=100)


class TestBodePlot(FigureTestBase):
    FIGURE_CLASS = BodePlot

    def test_init(self, fig):
        assert len(fig.axes) == 2
        maxes, paxes = fig.axes
        # test magnigtude axes
        assert maxes.get_xscale() == 'log'
        assert maxes.get_xlabel() == ''
        assert maxes.get_yscale() == 'linear'
        assert maxes.get_ylabel() == 'Magnitude [dB]'
        # test phase axes
        assert paxes.get_xscale() == 'log'
        assert paxes.get_xlabel() == 'Frequency [Hz]'
        assert paxes.get_yscale() == 'linear'
        assert paxes.get_ylabel() == 'Phase [deg]'

    def test_add_filter(self, fig):
        lm, lp = fig.add_filter(ZPK, analog=True)
        assert lm is fig.maxes.get_lines()[-1]
        assert lp is fig.paxes.get_lines()[-1]
        nptest.assert_array_equal(lm.get_xdata(), FREQUENCIES)
        nptest.assert_array_equal(lm.get_ydata(), MAGNITUDE)
        nptest.assert_array_equal(lp.get_xdata(), FREQUENCIES)
        nptest.assert_array_almost_equal(lp.get_ydata(), PHASE)

    def test_init_with_filter(self):
        fig = self.FIGURE_CLASS(ZPK, analog=True, title='ZPK')
        lm = fig.maxes.get_lines()[0]
        lp = fig.paxes.get_lines()[0]
        nptest.assert_array_equal(lm.get_xdata(), FREQUENCIES)
        nptest.assert_array_equal(lm.get_ydata(), MAGNITUDE)
        nptest.assert_array_equal(lp.get_xdata(), FREQUENCIES)
        nptest.assert_array_almost_equal(lp.get_ydata(), PHASE)
        assert fig.maxes.get_title() == 'ZPK'
        self.save_and_close(fig)

    def test_add_frequencyseries(self, fig):
        fs = FrequencySeries(numpy.random.random(100).astype(complex))
        fig.add_frequencyseries(fs)
        lm = fig.maxes.get_lines()[0]
        lp = fig.paxes.get_lines()[0]
        nptest.assert_array_equal(lm.get_xdata(), fs.xindex.value)
        nptest.assert_array_equal(
            lm.get_ydata(), 20 * numpy.log10(numpy.absolute(fs.value)))
        nptest.assert_array_almost_equal(lp.get_ydata(), numpy.angle(fs.value))

    def test_init_with_frequencyseries(self):
        fs = FrequencySeries(numpy.random.random(100).astype(complex))
        fig = self.FIGURE_CLASS(fs)
        lm = fig.maxes.get_lines()[0]
        nptest.assert_array_equal(lm.get_xdata(), fs.xindex.value)
        self.save_and_close(fig)
