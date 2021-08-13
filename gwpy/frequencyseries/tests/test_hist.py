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

"""Unit test for frequencyseries module
"""

from io import BytesIO

import pytest

import numpy

from matplotlib import rc_context

from astropy import units

from ...segments import Segment
from ...testing import utils
from ...types.tests.test_array2d import TestArray2D as _TestArray2D
from .. import SpectralVariance

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class TestSpectralVariance(_TestArray2D):
    TEST_CLASS = SpectralVariance

    # -- helpers --------------------------------

    @classmethod
    def setup_class(cls):
        super().setup_class()
        cls.bins = numpy.linspace(0, 1e5, cls.data.shape[1] + 1, endpoint=True)

    @classmethod
    def create(cls, *args, **kwargs):
        args = list(args)
        args.insert(0, cls.bins)
        return super().create(*args, **kwargs)

    # -- test properties ------------------------

    def test_y0(self, array):
        assert array.y0 == self.bins[0]
        with pytest.raises(AttributeError):
            array.y0 = 0

    def test_dy(self, array):
        assert array.dy == self.bins[1] - self.bins[0]
        with pytest.raises(AttributeError):
            array.dy = 0

    def test_yunit(self, array):
        assert array.unit == array.bins.unit

    def test_yspan(self, array):
        yspan = array.yspan
        assert isinstance(yspan, Segment)
        assert yspan == (self.bins[0], self.bins[-1])

    def test_yindex(self, array):
        utils.assert_array_equal(array.yindex, array.bins[:-1])

    def test_transpose(self, array):
        return NotImplemented

    # -- test utilities -------------------------

    def test_getitem(self, array):
        utils.assert_quantity_sub_equal(
            array[0::2, 0],
            self.TEST_CLASS._rowclass(
                array.value[0::2, 0], x0=array.x0, dx=array.dx*2,
                name=array.name, unit=array.unit, channel=array.channel,
                epoch=array.epoch,
            ),
        )
        with pytest.raises(NotImplementedError) as exc:
            array[0, ::2]
        assert str(exc.value) == 'cannot slice SpectralVariance across bins'

    # -- test methods ---------------------------

    def test_init(self, array):
        utils.assert_array_equal(array.value, self.data)
        utils.assert_array_equal(array.bins.value, self.bins)
        assert array.x0 == 0 * units.Hertz
        assert array.df == 1 * units.Hertz
        assert array.y0 == self.bins[0]
        assert array.dy == self.bins[1] - self.bins[0]

    def test_is_compatible_yindex(self, array):
        return NotImplemented

    def test_is_compatible_error_yindex(self, array):
        return NotImplemented

    def test_plot(self, array):
        with rc_context(rc={'text.usetex': False}):
            plot = array.plot(yscale='linear')
            assert len(plot.gca().collections) == 1
            plot.save(BytesIO(), format='png')
            plot.close()

    def test_value_at(self, array):
        assert array.value_at(5, self.bins[3]) == (
            self.data[5][3] * array.unit)
        assert array.value_at(8 * array.xunit,
                              self.bins[1] * array.yunit) == (
            self.data[8][1] * array.unit)
        with pytest.raises(IndexError):
            array.value_at(1.6, 5.8)
