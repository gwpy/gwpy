# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Unit test for frequencyseries module."""

from io import BytesIO

import numpy
import pytest
from astropy import units
from matplotlib import rc_context

from ...segments import Segment
from ...testing import utils
from ...types.tests.test_array2d import TestArray2D as _TestArray2D
from .. import SpectralVariance

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class TestSpectralVariance(_TestArray2D[SpectralVariance]):
    """Tests for `SpectralVariance`."""

    TEST_CLASS = SpectralVariance
    bins: numpy.ndarray

    # -- helpers ---------------------

    @classmethod
    def setup_class(cls, dtype=None):
        """Configure class-level parameters."""
        super().setup_class(dtype=dtype)
        cls.bins = numpy.linspace(0, 1e5, cls.data.shape[1] + 1, endpoint=True)

    @classmethod
    def create(cls, *args, **kwargs):
        """Create a `SpectralVariance` instance for testing."""
        return super().create(cls.bins, *args, **kwargs)

    # -- test properties -------------

    def test_y0(self, array):
        """Test `SpectralVariance.y0`."""
        assert array.y0 == self.bins[0]
        with pytest.raises(AttributeError):
            array.y0 = 0

    def test_dy(self, array):
        """Test `SpectralVariance.dy`."""
        assert array.dy == self.bins[1] - self.bins[0]
        with pytest.raises(AttributeError):
            array.dy = 0

    def test_yunit(self):
        """Test `SpectralVariance.unit`."""
        array = self.create()
        assert array.unit == array.bins.unit

    def test_yspan(self):
        """Test `SpectralVariance.yspan`."""
        array = self.create()
        assert isinstance(array.yspan, Segment)
        assert array.yspan == (array.bins[0], array.bins[-1])

    def test_yindex(self):
        """Test `SpectralVariance.yindex`."""
        array = self.create()
        utils.assert_array_equal(array.yindex, array.bins[:-1])

    def test_transpose(self, array):
        """Test `SpectralVariance.T`."""
        with pytest.raises(NotImplementedError):
            array.T  # noqa: B018

    # -- test utilities --------------

    def test_getitem(self, array):
        """Test `SpectralVariance.__getitem__`."""
        utils.assert_quantity_sub_equal(
            array[0::2, 0],
            self.TEST_CLASS._rowclass(
                array.value[0::2, 0], x0=array.x0, dx=array.dx*2,
                name=array.name, unit=array.unit, channel=array.channel,
                epoch=array.epoch,
            ),
        )
        with pytest.raises(
            NotImplementedError,
            match=r"^cannot slice SpectralVariance across bins$",
        ):
            array[0, ::2]

    # -- test i/o --------------------

    def test_read_write_csv(self, array):  # noqa: ARG002
        """Test reading and writing CSV format."""
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")

    # -- test methods ----------------

    def test_init(self, array):
        """Test `SpectralVariance.__init__`."""
        utils.assert_array_equal(array.value, self.data)
        utils.assert_array_equal(array.bins.value, self.bins)
        assert array.x0 == 0 * units.Hertz
        assert array.df == 1 * units.Hertz
        assert array.y0 == self.bins[0]
        assert array.dy == self.bins[1] - self.bins[0]

    def test_crop_float_precision_last_value(self):
        """Skip float precision crop tests."""
        pytest.skip("float precision test not supported for SpectralVariance")

    def test_crop_float_precision_last_value_float(self):
        """Skip float precision crop tests."""
        pytest.skip("float precision test not supported for SpectralVariance")

    def test_crop_between_grid_points_is_floored(self):
        """Skip float precision crop tests."""
        pytest.skip("float precision test not supported for SpectralVariance")

    def test_crop_float_precision_near_last_value_float(self):
        """Skip float precision crop tests."""
        pytest.skip("float precision test not supported for SpectralVariance")

    def test_crop_float_precision_first_value_float(self):
        """Skip float precision crop tests."""
        pytest.skip("float precision test not supported for SpectralVariance")

    def test_is_compatible_yindex(self):
        """Skip test for `SpectralVariance.is_compatible` on yindex."""
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")

    def test_is_compatible_error_yindex(self, array):  # noqa: ARG002
        """Skip test for `SpectralVariance.is_compatible` on yindex."""
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")

    def test_plot(self, array):
        """Test `SpectralVariance.plot`."""
        with rc_context(rc={"text.usetex": False}):
            plot = array.plot(yscale="linear")
            assert len(plot.gca().collections) == 1
            plot.save(BytesIO(), format="png")
            plot.close()

    def test_value_at(self):
        """Test `SpectralVariance.value_at`."""
        array = self.create()
        assert array.value_at(
            5,
            self.bins[3],
        ) == self.data[5][3] * array.unit
        assert array.value_at(
            8 * array.xunit,
            self.bins[1] * array.yunit,
        ) == self.data[8][1] * array.unit
        with pytest.raises(IndexError):
            array.value_at(1.6, 5.8)

    @pytest.mark.parametrize("inplace", [True, False])
    def test_inject(self, inplace):  # noqa: ARG002
        """Skip test for `SpectralVariance.inject`."""
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")
