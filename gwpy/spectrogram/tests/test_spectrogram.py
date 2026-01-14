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

"""Unit tests for :mod:`gwpy.spectrogram.spectrogram`."""

from __future__ import annotations

from io import BytesIO
from typing import (
    TYPE_CHECKING,
    TypeVar,
)

import numpy
import pytest
from astropy import units
from matplotlib import rc_context
from scipy import signal

from ...testing import utils
from ...types.tests.test_array2d import TestArray2D as _TestArray2D
from .. import Spectrogram

if TYPE_CHECKING:
    from ...signal.filter_design import ZpkType

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

SpectrogramType = TypeVar("SpectrogramType", bound=Spectrogram)


class TestSpectrogram(_TestArray2D[Spectrogram]):
    """Tests of `gwpy.spectrogram.Spectrogram`."""

    TEST_CLASS = Spectrogram

    def test_new(self):
        """Test creating new `Spectrogram` objects."""
        super().test_new()

        # check handling of epoch vs t0
        a = self.create(epoch=10)
        b = self.create(t0=10)
        utils.assert_quantity_sub_equal(a, b)

    def test_new_redundant_args(self):
        """Test creating new `Spectrogram` objects with redundant args."""
        with pytest.raises(
            ValueError,
            match=r"^give only one of epoch or t0$",
        ):
            self.TEST_CLASS(self.data, epoch=1, t0=1)

    def test_new_times(self):
        """Test creating new `Spectrogram` objects with ``times`` argument."""
        times = numpy.arange(self.data.shape[0])
        a = self.create(times=times)
        utils.assert_quantity_equal(a.times, times * units.second)

    def test_epoch(self, array):
        """Test `Spectrogram.epoch` property."""
        assert array.epoch.gps == array.x0.value

    def test_value_at(self):
        """Test `Spectrogram.value_at` method."""
        super().test_value_at()
        array = self.create()
        assert array.value_at(
            5000 * units.millisecond,
            2000 * units.milliHertz,
        ) == self.data[5][2] * array.unit

    @pytest.mark.parametrize("ratio", ["mean", "median"])
    def test_ratio(self, array: SpectrogramType, ratio):
        """Test `Spectrogram.ratio` method."""
        rat = array.ratio(ratio)
        array_meth = getattr(array, ratio)
        utils.assert_quantity_sub_equal(rat, array / array_meth(axis=0))

    def test_ratio_invalid_operand(self, array: SpectrogramType):
        """Test `Spectrogram.ratio` method with invalid input."""
        with pytest.raises(
            ValueError,
            match="operand 'blah' unrecognised",
        ):
            array.ratio("blah")

    @pytest.fixture
    def min_max_mean(self, array: SpectrogramType):
        """Return min, max, mean `Spectrogram` objects for testing."""
        min_ = array.min(axis=0)
        max_ = array.max(axis=0)
        mean = array.mean(axis=0)
        return min_, max_, mean

    def test_from_spectra(self, min_max_mean):
        """Test `Spectrogram.from_spectra` class method."""
        min_, max_, mean = min_max_mean

        # check basic stack works
        new = self.TEST_CLASS.from_spectra(mean, min_, max_, dt=1)
        assert new.shape == (3, min_.size)
        assert new.name == mean.name
        assert new.epoch == mean.epoch
        assert new.f0 == mean.f0
        assert new.df == mean.df
        assert new.unit == mean.unit
        assert new.dt == 1 * units.second
        utils.assert_array_equal(
            new.value,
            numpy.vstack((mean.value, min_.value, max_.value)),
        )

        # check kwargs
        new = self.TEST_ARRAY.from_spectra(
            mean,
            min_,
            max_,
            dt=2,
            epoch=0,
            f0=100,
            df=.5,
            unit="meter",
            name="test",
        )
        assert new.name == "test"
        assert new.epoch.gps == 0
        assert new.f0 == 100 * units.Hertz
        assert new.df == 0.5 * units.Hertz
        assert new.unit == units.meter

    def test_from_spectra_error_dt(self):
        """Test `Spectrogram.from_spectra` dt error handling."""
        fs = self.TEST_CLASS._columnclass(self.data[0], epoch=0, f0=0, df=1)

        # Check that dt is required with a single input
        with pytest.raises(
            ValueError,
            match="cannot determine dt",
        ):
            self.TEST_ARRAY.from_spectra(fs)
        self.TEST_ARRAY.from_spectra(fs, dt=1)

    def test_from_spectra_error_shape(self):
        """Test `Spectrogram.from_spectra` shape error handling."""
        fs = self.TEST_CLASS._columnclass(self.data[0], epoch=0, f0=0, df=1)

        with pytest.raises(
            ValueError,
            match="all the input array dimensions",
        ):
            self.TEST_CLASS.from_spectra(fs, fs[1:])
        with pytest.raises(
            ValueError,
            match="all the input array dimensions",
        ):
            self.TEST_CLASS.from_spectra(fs, fs[::2])

    def test_crop_frequencies(self, array):
        """Test `Spectrogram.crop_frequencies` method."""
        # test simple
        array2 = array.crop_frequencies()
        utils.assert_quantity_sub_equal(array, array2)
        assert numpy.may_share_memory(array.value, array2.value)

    def test_crop_frequencies_normal(self, array):
        """Test `Spectrogram.crop_frequencies` normal cropping."""
        array2 = array.crop_frequencies(2, 5)
        utils.assert_array_equal(array2.value, array.value[:, 2:5])
        assert array2.f0 == 2 * units.Hertz
        assert array2.df == array.df

    def test_crop_frequencies_copy(self, array: SpectrogramType):
        """Test `Spectrogram.crop_frequencies` copy option."""
        # test copy
        array2 = array.crop_frequencies(copy=True)
        assert not numpy.may_share_memory(array.value, array2.value)

    def crop_frequencies_warnings(self, array):
        """Test `Spectrogram.crop_frequencies` warnings."""
        # test warnings
        with pytest.warns(
            UserWarning,
            match="FIXME",
        ):
            array.crop_frequencies(array.yspan[0]-1, array.yspan[1])
        with pytest.warns(
            UserWarning,
            match="FIXME",
        ):
            array.crop_frequencies(array.yspan[0], array.yspan[1]+1)

    @pytest.mark.parametrize("method", ["imshow", "pcolormesh"])
    def test_plot(self, array, method):
        """Test `Spectrogram.plot` method."""
        with rc_context(rc={"text.usetex": False}):
            plot = array.plot(method=method)
            ax = plot.gca()
            assert len(ax.lines) == 0
            if method == "imshow":
                assert len(ax.images) == 1
            else:
                assert len(ax.collections) == 1
            assert ax.get_epoch() == array.x0.value
            assert ax.get_xlim() == array.xspan
            assert ax.get_ylim() == array.yspan
            plot.save(BytesIO(), format="png")
            plot.close()

    @pytest.fixture
    def zpk(self) -> ZpkType:
        """Return a sample zero-pole-gain filter."""
        z = numpy.asarray([0j, -1j, 1j])
        p = numpy.asarray([-0.5 + 0.5j, -0.5 - 0.5j, -2])
        k = 1
        return z, p, k

    def test_zpk(self, array, zpk):
        """Test `Spectrogram.zpk` method."""
        utils.assert_quantity_sub_equal(
            array.zpk(*zpk),
            array.filter(zpk, analog=False, unit="rad/s"),
        )

    def test_filter(self, array, zpk):
        """Test `Spectrogram.filter` method."""
        # build filter - convert to rad/s since freqresp expects rad/s
        # and filter() defaults to rad/s
        lti = signal.lti(*zpk)
        # freqresp expects omega (rad/s), convert Hz to rad/s
        omega = array.frequencies.value * 2 * numpy.pi
        fresp = numpy.nan_to_num(abs(lti.freqresp(w=omega)[1]))

        # test simple filter
        a2 = array.filter(zpk, analog=True)
        utils.assert_quantity_sub_equal(array * fresp, a2)

    def test_filter_inplace(self, array, zpk):
        """Test `Spectrogram.filter(..., inplace=True)`."""
        a2 = array.filter(zpk, inplace=False, analog=True)
        array.filter(zpk, inplace=True, analog=True)
        utils.assert_quantity_sub_equal(array, a2)

    def test_filter_bad_kwargs(self, array, zpk):
        """Test `Spectrogram.filter` method."""
        # test errors
        with pytest.raises(
            TypeError,
            match=r"got an unexpected keyword argument 'blah'",
        ):
            array.filter(zpk, blah=1)

    def test_read_write_hdf5(self):
        """Test HDF5 read/write of `Spectrogram` objects."""
        array = self.create(name="X1:TEST")
        utils.test_read_write(
            array,
            "hdf5",
            write_kw={"overwrite": True},
        )

    def test_percentile(self):
        """Test `Spectrogram.percentile` method."""
        array = self.create(name="Test", unit="m")
        a2 = array.percentile(50)
        utils.assert_quantity_sub_equal(
            array.median(axis=0),
            a2,
            exclude=("name",),
        )
        assert a2.name == "Test: 50th percentile"
        assert a2.unit == array.unit
