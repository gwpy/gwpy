# Copyright (c) 2013-2017 Louisiana State University
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

"""Tests for LAL signal processing interface."""

import pytest
from astropy import units

from ..spectral import _lal as fft_lal

lal = pytest.importorskip("lal")


def test_generate_window():
    """Test :func:`gwpy.signal.spectral.lal.generate_window`."""
    # test default arguments
    w = fft_lal.generate_window(128)
    assert isinstance(w, lal.REAL8Window)
    assert w.data.data.size == 128
    assert w.sum == 32.31817089602309

    # test generating the same window again returns the same object
    assert fft_lal.generate_window(128) is w


def test_generate_window_dtype():
    """Test :func:`gwpy.signal.spectral.lal.generate_window` dtype argument."""
    w = fft_lal.generate_window(128, dtype="float32")
    assert isinstance(w, lal.REAL4Window)
    assert w.sum == 32.31817089602309


@pytest.mark.parametrize(("args", "kwargs", "exc_type"), [
    # invalid window
    ((128,), {"window": "unknown"}, ValueError),
    # unsupported data type
    ((128,), {"dtype": int}, AttributeError),
])
def test_generate_window_error(args, kwargs, exc_type):
    """Test :func:`gwpy.signal.spectral.lal.generate_window` error handling."""
    with pytest.raises(exc_type):
        fft_lal.generate_window(*args, **kwargs)


def test_generate_fft_plan():
    """Test :func:`gwpy.signal.spectral.lal.generate_fft_plan`."""
    # test default arguments
    plan = fft_lal.generate_fft_plan(128)
    assert isinstance(plan, lal.REAL8FFTPlan)
    # test generating the same fft_plan again returns the same object
    assert fft_lal.generate_fft_plan(128) is plan


def test_generate_fft_plan_dtype():
    """Test :func:`gwpy.signal.spectral.lal.generate_fft_plan` dtype."""
    # test dtype works
    plan = fft_lal.generate_fft_plan(128, dtype="float32")
    assert isinstance(plan, lal.REAL4FFTPlan)


def test_generate_fft_plan_forward_backward():
    """Test :func:`gwpy.signal.spectral.lal.generate_fft_plan` forward."""
    # test forward/backward works
    plan = fft_lal.generate_fft_plan(128)
    rvrs = fft_lal.generate_fft_plan(128, forward=False)
    assert isinstance(rvrs, lal.REAL8FFTPlan)
    assert rvrs is not plan


def test_generate_fft_plan_error():
    """Test :func:`gwpy.signal.spectral.lal.generate_fft_plan` error handling."""
    # test errors
    with pytest.raises(AttributeError):
        fft_lal.generate_fft_plan(128, dtype=int)


def test_welch(noisy_sinusoid):
    """Test :func:`gwpy.signal.spectral.lal.welch`."""
    psd = fft_lal.welch(noisy_sinusoid, 4096, noverlap=2048)
    # assert PSD peaks at 500 Hz (as designed)
    assert psd.max() == psd.value_at(500.)
    assert psd.unit == noisy_sinusoid.unit ** 2 / units.Hz
    assert psd.channel is noisy_sinusoid.channel
    assert psd.name is noisy_sinusoid.name


def test_welch_warning(noisy_sinusoid):
    """Test :func:`gwpy.signal.spectral.lal.welch` warnings."""
    # check warning with hanging data
    with pytest.warns(UserWarning):
        fft_lal.welch(noisy_sinusoid, 1000)


def test_median(corrupt_noisy_sinusoid):
    """Test :func:`gwpy.signal.spectral.lal.median`."""
    psd = fft_lal.median(corrupt_noisy_sinusoid, 4096, noverlap=2048)
    assert psd.max() == psd.value_at(500.)
    assert psd.median() < fft_lal.welch(
        corrupt_noisy_sinusoid,
        4096,
        noverlap=2048,
    ).median()


def test_median_mean(corrupt_noisy_sinusoid):
    """Test :func:`gwpy.signal.spectral.lal.median_mean`."""
    psd = fft_lal.median_mean(corrupt_noisy_sinusoid, 8192, noverlap=4096)
    assert psd.max() == psd.value_at(500.)
    assert psd.median() < fft_lal.welch(
        corrupt_noisy_sinusoid,
        4096,
        noverlap=2048,
    ).median()


def test_median_mean_error(corrupt_noisy_sinusoid):
    """Test :func:`gwpy.signal.spectral.lal.median_mean` error handling."""
    # check failure with single segment
    with pytest.raises(ValueError):
        fft_lal.median_mean(
            corrupt_noisy_sinusoid,
            corrupt_noisy_sinusoid.size,
        )
