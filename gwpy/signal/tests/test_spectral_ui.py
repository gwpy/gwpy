# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013-2020)
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

"""Unit test for signal module
"""

import numpy

import pytest

from scipy import signal

from astropy import units

from ...testing.utils import (
    assert_array_equal,
    assert_quantity_sub_equal,
)
from ...timeseries import TimeSeries
from ..spectral import _ui as fft_ui


def test_seconds_to_samples():
    """Test :func:`gwpy.signal.spectral.ui.seconds_to_samples`
    """
    assert fft_ui.seconds_to_samples(4, 256) == 1024
    assert fft_ui.seconds_to_samples(1 * units.minute, 16) == 960
    assert fft_ui.seconds_to_samples(
        4 * units.second, 16.384 * units.kiloHertz) == 65536


def test_normalize_fft_params():
    """Test :func:`gwpy.signal.spectral.ui.normalize_fft_params`
    """
    ftp = fft_ui.normalize_fft_params(
        TimeSeries(numpy.zeros(1024), sample_rate=256))
    assert ftp == {'nfft': 1024, 'noverlap': 0}


def test_normalize_fft_params_window_str():
    ftp = fft_ui.normalize_fft_params(
        TimeSeries(numpy.zeros(1024), sample_rate=256),
        {'window': 'hann'})
    win = signal.get_window('hann', 1024)
    assert ftp.pop('nfft') == 1024
    assert ftp.pop('noverlap') == 512
    assert_array_equal(ftp.pop('window'), win)
    assert not ftp


def test_normalize_fft_params_window_array():
    win = signal.get_window(("kaiser", 14), 1024)
    ftp = fft_ui.normalize_fft_params(
        TimeSeries(numpy.zeros(1024), sample_rate=256),
        {"window": win},
    )
    assert_array_equal(ftp.pop("window"), win)


@pytest.mark.requires("lal")
@pytest.mark.parametrize("win", [
    "hann",
    signal.get_window(("kaiser", 14), 1024),
])
def test_normalize_fft_params_window_lal(win):
    import lal
    from gwpy.signal.spectral._lal import welch
    ftp = fft_ui.normalize_fft_params(
        TimeSeries(numpy.zeros(1024, dtype="float32"), sample_rate=256),
        kwargs={'window': win},
        func=welch,
    )
    assert isinstance(ftp.pop("window"), lal.REAL4Window)


def test_chunk_timeseries():
    """Test :func:`gwpy.signal.spectral.ui._chunk_timeseries`
    """
    a = TimeSeries(numpy.arange(400))
    chunks = list(fft_ui._chunk_timeseries(a, 100, 50))
    for i, (idxa, idxb) in enumerate([
            (None, 150),
            (75, 225),
            (175, 325),
            (250, 400),
    ]):
        assert_quantity_sub_equal(chunks[i], a[idxa:idxb])
