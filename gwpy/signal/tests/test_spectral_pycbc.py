# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Tests for :mod:`gwpy.signal.spectral.pycbc`

Here we check `welch` thoroughly, and the others less so, because
they just call out to that method anyway.
"""

import pytest

from ..spectral import _pycbc as fft_pycbc

pytest.importorskip('pycbc.psd')


def test_welch(noisy_sinusoid):
    psd = fft_pycbc.welch(noisy_sinusoid, 4096, noverlap=2048)
    # assert PSD peaks at 500 Hz (as designed)
    assert psd.max() == psd.value_at(500.)
    # and has a median that matches the RMS withinn 10%
    assert psd.median().value == pytest.approx(1e-3, rel=1e-1)
    # check metadata
    assert psd.unit == noisy_sinusoid.unit ** 2 / 'Hz'
    assert psd.channel is noisy_sinusoid.channel
    assert psd.name is noisy_sinusoid.name


def test_bartlett(noisy_sinusoid):
    psd = fft_pycbc.bartlett(noisy_sinusoid, 4096)
    assert psd.max() == psd.value_at(500.)


def test_median(noisy_sinusoid):
    psd = fft_pycbc.median(noisy_sinusoid, 4096, noverlap=2048)
    assert psd.max() == psd.value_at(500.)
    assert psd.median() < fft_pycbc.welch(noisy_sinusoid, 4096,
                                          noverlap=2048).median()
    assert psd.median().value == pytest.approx(1e-3, rel=1e-1)


def test_median_mean(noisy_sinusoid):
    psd = fft_pycbc.median_mean(noisy_sinusoid, 4096, noverlap=2048)
    assert psd.max() == psd.value_at(500.)
    assert psd.median() < fft_pycbc.welch(noisy_sinusoid, 4096,
                                          noverlap=2048).median()
    assert psd.median().value == pytest.approx(1e-3, rel=1e-1)
