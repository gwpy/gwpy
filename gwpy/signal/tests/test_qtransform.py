# Copyright (c) 2018-2020 Louisiana State University
#               2018-2025 Cardiff University
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

"""Tests for :mod:`gwpy.signal.qtransform`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy
import pytest
from numpy.random import default_rng
from numpy.testing import (
    assert_allclose,
)
from scipy.signal import gausspulse

from ...table import EventTable
from ...timeseries import TimeSeries
from .. import qtransform

if TYPE_CHECKING:
    from ...spectrogram import Spectrogram
    from ..qtransform import QGram

__author__ = "Alex Urban <alexander.urban@ligo.org>"

RNG = default_rng(seed=0)


# -- Fixtures -----------------------

SEARCH = (-0.25, 0.25)

@pytest.fixture(scope="module")
def noise() -> TimeSeries:
    """Generate a noise TimeSeries for testing."""
    return TimeSeries(
        RNG.normal(size=4096 * 10),
        sample_rate=4096,
        epoch=-5,
    )


@pytest.fixture(scope="module")
def glitch(noise) -> TimeSeries:
    """Generate a glitch TimeSeries for testing."""
    return TimeSeries(
        gausspulse(noise.times.value, fc=500) * 10,
        sample_rate=4096,
        epoch=-0.1,
    )


@pytest.fixture(scope="module")
def data(noise, glitch) -> TimeSeries:
    """Combine noise and glitch TimeSeries for testing."""
    return noise + glitch


@pytest.fixture(scope="module")
def qgram_far(data) -> tuple[QGram, float]:
    """Generate a Q-transform QGram and FAR for testing."""
    return qtransform.q_scan(data, search=SEARCH)


@pytest.fixture(scope="module")
def qgram(qgram_far) -> QGram:
    """Extract QGram from QGram and FAR tuple."""
    return qgram_far[0]


@pytest.fixture(scope="module")
def qfar(qgram_far) -> float:
    """Extract FAR from QGram and FAR tuple."""
    return qgram_far[1]


@pytest.fixture(scope="module")
def qspecgram(qgram) -> Spectrogram:
    """Interpolate a Q-transform QGram to a QSpecGram for testing."""
    return qgram.interpolate()


# -- Tests ---------------------------

def test_q_scan_gaussian(noise):
    """Test `qtransform.q_scan()` with a Gaussian signal."""
    qgram, far = qtransform.q_scan(noise, search=SEARCH)

    # Check FAR
    assert far == pytest.approx(0.029618458114325303)

    # Check that Q-plane frequencies are strictly increasing
    freq = qgram.plane.frequencies
    assert (freq[1:] > freq[:-1]).all()

    # Check peak
    peak = qgram.peak
    assert peak["energy"] == pytest.approx(13.86928)
    assert peak["frequency"] == pytest.approx(710.9)
    assert peak["snr"] == pytest.approx(5.266740798950195)


def test_q_scan_glitch(data):
    """Test `qtransform.q_scan()`."""
    qgram, far = qtransform.q_scan(data, search=SEARCH)

    # test that FAR is better than 1 / Hubble time
    assert far < 1 / (1.37e10 * 365 * 86400)

    # Check peak
    peak = qgram.peak
    assert peak["energy"] == pytest.approx(405.44434)
    assert peak["frequency"] == pytest.approx(505.6)
    assert peak["snr"] == pytest.approx(28.476107)


def test_timeseries_q_transform(data, qspecgram):
    """Test `TimeSeries.q_transform()` against `qtransform.q_scan()`."""
    # scan with the TimeSeries method
    ts_qspecgram = data.q_transform(whiten=False)

    # test spectrogram output
    assert ts_qspecgram.q == qspecgram.q
    assert ts_qspecgram.shape == qspecgram.shape
    assert ts_qspecgram.dtype == numpy.dtype("float32")
    assert_allclose(ts_qspecgram.value, qspecgram.value)


def test_timeseries_q_transform_unnormalised(data, qspecgram):
    """Test `TimeSeries.q_transform()` with norm=False."""
    # scan with norm=False
    ts_qspecgram = data.q_transform(whiten=False, norm=False)

    # test spectrogram output
    assert ts_qspecgram.q == qspecgram.q
    assert ts_qspecgram.dtype == numpy.dtype("float64")


def test_q_scan_fd(data, qfar, qspecgram):
    """Test frequency-domain `qtransform.q_scan()` against time-domain."""
    # create test object from frequency-domain input
    fs = data.fft()
    fs_qgram, far = qtransform.q_scan(
        fs,
        duration=abs(data.span),
        sampling=data.sample_rate.value,
        search=SEARCH,
        epoch=fs.epoch.value,
    )
    fs_qspecgram = fs_qgram.interpolate()

    # test that the output is the same
    assert far == qfar
    assert fs_qspecgram.q == qspecgram.q
    assert fs_qspecgram.dtype == numpy.dtype("float32")
    assert fs_qspecgram.shape == qspecgram.shape
    assert_allclose(fs_qspecgram.value, qspecgram.value, rtol=3e-2)


def test_qtable(qgram):
    """Test Q-transform QGram table output."""
    qtable = qgram.table()

    # Check basic properties
    assert isinstance(qtable, EventTable)
    assert qtable.meta["q"] == qgram.plane.q
    assert qtable["time"].shape == qtable["frequency"].shape

    # Check that peak properties are correctly reported
    imax = qtable["energy"].argmax()
    assert qtable["time"][imax] == qgram.peak["time"]
    assert qtable["duration"][imax] == 1 / 1638.4
    assert qtable["frequency"][imax] == qgram.peak["frequency"]
    assert qtable["bandwidth"][imax] == pytest.approx(
        2
        * numpy.pi ** (1 / 2.)
        * qtable["frequency"][imax]
        / qgram.plane.q,
    )
    assert qtable["energy"][imax] == qgram.peak["energy"]



@pytest.mark.parametrize(("snr", "size"), [
    pytest.param(0, 223232, id="0"),
    pytest.param(10, 37, id="10"),
    pytest.param(1e9, 0, id="high"),
])
def test_qtable_snrthresh(qgram, snr, size):
    """Test Q-transform QGram table output with SNR threshold."""
    # Test that too high an SNR threshold returns an empty table
    assert len(qgram.table(snrthresh=snr)) == size
