# Copyright (c) 2023-2025 Cardiff University
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

"""Custom filtering utilities for the `TimeSeries`."""

__author__ = "Alex Southgate <alex.southgate@ligo.org>"

import warnings

import numpy as np
import pytest
import scipy.signal as sig

from ...signal import spectral
from ...timeseries import TimeSeries

RNG = np.random.default_rng(12345)


@pytest.fixture
def series_data():
    """Create some fake data with equal sampling frequencies.

    Returns
    -------
        firstarr: an array of data, simple mixture of waves
        secondarr: a second array of data from different mixture
        seglen: segment length param to reuse for ffts
    """
    seglen = 512
    n_segs = 10
    n_t = seglen * n_segs
    t_end = 2 * np.pi

    ts = np.linspace(0, t_end, n_t)

    firstarr = 0.1 * np.cos(ts + 0.1) + 0.9 * np.sin(2 * ts + 5)
    firstarr += RNG.normal(5.8, 2, n_t)

    secondarr = 0.5 * np.cos(ts + 0.1) + 0.1 * np.sin(5 * ts + 10)
    secondarr += RNG.normal(5.8, 2, n_t)

    return firstarr, secondarr, seglen


def test_coherence_happy(series_data):
    """Test the interface to scipy.signal.coherence.

    For other tests see timeseries/tests/timeseries.py
    """
    firstarr, secondarr, seglen = series_data
    f_s = 0.001

    first = TimeSeries(firstarr, sample_rate=f_s)
    second = TimeSeries(secondarr, sample_rate=f_s)

    coh = spectral.coherence(first, second, segmentlength=seglen)
    _, cxytemp = sig.coherence(firstarr, secondarr, f_s, nperseg=seglen)

    coharr = np.array(coh.data)

    assert all(coharr == cxytemp)


def test_coherence_resample(series_data):
    """Ensure warning is raised by unequal sampling frequencies."""
    firstarr, secondarr, seglen = series_data
    f_s = 0.001

    first = TimeSeries(firstarr, sample_rate=f_s)
    second = TimeSeries(secondarr, sample_rate=f_s * 2.32)

    with pytest.warns(
        UserWarning,
        match="Sampling frequencies are unequal",
    ):
        coh1 = spectral.coherence(first, second, segmentlength=seglen)

    # check that forcibly disabling downsample results in an error
    with pytest.raises(
        ValueError,
        match="Cannot calculate coherence",
    ):
        spectral.coherence(
            first,
            second,
            segmentlength=seglen,
            downsample=False,
        )

    # but that accepting downsampling gives you the same result as
    # doing nothing (but doesn't emit a warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        coh2 = spectral.coherence(
            first,
            second,
            segmentlength=seglen,
            downsample=True,
        )

    assert all(np.array(coh1.data) == np.array(coh2.data))
