# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2023)
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

"""Custom filtering utilities for the `TimeSeries`
"""

__author__ = "Alex Southgate <alex.southgate@ligo.org>"

import warnings

import numpy as np
import pytest
import scipy.signal as sig

from ...signal import spectral
from ...timeseries import TimeSeries


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
    firstarr += np.random.normal(5.8, 2, n_t)

    secondarr = 0.5 * np.cos(ts + 0.1) + 0.1 * np.sin(5 * ts + 10)
    secondarr += np.random.normal(5.8, 2, n_t)

    return firstarr, secondarr, seglen


@pytest.fixture
def unequal_fs_series_data():
    """Create some fake data with unequal sampling frequencies.

    Returns
    -------
        ts1: array of time points for first data array
        ts2: array of time points for second data array
        firstarr: an array of data, simple mixture of waves
        secondarr: a second array of data from different mixture
        seglen: segment length param to reuse for ffts
        fs_1: sampling frequency 1
        fs_2: sampling frequency 2
    """
    seglen = 512
    n_segs1 = 10
    n_segs2 = 20
    n_t1 = seglen * n_segs1
    n_t2 = seglen * n_segs2
    t_end = 10

    fs_1 = n_t1 / t_end
    fs_2 = n_t2 / t_end

    ts1 = np.linspace(0, t_end, n_t1)
    ts2 = np.linspace(0, t_end, n_t2)

    firstarr = np.sin(2 * np.pi * ts1) + 0.1 * np.sin(2 * np.pi * ts1 * 5)
    secondarr = np.sin(2 * np.pi * ts2)

    return ts1, ts2, firstarr, secondarr, seglen, fs_1, fs_2


def test_coherence_happy(series_data):
    """Test the interface to scipy.signal.coherence.

    For other tests see timeseries/tests/timeseries.py
    """
    firstarr, secondarr, seglen = series_data
    f_s = 0.001

    first = TimeSeries(firstarr, sample_rate=f_s)
    second = TimeSeries(secondarr, sample_rate=f_s)

    coh = spectral.coherence(first, second, segmentlength=seglen)
    ftemp, cxytemp = sig.coherence(firstarr, secondarr, f_s, nperseg=seglen)

    coharr = np.array(coh.data)

    assert all(coharr == cxytemp)


def test_coherence_resample(unequal_fs_series_data):
    """Ensure warning is raised by unequal sampling frequencies.
    """
    ts1, ts2, firstarr, secondarr, seglen, fs_1, fs_2 = unequal_fs_series_data

    # first and second arrays are different, secondarr should have
    # sampling frequency fs_2, but sometimes a mistake is made
    first = TimeSeries(firstarr, sample_rate=fs_1)
    second = TimeSeries(secondarr, sample_rate=fs_1)
    third = TimeSeries(secondarr, sample_rate=fs_2)

    # the first coherence val coh12 is broken intentionally since
    # secondarr data should not have fs_1, instead fs_2
    coh12 = spectral.coherence(first, second, segmentlength=seglen)
    with pytest.warns(
        UserWarning,
        match="Sampling frequencies are unequal",
    ):
        coh13 = spectral.coherence(first, third, segmentlength=seglen)

    # get the frequency at minimum coherence, this should be the extra
    # component in secondarr
    maxi12 = np.argmin(coh12[:50])
    maxf12 = coh12.frequencies[maxi12]
    maxi13 = np.argmin(coh13[:50])
    maxf13 = coh12.frequencies[maxi13]

    # this one is close to 5 -- the extra freq component in secondarr
    assert 4 <= maxf13.value <= 6
    # this one is totally broken
    assert not (4 <= maxf12.value <= 6)


def test_coherence_resample_downsample(series_data):
    """Ensure warning is raised by unequal sampling frequencies.
    """
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
    with pytest.raises(ValueError):
        spectral.coherence(
            first,
            second,
            segmentlength=seglen,
            downsample=False,
        )

    # but that accepting downsampling gives you the same result as
    # doing nothing (but doesn't emit a warning)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        coh2 = spectral.coherence(
            first,
            second,
            segmentlength=seglen,
            downsample=True,
        )

    assert all(np.array(coh1.data) == np.array(coh2.data))
