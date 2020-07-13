# -*- coding: utf-8 -*-
# Copyright (C) Alex Urban (2018-2020)
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

"""Unit tests for :mod:`gwpy.signal.qtransform`
"""

import numpy
from numpy import testing as nptest
from scipy.signal import gausspulse

from .. import qtransform
from ...table import EventTable
from ...segments import Segment
from ...timeseries import TimeSeries

__author__ = 'Alex Urban <alexander.urban@ligo.org>'


# -- global variables ---------------------------------------------------------

# create noise and a glitch template at 1000 Hz
NOISE = TimeSeries(
    numpy.random.normal(size=4096 * 10), sample_rate=4096, epoch=-5)
GLITCH = TimeSeries(
    gausspulse(NOISE.times.value, fc=500)*10, sample_rate=4096)
DATA = NOISE + GLITCH

# global test objects
SEARCH = Segment(-0.25, 0.25)
QGRAM, FAR = qtransform.q_scan(DATA, search=SEARCH)
QSPECGRAM = QGRAM.interpolate()


# -- test utilities -----------------------------------------------------------

def test_far():
    # test that FAR is better than 1 / Hubble time
    assert FAR < 1 / (1.37e10 * 365 * 86400)


def test_monotonicity():
    # test that Q-plane frequencies are strictly increasing
    freq = QGRAM.plane.frequencies
    assert (freq[1:] > freq[:-1]).all()


def test_q_scan():
    # scan with the TimeSeries method
    ts_qspecgram = DATA.q_transform(whiten=False)

    # test spectrogram output
    assert ts_qspecgram.q == QSPECGRAM.q
    assert ts_qspecgram.shape == QSPECGRAM.shape
    assert ts_qspecgram.dtype == numpy.dtype('float32')
    nptest.assert_allclose(ts_qspecgram.value, QSPECGRAM.value)


def test_unnormalised_q_scan():
    # scan with norm=False
    ts_qspecgram = DATA.q_transform(whiten=False, norm=False)

    # test spectrogram output
    assert ts_qspecgram.q == QSPECGRAM.q
    assert ts_qspecgram.dtype == numpy.dtype('float64')


def test_q_scan_fd():
    # create test object from frequency-domain input
    fdata = DATA.fft()
    fs_qgram, far = qtransform.q_scan(
        fdata, duration=abs(DATA.span), sampling=DATA.sample_rate.value,
        search=SEARCH, epoch=fdata.epoch.value)
    fs_qspecgram = fs_qgram.interpolate()

    # test that the output is the same
    assert far == FAR
    assert fs_qspecgram.q == QSPECGRAM.q
    assert fs_qspecgram.dtype == numpy.dtype('float32')
    assert fs_qspecgram.shape == QSPECGRAM.shape
    nptest.assert_allclose(fs_qspecgram.value, QSPECGRAM.value, rtol=3e-2)


def test_qtable():
    # test EventTable output
    qtable = QGRAM.table()
    imax = qtable['energy'].argmax()
    assert isinstance(qtable, EventTable)
    assert qtable.meta['q'] == QGRAM.plane.q
    nptest.assert_almost_equal(qtable['time'][imax], QGRAM.peak['time'])
    nptest.assert_almost_equal(qtable['duration'][imax], 1/1638.4)
    nptest.assert_almost_equal(qtable['frequency'][imax],
                               QGRAM.peak['frequency'])
    nptest.assert_almost_equal(
        qtable['bandwidth'][imax],
        2 * numpy.pi ** (1/2.) * qtable['frequency'][imax] / QGRAM.plane.q)
    nptest.assert_almost_equal(qtable['energy'][imax], QGRAM.peak['energy'])

    # it's enough to check consistency between the shape of time and
    # frequency columns, because of the way they're calculated
    assert qtable['time'].shape == qtable['frequency'].shape

    # test that too high an SNR threshold returns an empty table
    assert len(QGRAM.table(snrthresh=1e9)) == 0
