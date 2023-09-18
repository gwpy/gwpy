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

"""Unit tests for :mod:`gwpy.signal.filter_design`
"""

from unittest import mock

import numpy

import pytest

from astropy import units

from scipy import signal

from ...testing import utils
from .. import filter_design

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

ONE_HZ = units.Quantity(1, 'Hz')

_nyq = 16384 / 2.
NOTCH_60HZ = signal.iirdesign(
    (59 / _nyq, 61 / _nyq),
    (59.9 / _nyq, 60.1 / _nyq),
    1, 10,
    analog=False, ftype='ellip', output='zpk',
)

_nyq = 1024 / 2.
LOWPASS_IIR_100HZ = signal.iirdesign(
    100 / _nyq,
    150 / _nyq,
    2, 30,
    analog=False, ftype='cheby1', output='zpk',
)
LOWPASS_FIR_100HZ = signal.firwin(
    30, 100, window='hamming', width=50., nyq=512.,
)

HIGHPASS_IIR_100HZ = signal.iirdesign(
    100 / _nyq,
    100 * 2/3. / _nyq,
    2, 30,
    analog=False, ftype='cheby1', output='zpk',
)
HIGHPASS_FIR_100HZ = signal.firwin(
    45, 100, window='hamming', pass_zero=False, width=-100/3., nyq=512.,
)

BANDPASS_IIR_100HZ_200HZ = signal.iirdesign(
    (100 / _nyq, 200 / _nyq),
    (100 * 2/3. / _nyq, 300 / _nyq),
    2, 30,
    analog=False, ftype='cheby1', output='zpk',
)
BANDPASS_FIR_100HZ_200HZ = signal.firwin(
    45, (100, 200.), window='hamming', pass_zero=False, nyq=512.,
)


def test_truncate():
    series = numpy.ones(64)

    # test truncate_transfer
    trunc1 = filter_design.truncate_transfer(series)
    assert trunc1[0] == 0
    assert trunc1[-1] == 0
    utils.assert_allclose(trunc1[5:59], trunc1[5:59])

    # test truncate_impulse
    trunc2 = filter_design.truncate_impulse(series, ntaps=10)
    assert trunc2[0] != 0
    assert trunc2[-1] != 0
    utils.assert_allclose(trunc2[5:59], numpy.zeros(54))


def test_fir_from_transfer():
    frequencies = numpy.arange(0, 64)
    fseries = numpy.cos(2*numpy.pi*frequencies)

    # prepare the time domain filter
    fir = filter_design.fir_from_transfer(fseries, ntaps=10)

    # test the filter
    assert abs(fir[0] / fir.max()) <= 1e-2
    assert abs(fir[-1] / fir.max()) <= 1e-2
    assert fir.size == 10


def test_notch_design():
    # test simple notch
    zpk = filter_design.notch(60, 16384)
    utils.assert_zpk_equal(zpk, NOTCH_60HZ)

    # test Quantities
    zpk2 = filter_design.notch(60 * ONE_HZ, 16384 * ONE_HZ)
    utils.assert_zpk_equal(zpk, zpk2)

    # test FIR notch doesn't work
    with pytest.raises(NotImplementedError):
        filter_design.notch(60, 16384, type='fir')


def test_lowpass():
    iir = filter_design.lowpass(100, 1024)
    utils.assert_zpk_equal(iir, LOWPASS_IIR_100HZ)
    fir = filter_design.lowpass(100, 1024, type='fir')
    utils.assert_allclose(fir, LOWPASS_FIR_100HZ)


def test_highpass():
    iir = filter_design.highpass(100, 1024)
    utils.assert_zpk_equal(iir, HIGHPASS_IIR_100HZ)
    fir = filter_design.highpass(100, 1024, type='fir')
    utils.assert_allclose(fir, HIGHPASS_FIR_100HZ)


def test_bandpass():
    iir = filter_design.bandpass(100, 200, 1024)
    utils.assert_zpk_equal(iir, BANDPASS_IIR_100HZ_200HZ)
    fir = filter_design.bandpass(100, 200, 1024, type='fir')
    utils.assert_allclose(fir, BANDPASS_FIR_100HZ_200HZ)


def test_concatenate_zpks():
    zpk1 = ([1, 2, 3], [4, 5, 6], 1.)
    zpk2 = ([1, 2, 3, 4], [5, 6, 7, 8], 100)
    utils.assert_zpk_equal(
        filter_design.concatenate_zpks(zpk1, zpk2),
        ([1, 2, 3, 1, 2, 3, 4], [4, 5, 6, 5, 6, 7, 8], 100))


def test_parse_filter():
    fir = numpy.arange(10)
    assert filter_design.parse_filter(fir) == ('ba', (fir, [1.]))
    zpk = ([1, 2, 3], [4, 5, 6], 1.)
    parsed = filter_design.parse_filter(zpk)
    assert parsed[0] == 'zpk'
    utils.assert_zpk_equal(parsed[1], zpk)


@pytest.fixture
def example_zpk_fs_tuple():
    z, p, k = [1], [2], 0.1
    fs = 0.1
    return z, p, k, fs


def test_convert_to_digital_zpk(example_zpk_fs_tuple):
    z, p, k, fs = example_zpk_fs_tuple

    dform, dfilt = filter_design.convert_to_digital((z, p, k), fs)

    assert dform == 'zpk'
    assert dfilt == signal.bilinear_zpk(z, p, k, fs)


def test_convert_to_digital_ba(example_zpk_fs_tuple):
    z, p, k, fs = example_zpk_fs_tuple
    b, a = signal.zpk2tf(z, p, k)

    # this should be converted to ZPK form
    dform, dfilt = filter_design.convert_to_digital((b, a), fs)
    assert dform == 'zpk'
    assert dfilt == signal.bilinear_zpk(z, p, k, fs)


def test_convert_to_digital_fir(example_zpk_fs_tuple):
    fs = 0.1
    b = numpy.array([1, 0.2, 0.5])
    # this should be converted to ZPK form
    dform, dfilt = filter_design.convert_to_digital(b, fs)
    assert dform == 'ba'
    assert numpy.allclose(dfilt, signal.bilinear(b, [1], fs))


def test_convert_to_digital_complex_type_preserved():
    """Test that conversion to digital does not erroneously convert
    to float types.

    Tests regression against:
     https://github.com/gwpy/gwpy/issues/1630#issuecomment-1721674653
    """
    z, p, k = signal.butter(3, 30, 'low', analog=True, output='zpk')
    form, filt = filter_design.convert_to_digital((z, p, k), 100)
    zd, pd, kd = filt
    assert p.dtype == pd.dtype
    assert numpy.iscomplexobj(pd)

def test_convert_to_digital_invalid_form():
    with mock.patch('gwpy.signal.filter_design.parse_filter') as tmp_mock:
        tmp_mock.return_value = ("invalid", [1, 2, 3])
        with pytest.raises(ValueError, match='convert invalid'):
            filter_design.convert_to_digital([1, 2, 3], sample_rate=1)


def test_convert_to_digital_fir_still_zpk(example_zpk_fs_tuple):
    """ Test that a filter with poles at zero after bilinear is ZPK. """

    # Why do we always return zpk? We do so for all IIR filters.
    # Only a filter with all poles equal to -2*fs would be a
    # FIR after bilinear transform. However, here, such a filter
    # would be first converted to ZPK form. So we always output ZPK.
    # The "poles" are "implicit", all z=0 (or |z| -> inf),
    # and in some instances are explicitly used by scipy, such as:
    # >>> sig.tf2zpk([1, 0.1, -0.5], [1, 0])
    # >>> (array([-0.75887234,  0.65887234]), array([0.]), 1.0)
    # >>> sig.tf2zpk([1, 0.1, -0.5], [1])
    # >>> (array([-0.75887234,  0.65887234]), array([]), 1.0)

    fs = 0.1
    z = [1, -0.2, 0.3]
    p = [-2 * fs] * len(z)
    k = 0.1

    dform, dfilt = filter_design.convert_to_digital(
        (z, p, k),
        fs,
    )
    assert dform == 'zpk'

    dz, dp, dk = dfilt
    assert numpy.allclose(numpy.array(dp), 0)
