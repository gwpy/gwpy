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

"""Tests for `gwpy.signal.filter_design`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest import mock

import numpy
import pytest
from astropy import units
from numpy.testing import assert_array_equal
from scipy import signal

from ...testing import utils
from .. import filter_design

if TYPE_CHECKING:
    from ..filter_design import (
        TapsType,
        ZpkCompatible,
        ZpkType,
    )

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

ONE_HZ = units.Quantity(1, "Hz")

# -- filters --------------------------

FILTER_FS = 1024.
FILTER_NYQ = FILTER_FS / 2.
NOTCH_F = 60.
LOWPASS_F = 100.
HIGHPASS_F = 200.
BANDPASS_F = (LOWPASS_F, HIGHPASS_F)


@pytest.fixture(scope="module")
def notch_60() -> ZpkType:
    """60 Hz notch filter fixture."""
    passband = (NOTCH_F - 1) / FILTER_NYQ, (NOTCH_F + 1) / FILTER_NYQ
    stopband = (NOTCH_F - .1) / FILTER_NYQ, (NOTCH_F + .1) / FILTER_NYQ
    return signal.iirdesign(
        passband,
        stopband,
        1,  # max passband loss (dB)
        10,  # min stopband loss (dB)
        analog=False,
        ftype="ellip",
        output="zpk",
    )


@pytest.fixture(scope="module")
def lowpass_100_iir() -> ZpkType:
    """100 Hz lowpass filter fixture."""
    return signal.iirdesign(
        LOWPASS_F / FILTER_NYQ,
        LOWPASS_F * 1.5 / FILTER_NYQ,
        2,
        30,
        analog=False,
        ftype="cheby1",
        output="zpk",
    )


@pytest.fixture(scope="module")
def lowpass_100_fir() -> TapsType:
    """100 Hz lowpass FIR filter fixture."""
    return signal.firwin(
        30,
        LOWPASS_F,
        window="hamming",
        width=50.,
        fs=FILTER_FS,
    )


@pytest.fixture(scope="module")
def highpass_100_iir() -> ZpkType:
    """100 Hz highpass filter fixture."""
    return signal.iirdesign(
        HIGHPASS_F / FILTER_NYQ,
        HIGHPASS_F * 2 / 3. / FILTER_NYQ,
        2,
        30,
        analog=False,
        ftype="cheby1",
        output="zpk",
    )


@pytest.fixture(scope="module")
def highpass_100_fir() -> TapsType:
    """100 Hz highpass FIR filter fixture."""
    return signal.firwin(
        23,
        HIGHPASS_F,
        window="hamming",
        pass_zero=False,
        width=-HIGHPASS_F / 3.,
        fs=FILTER_FS,
    )


@pytest.fixture(scope="module")
def bandpass_100_200_iir() -> ZpkType:
    """100-200 Hz bandpass FIR filter fixture."""
    return signal.iirdesign(
        (LOWPASS_F / FILTER_NYQ, HIGHPASS_F / FILTER_NYQ),
        (LOWPASS_F * 2 / 3. / FILTER_NYQ, HIGHPASS_F * 3 / 2. / FILTER_NYQ),
        2,
        30,
        analog=False,
        ftype="cheby1",
        output="zpk",
    )


@pytest.fixture(scope="module")
def bandpass_100_200_fir() -> TapsType:
    """100-200 Hz bandpass FIR filter fixture."""
    return signal.firwin(
        45,
        BANDPASS_F,
        window="hamming",
        pass_zero=False,
        fs=FILTER_FS,
    )


# -- tests ----------------------------

def test_truncate_transfer():
    """Test `_truncate_transfer()`."""
    series = numpy.ones(64)
    trunc = filter_design._truncate_transfer(series)
    assert trunc[0] == 0
    assert trunc[-1] == 0
    utils.assert_allclose(series[5:59], trunc[5:59])


def test_truncate_impulse():
    """Test `_truncate_impulse()`."""
    series = numpy.ones(64)
    trunc = filter_design._truncate_impulse(series, ntaps=10)
    assert trunc[0] != 0
    assert trunc[-1] != 0
    utils.assert_allclose(trunc[5:59], numpy.zeros(54))


def test_fir_from_transfer():
    """Test `fir_from_transfer()`."""
    frequencies = numpy.arange(0, 64)
    fseries = numpy.cos(2 * numpy.pi * frequencies)

    # prepare the time domain filter
    fir = filter_design.fir_from_transfer(fseries, ntaps=10)

    # test the filter
    assert abs(fir[0] / fir.max()) <= 1e-2
    assert abs(fir[-1] / fir.max()) <= 1e-2
    assert fir.size == 10


def test_notch_iir(notch_60):
    """Test `notch()` with an IIR filter."""
    # test simple notch
    zpk = filter_design.notch(NOTCH_F, FILTER_FS)
    utils.assert_zpk_equal(zpk, notch_60)


def test_notch_iir_quantities(notch_60):
    """Test `notch()` with an IIR filter."""
    # test Quantities
    zpk = filter_design.notch(NOTCH_F * ONE_HZ, FILTER_FS * ONE_HZ)
    utils.assert_zpk_equal(zpk, notch_60)


def test_notch_fir_notimplemented():
    """Test `notch()` with an FIR filter."""
    with pytest.raises(NotImplementedError):
        filter_design.notch(60, 16384, type="fir")  # type: ignore[call-overload]


def test_lowpass_iir(lowpass_100_iir):
    """Test `lowpass()` with an IIR filter."""
    iir = filter_design.lowpass(LOWPASS_F, FILTER_FS)
    utils.assert_zpk_equal(iir, lowpass_100_iir)


def test_lowpass_fir(lowpass_100_fir):
    """Test `lowpass()` with an FIR filter."""
    fir = filter_design.lowpass(LOWPASS_F, FILTER_FS, type="fir")
    utils.assert_allclose(fir, lowpass_100_fir)


def test_highpass_iir(highpass_100_iir):
    """Test `highpass()` with an IIR filter."""
    iir = filter_design.highpass(HIGHPASS_F, FILTER_FS)
    utils.assert_zpk_equal(iir, highpass_100_iir)


def test_highpass_fir(highpass_100_fir):
    """Test `highpass()` with an FIR filter."""
    fir = filter_design.highpass(HIGHPASS_F, FILTER_FS, type="fir")
    utils.assert_allclose(fir, highpass_100_fir)


def test_bandpass_iir(bandpass_100_200_iir):
    """Test `bandpass()` with an IIR filter."""
    iir = filter_design.bandpass(LOWPASS_F, HIGHPASS_F, FILTER_FS)
    utils.assert_zpk_equal(iir, bandpass_100_200_iir)


def test_bandpass_fir(bandpass_100_200_fir):
    """Test `bandpass()` with an FIR filter."""
    fir = filter_design.bandpass(LOWPASS_F, HIGHPASS_F, FILTER_FS, type="fir")
    utils.assert_allclose(fir, bandpass_100_200_fir)


def test_concatenate_zpks():
    """Test `concatenate_zpks()`."""
    asa = numpy.asarray
    z1, p1, k1 = [1, 2, 3], [4, 5, 6], 1.
    z2, p2, k2 = [1, 2, 3, 4], [5, 6, 7, 8], 100
    utils.assert_zpk_equal(
        filter_design.concatenate_zpks((z1, p1, k1), (z2, p2, k2)),
        (asa(z1 + z2), asa(p1 + p2), k1 * k2),
    )


def test_parse_filter_fir():
    """Test `parse_filter()` with an FIR filter."""
    taps = numpy.arange(10)
    assert filter_design.parse_filter(taps) == (
        "ba",
        (taps, [1.]),
    )


def test_parse_filter_ba():
    """Test :func:`parse_filter` with a transfer function filter."""
    in_ = signal.butter(1, 0.5, output="ba")
    typ, out = filter_design.parse_filter(in_)
    assert typ == "ba"
    assert_array_equal(in_[0], out[0])
    assert_array_equal(in_[1], out[1])


def test_parse_filter_zpk():
    """Test `parse_filter()` with a ZPK filter."""
    in_ = signal.butter(1, 0.5, output="zpk")
    typ, out = filter_design.parse_filter(in_)
    assert typ == "zpk"
    utils.assert_zpk_equal(in_, out)


def test_parse_filter_sos():
    """Test `parse_filter()` with a SOS filter."""
    in_ = signal.butter(4, 0.5, output="sos")
    typ, out = filter_design.parse_filter(in_)
    assert typ == "sos"
    assert_array_equal(in_, out)


def test_parse_filter_lti():
    """Test `parse_filter()` with an LTI filter."""
    in_ = signal.lti(*signal.butter(2, 0.5, output="zpk"))
    typ, out = filter_design.parse_filter(in_)
    assert typ == "zpk"
    utils.assert_zpk_equal(
        (in_.zeros, in_.poles, in_.gain),
        out,
    )


@pytest.fixture
def example_zpk_fs_tuple() -> tuple[ZpkCompatible, float]:
    """Return an example ZPK and sample rate."""
    z, p, k = [1], [2], 0.1
    fs = 0.1
    return (z, p, k), fs


def test_convert_to_digital_zpk(example_zpk_fs_tuple):
    """Test `convert_to_digital()` with ZPK input."""
    (z, p, k), fs = example_zpk_fs_tuple

    dform, dfilt = filter_design._convert_to_digital((z, p, k), fs)

    assert dform == "zpk"
    assert dfilt == signal.bilinear_zpk(z, p, k, fs)


def test_convert_to_digital_ba():
    """Test `convert_to_digital()` with BA input."""
    b, a = signal.butter(1, .5, output="ba")

    # this should be converted to ZPK form
    dform, dfilt = filter_design._convert_to_digital((b, a), 16)
    assert dform == "ba"
    dfilt_exp = signal.bilinear(b, a, fs=16)
    assert_array_equal(dfilt[0], dfilt_exp[0])
    assert_array_equal(dfilt[1], dfilt_exp[1])


def test_convert_to_digital_fir():
    """Test `convert_to_digital()` with FIR (numerator only) input."""
    fs = 0.1
    b = numpy.array([1, 0.2, 0.5])
    # this should be converted to ZPK form
    dform, dfilt = filter_design._convert_to_digital(b, fs)
    assert dform == "ba"
    assert numpy.allclose(dfilt, signal.bilinear(b, [1], fs))


def test_convert_to_digital_complex_type_preserved():
    """Test that conversion to digital does not erroneously convert to float types.

    Tests regression against:
    https://gitlab.com/gwpy/gwpy/-/issues/1630#note_2001485594
    """
    z, p, k = signal.butter(3, 30, "low", analog=True, output="zpk")
    form, filt = filter_design._convert_to_digital((z, p, k), 100)
    assert form == "zpk"
    assert p.dtype == filt[1].dtype
    assert numpy.iscomplexobj(filt[1])


@mock.patch(
    "gwpy.signal.filter_design.parse_filter",
    return_value=("invalid", [1, 2, 3]),
)
def test_convert_to_digital_invalid_form(mock_parse_filter):
    """Test that `convert_to_digital()` raises ValueError on invalid filter form."""
    with pytest.raises(
        ValueError,
        match="convert 'invalid'",
    ):
        filter_design._convert_to_digital([1, 2, 3], sample_rate=1)
