# Copyright (c) 2018-2025 Cardiff University
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

"""Custom pytest fixtures for GWpy.

This module is imported in gwpy.conftest such that all fixtures declared
here are available to test functions/methods by default.

Developer note: **none of the fixtures here should declare autouse=True**.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING
from unittest import mock

import numpy
import pytest
from matplotlib import rc_context

from ..plot.tex import has_tex
from ..timeseries import TimeSeries
from . import mocks

if TYPE_CHECKING:
    from collections.abc import Iterator

    import nds2

# -- plotting ------------------------

SKIP_TEX: pytest.MarkDecorator = pytest.mark.skipif(
    not has_tex(),
    reason="TeX is not available",
)


@pytest.fixture(params=[
    pytest.param(False, id="no-tex"),
    pytest.param(True, id="usetex", marks=SKIP_TEX),
])
def usetex(
    request: pytest.FixtureRequest,
) -> Iterator[bool]:
    """Repeat a test with ``text.usetex`` `False` and `True`.

    If TeX is not available on the test machine (determined by
    :func:`gwpy.plot.tex.has_tex`), the ``usetex=True`` tests will be skipped.
    """
    use_ = request.param
    with rc_context(rc={"text.usetex": use_}):
        yield use_


# -- various useful mock data series -

_FIXTURE_TS_SAMPLE_RATE = 2048.
_FIXTURE_TS_DURATION = 10.  # seconds
_FIXTURE_TS_NSAMPLES = int(_FIXTURE_TS_SAMPLE_RATE * _FIXTURE_TS_DURATION)

@pytest.fixture
def white_noise() -> TimeSeries:
    """10s of white noise, sampled at 2kHz, starting at t0=0s."""
    rng = numpy.random.default_rng(seed=0)
    noise = rng.normal(size=_FIXTURE_TS_NSAMPLES)
    return TimeSeries(
        noise / numpy.std(noise),  # Scale to 1V/rtHz
        dt=1 / _FIXTURE_TS_SAMPLE_RATE,
        t0=0,
        name="noise",
    )

@pytest.fixture
def colored_noise(white_noise: TimeSeries) -> TimeSeries:
    """10s of 1/f^2 coloured noise, sampled at 2kHz, starting at t0=0s."""
    freqs = numpy.fft.rfftfreq(_FIXTURE_TS_NSAMPLES, 1/_FIXTURE_TS_SAMPLE_RATE)

    # Create coloured noise
    noise_fft = numpy.fft.rfft(white_noise)
    coloring_filter = numpy.ones_like(freqs)
    coloring_filter[1:] = 1 / (freqs[1:]**2)
    coloring_filter[0] = coloring_filter[1]  # Handle DC
    colored_noise = numpy.fft.irfft(noise_fft * coloring_filter, n=_FIXTURE_TS_NSAMPLES)
    return TimeSeries(
        colored_noise / numpy.std(colored_noise) * 5.0,  # Scale noise
        dt=1 / _FIXTURE_TS_SAMPLE_RATE,
        t0=0,
        name="noise",
    )


@pytest.fixture
def gausspulse() -> TimeSeries:
    """0.25s Gaussian pulse at 200 Hz, centred on t0=0."""
    from scipy.signal import gausspulse
    dur = 0.25
    nsamp = int(dur * _FIXTURE_TS_SAMPLE_RATE)
    times = numpy.linspace(-dur/2, dur/2, nsamp, endpoint=False)
    pulse = gausspulse(times, fc=200, bw=.5)
    return TimeSeries(
        pulse,
        times=times,
        name="gausspulse",
    )


@pytest.fixture
def noisy_sinusoid() -> TimeSeries:
    """10s of 2V/rtHz RMS sine wave at 500Hz with 1mV/rtHz white noise (2kHz)."""
    # see :func:`scipy.signal.welch`
    rng = numpy.random.default_rng(seed=1234)

    # params
    freq = 500.
    rate = 2048.
    size = rate * 10

    # create noisy sinusoid
    amp = 2 * numpy.sqrt(2)  # 2V RMS sinusoid
    noise_power = 1e-3 * rate / 2  # mV RMS white noise
    time = numpy.arange(size) / rate
    x = amp * numpy.sin(2 * numpy.pi * freq * time)
    x += rng.normal(scale=numpy.sqrt(noise_power), size=time.shape)
    return TimeSeries(
        x,
        times=time,
        unit="V",
        name="noisy sinusoid",
    )


@pytest.fixture
def corrupt_noisy_sinusoid(noisy_sinusoid: TimeSeries) -> TimeSeries:
    """Return the `noisy_sinusoid` with 10 samples of corruption in the middle."""
    # add corruption in part of the signal
    size = noisy_sinusoid.size
    noisy_sinusoid.value[int(size // 2):int(size // 2) + 10] *= 50.
    noisy_sinusoid.name = "corrupt noisy sinusoid"
    return noisy_sinusoid


# -- NDS2 fixtures -------------------

@pytest.fixture
def nds2_connection(request: pytest.FixtureRequest) -> Iterator[nds2.connection]:
    """Yield an open mocked NDS2 connection.

    Set ``NDS2_CONNECTION_FIXTURE_DATA`` as an iterable of `TimeSeries` in
    the test class or module to customise the data buffers available.

    This fixture also mocks out `os.environ` and ``pop``s any GWDataFind
    server variables, in an attempt to ensure that NDS2 is used by
    `TimeSeries.get` and friends.
    """
    rng = numpy.random.default_rng(seed=0)
    data = None
    for scope in (request.instance, request.module):
        if data := getattr(scope, "NDS2_CONNECTION_FIXTURE_DATA", None):
            break
    else:
        data = [
            TimeSeries(
                rng.random(size=128),
                name="X1:test",
                t0=1000000000,
                sample_rate=16,
                unit="m",
            ),
            TimeSeries(
                rng.random(size=1024),
                name="Y1:test",
                t0=1000000000,
                sample_rate=128,
                unit="V",
            ),
        ]
    nds2conn = mocks.nds2_connection(
        buffers=map(mocks.nds2_buffer_from_timeseries, data),
    )
    mock_environ = mock.patch.dict("os.environ")
    with mock.patch("nds2.connection") as mockconn, mock_environ:
        os.environ.pop("GWDATAFIND_SERVER", "")
        os.environ.pop("LIGO_DATAFIND_SERVER", "")
        mockconn.return_value = nds2conn
        yield nds2conn
