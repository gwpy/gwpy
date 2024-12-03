# Copyright (C) Cardiff University (2018-)
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

from collections.abc import Iterator

import numpy
import pytest
from matplotlib import rc_context

from ..plot.tex import has_tex
from ..timeseries import TimeSeries

# -- plotting ------------------------

SKIP_TEX: pytest.MarkDecorator = pytest.mark.skipif(
    not has_tex(),
    reason="TeX is not available",
)


@pytest.fixture(scope="function", params=[
    pytest.param(False, id="no-tex"),
    pytest.param(True, id="usetex", marks=SKIP_TEX)
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

@pytest.fixture
def noisy_sinusoid() -> TimeSeries:
    """10s of 2V/rtHz RMS sine wave at 500Hz with 1mV/rtHz white noise (2kHz).

    See also
    --------
    scipy.signal.welch
    """
    # see :func:`scipy.signal.welch`
    numpy.random.seed(1234)

    # params
    freq = 500.
    rate = 2048.
    size = rate * 10

    # create noisy sinusoid
    amp = 2 * numpy.sqrt(2)  # 2V RMS sinusoid
    noise_power = 1e-3 * rate / 2  # mV RMS white noise
    time = numpy.arange(size) / rate
    x = amp * numpy.sin(2 * numpy.pi * freq * time)
    x += numpy.random.normal(scale=numpy.sqrt(noise_power), size=time.shape)
    return TimeSeries(x, xindex=time, unit='V', name="noisy sinusoid")


@pytest.fixture
def corrupt_noisy_sinusoid(
    noisy_sinusoid: TimeSeries,
) -> TimeSeries:
    """The `noisy_sinusoid` but with 10 samples of corruption in the middle.

    See also
    --------
    scipy.signal.welch
    """
    # add corruption in part of the signal
    size = noisy_sinusoid.size
    noisy_sinusoid.value[int(size // 2):int(size // 2) + 10] *= 50.
    noisy_sinusoid.name = "corrupt noisy sinusoid"
    return noisy_sinusoid
