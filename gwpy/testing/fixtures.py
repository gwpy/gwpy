# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2020)
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

"""Custom pytest fixtures for GWpy

This module is imported in gwpy.conftest such that all fixtures declared
here are available to test functions/methods by default.

Developer note: **none of the fixtures here should declare autouse=True**.
"""

import pytest

import numpy

from matplotlib import rc_context

from ..plot.tex import HAS_TEX
from ..timeseries import TimeSeries
from ..utils.decorators import deprecated_function
from .utils import TemporaryFilename


# -- I/O ---------------------------------------------------------------------

@pytest.fixture
@deprecated_function
def tmpfile():
    """Return a temporary filename using `tempfile.mktemp`.

    The fixture **does not create the named file**, but will delete it
    when the test exists if it was created in the mean time.
    """
    with TemporaryFilename() as tmp:
        yield tmp


# -- plotting -----------------------------------------------------------------

def _test_usetex():
    """Return `True` if we can render figures using `text.usetex`.
    """
    from matplotlib import pyplot
    with rc_context(rc={'text.usetex': True}):
        fig = pyplot.figure()
        fig.gca()
        try:
            fig.canvas.draw()
        except RuntimeError:
            return False
        else:
            return True
        finally:
            pyplot.close(fig)


SKIP_TEX = pytest.mark.skipif(
    not HAS_TEX or not _test_usetex(),
    reason='TeX is not available',
)


@pytest.fixture(scope='function', params=[
    pytest.param(False, id='no-tex'),
    pytest.param(True, id='usetex', marks=SKIP_TEX)
])
def usetex(request):
    """Repeat a test with matplotlib's `text.usetex` param False and True.

    If TeX is not available on the test machine (determined by
    `gwpy.plot.tex.has_tex()`), the usetex=True tests will be skipped.
    """
    use_ = request.param
    with rc_context(rc={'text.usetex': use_}):
        yield use_


# -- various useful mock data series -----------------------------------------

@pytest.fixture
def noisy_sinusoid():
    """10s of 2V/rtHz RMS sine wave at 500Hz with 1mV/rtHz white noise (2kHz)

    See :func:`scipy.signal.welch`
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
def corrupt_noisy_sinusoid(noisy_sinusoid):
    """The `noisy_sinusoid` but with 10 samples of corruption in the middle

    See :func:`scipy.signal.welch`
    """
    # add corruption in part of the signal
    size = noisy_sinusoid.size
    noisy_sinusoid.value[int(size//2):int(size//2)+10] *= 50.
    noisy_sinusoid.name = "corrupt noisy sinusoid"
    return noisy_sinusoid
