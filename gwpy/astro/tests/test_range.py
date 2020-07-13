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

"""Unit tests for :mod:`gwpy.astro.range`
"""

import pytest

from astropy import units

from ... import astro
from ...testing import utils
from ...timeseries import TimeSeries
from ...frequencyseries import FrequencySeries
from ...spectrogram import Spectrogram

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Alex Urban <alexander.urban@ligo.org>'

# -- test results -------------------------------------------------------------

TEST_RESULTS = {
    'inspiral_range': 19.63418297300058 * units.Mpc,
    'inspiral_range_psd': 7.922730174148789 * units.Mpc ** 2 / units.Hz,
    'burst_range': 13.81353542862508 * units.Mpc,
    'burst_range_spectrum': 35.211595883103456 * units.Mpc,
}


# -- utilities ----------------------------------------------------------------

@pytest.fixture(scope='module')
def psd():
    try:
        data = TimeSeries.read(utils.TEST_HDF5_FILE, 'L1:LDAS-STRAIN',
                               format='hdf5')
    except ImportError as e:  # pragma: no-cover
        pytest.skip(str(e))
    return data.psd(.4, overlap=.2, window=('kaiser', 24))


@pytest.fixture(scope='module')
def hoft():
    try:
        data = TimeSeries.read(utils.TEST_HDF5_FILE, 'L1:LDAS-STRAIN',
                               format='hdf5')
    except ImportError as e:  # pragma: no-cover
        pytest.skip(str(e))
    return data


# -- gwpy.astro.range ---------------------------------------------------------

def test_inspiral_range_psd(psd):
    """Test for :func:`gwpy.astro.inspiral_range_psd`
    """
    r = astro.inspiral_range_psd(psd[1:])  # avoid DC
    assert isinstance(r, FrequencySeries)
    utils.assert_quantity_almost_equal(r.max(),
                                       TEST_RESULTS['inspiral_range_psd'])


def test_inspiral_range(psd):
    """Test for :func:`gwpy.astro.inspiral_range_psd`
    """
    r = astro.inspiral_range(psd, fmin=40)
    utils.assert_quantity_almost_equal(r, TEST_RESULTS['inspiral_range'])


def test_burst_range(psd):
    """Test for :func:`gwpy.astro.burst_range`
    """
    r = astro.burst_range(psd.crop(None, 1000)[1:])
    utils.assert_quantity_almost_equal(r, TEST_RESULTS['burst_range'])


def test_burst_range_spectrum(psd):
    """Test for :func:`gwpy.astro.burst_range_spectrum`
    """
    r = astro.burst_range_spectrum(psd.crop(None, 1000)[1:])
    utils.assert_quantity_almost_equal(r.max(),
                                       TEST_RESULTS['burst_range_spectrum'])


@pytest.mark.parametrize('rangekwargs', [
    {'mass1': 1.4, 'mass2': 1.4},
    {'energy': 1e-2},
])
def test_range_timeseries(hoft, rangekwargs):
    trends = astro.range_timeseries(
        hoft, 0.5, fftlength=0.25, overlap=0.125, nproc=2, **rangekwargs)
    assert isinstance(trends, TimeSeries)
    assert trends.size == 2
    assert trends.unit == 'Mpc'
    assert trends.dt == 0.5 * units.second


@pytest.mark.parametrize('rangekwargs, outunit', [
    ({'mass1': 1.4, 'mass2': 1.4}, units.Mpc ** 2 / units.Hz),
    ({'energy': 1e-2}, units.Mpc),
])
def test_range_spectrogram(hoft, rangekwargs, outunit):
    spec = astro.range_spectrogram(
        hoft, 0.5, fftlength=0.25, overlap=0.125, nproc=2, **rangekwargs)
    assert isinstance(spec, Spectrogram)
    assert spec.shape[0] == 2
    assert spec.unit == outunit
    assert spec.f0 == spec.df
    assert spec.dt == 0.5 * units.second
    assert spec.df == 4 * units.Hertz


@pytest.mark.parametrize('range_func', [
    astro.range_timeseries,
    astro.range_spectrogram,
])
def test_range_incompatible_input(range_func):
    with pytest.raises(TypeError) as exc:
        range_func(2, 0.5)
    assert str(exc.value).startswith(
        'Could not produce a spectrogram from the input')
