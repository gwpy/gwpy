# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

from scipy import __version__ as scipy_version

from astropy import units

from ... import astro
from ...testing import utils
from ...timeseries import TimeSeries
from ...frequencyseries import FrequencySeries
from ...spectrogram import Spectrogram

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Alex Urban <alexander.urban@ligo.org>'

# -- test results -------------------------------------------------------------

# hack up constants that changed between astropy 1.3 and 2.0
# TODO: might want to do this in reverse, i.e. hard-coding the answers for 2.0
from astropy import __version__ as astropy_version  # nopep8
if astropy_version >= '2.0':
    from astropy import constants
    from astropy.constants import (si, astropyconst13)
    units.M_sun._represents = units.Unit(astropyconst13.M_sun)
    constants.M_sun = si.M_sun = astropyconst13.M_sun
    constants.G = si.G = astropyconst13.G
    constants.c = si.c = astropyconst13.c
    constants.pc = si.pc = astropyconst13.pc

# something changed in scipy 0.19, something FFT-related
if scipy_version < '0.19':
    TEST_RESULTS = {
        'inspiral_range': 19.63704209223392 * units.Mpc,
        'inspiral_range_psd': 7.915847068684727 * units.Mpc ** 2 / units.Hz,
        'burst_range': 13.813232309724613 * units.Mpc,
        'burst_range_spectrum': 35.19303454822539 * units.Mpc,
    }
else:
    TEST_RESULTS = {
        'inspiral_range': 19.63872448570372 * units.Mpc,
        'inspiral_range_psd': 7.92640311063505 * units.Mpc ** 2 / units.Hz,
        'burst_range': 13.815456279746522 * units.Mpc,
        'burst_range_spectrum': 35.216492263916535 * units.Mpc,
    }


# -- utilities ----------------------------------------------------------------

@pytest.fixture(scope='module')
def psd():
    try:
        data = TimeSeries.read(utils.TEST_HDF5_FILE, 'L1:LDAS-STRAIN',
                               format='hdf5')
    except ImportError as e:
        pytest.skip(str(e))
    return data.psd(.4, overlap=.2, window=('kaiser', 24))


@pytest.fixture(scope='module')
def hoft():
    try:
        data = TimeSeries.read(utils.TEST_HDF5_FILE, 'L1:LDAS-STRAIN',
                               format='hdf5')
    except ImportError as e:
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
