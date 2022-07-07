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

"""Unit tests for :mod:`gwpy.signal.window`
"""

import numpy
import pytest

from ...testing import utils
from .. import window

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_canonical_name():
    assert window.canonical_name('Han') == 'hann'
    with pytest.raises(ValueError) as exc:
        window.canonical_name('blah')
    assert str(exc.value) == ('no window function in scipy.signal '
                              'equivalent to \'blah\'')


def test_recommended_overlap():
    assert window.recommended_overlap('hann') == .5
    assert window.recommended_overlap('Hamming') == .5
    assert window.recommended_overlap('barthann', nfft=128) == 64
    with pytest.raises(ValueError) as exc:
        window.recommended_overlap('kaiser')
    assert str(exc.value) == ('no recommended overlap for \'kaiser\' '
                              'window')


def test_planck():
    series = numpy.ones(64)
    wseries = series * window.planck(64, nleft=5, nright=5)
    assert wseries[0] == 0
    assert wseries[-1] == 0
    utils.assert_allclose(wseries[5:59], series[5:59])
