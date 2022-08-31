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


@pytest.mark.parametrize(("in_, out"), [
    ("Han", "hann"),
])
def test_canonical_name(in_, out):
    assert window.canonical_name(in_) == out


def test_canonical_name_error():
    with pytest.raises(
        ValueError,
        match="^no window function in scipy.signal equivalent to 'blah'$",
    ):
        window.canonical_name('blah')


@pytest.mark.parametrize(("name", "overlap"), [
    ("hann", 0.5),
    ("Hamming", 0.5),
])
def test_recommended_overlap(name, overlap):
    assert window.recommended_overlap(name) == overlap


def test_recommended_overlap_nfft():
    assert window.recommended_overlap('barthann', nfft=128) == 64


def test_recommended_overlap_error():
    with pytest.raises(
        ValueError,
        match="^no recommended overlap for 'kaiser' window$",
    ):
        window.recommended_overlap('kaiser')


def test_planck():
    series = numpy.ones(64)
    wseries = series * window.planck(64, nleft=5, nright=5)
    assert wseries[0] == 0
    assert wseries[-1] == 0
    utils.assert_allclose(wseries[5:59], series[5:59])
