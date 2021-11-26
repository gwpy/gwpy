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

"""Unit tests for :mod:`gwpy.cli.coherence`
"""

from ... import cli
from .base import _TestCliProduct
from .test_spectrum import TestCliSpectrum as _TestCliSpectrum


class TestCliCoherence(_TestCliSpectrum):
    TEST_CLASS = cli.Coherence
    ACTION = 'coherence'
    TEST_ARGS = _TestCliProduct.TEST_ARGS + [
        '--chan', 'Y1:TEST-CHANNEL', '--secpfft', '0.25',
    ]

    def test_init(self, prod):
        assert prod.chan_list == ['X1:TEST-CHANNEL', 'Y1:TEST-CHANNEL']
        assert prod.ref_chan == prod.chan_list[0]

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == f'Coherence: {prod.chan_list[0]}'
