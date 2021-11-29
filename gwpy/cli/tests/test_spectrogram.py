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

"""Unit tests for :mod:`gwpy.cli.spectrogram`
"""

import pytest

from ... import cli
from .base import (_TestImageProduct, _TestTimeDomainProduct, _TestFFTMixin)


class TestCliSpectrogram(_TestFFTMixin, _TestTimeDomainProduct,
                         _TestImageProduct):
    TEST_CLASS = cli.Spectrogram
    ACTION = 'spectrogram'

    @classmethod
    @pytest.fixture
    def dataprod(cls, prod):
        cls._prod_add_data(prod)
        prod.result = prod.get_spectrogram()
        return prod

    def test_get_title(self, prod):
        assert prod.get_title() == ', '.join([
            f'fftlength={prod.args.secpfft}',
            f'overlap={prod.args.overlap}',
        ])

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == f'Spectrogram: {prod.chan_list[0]}'
