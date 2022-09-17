# -*- coding: utf-8 -*-
# Copyright (C) Evan Goetz (2021)
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

"""Unit tests for :mod:`gwpy.cli.transferfunction`
"""

from astropy.time import Time

from ... import cli
from .base import _TestCliProduct, _TestTransferFunctionProduct

__author__ = 'Evan Goetz <evan.goetz@ligo.org>'


class TestCliTransferFunction(_TestTransferFunctionProduct):
    TEST_CLASS = cli.TransferFunction
    ACTION = 'transferfunction'
    TEST_ARGS = _TestCliProduct.TEST_ARGS + [
        '--chan', 'Y1:TEST-CHANNEL', '--secpfft', '0.25',
    ]

    def test_init(self, prod):
        assert prod.chan_list == ['X1:TEST-CHANNEL', 'Y1:TEST-CHANNEL']
        assert prod.ref_chan == prod.chan_list[0]
        assert prod.test_chan == prod.chan_list[1]

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == (f'Transfer function: '
                                       f'{prod.chan_list[1]}/'
                                       f'{prod.chan_list[0]}')

    def test_get_title(self, prod):
        epoch = prod.start_list[0]
        utc = Time(epoch, format='gps', scale='utc').iso
        t = ', '.join([
            f'{utc} | {epoch} ({prod.duration})',
            f'fftlength={prod.args.secpfft}',
            f'overlap={prod.args.overlap}',
        ])
        assert prod.get_title() == t

    def test_set_plot_properties(self, prod):
        pass
