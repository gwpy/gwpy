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

"""Unit tests for :mod:`gwpy.cli.spectrum`
"""

from astropy.time import Time

from ... import cli
from .base import _TestFrequencyDomainProduct


class TestCliSpectrum(_TestFrequencyDomainProduct):
    TEST_CLASS = cli.Spectrum
    ACTION = 'spectrum'

    def test_get_title(self, prod):
        epoch = prod.start_list[0]
        utc = Time(epoch, format='gps', scale='utc').iso
        t = ', '.join([
            '{0} | {1} ({2})'.format(utc, epoch, prod.duration),
            'fftlength={0}'.format(prod.args.secpfft),
            'overlap={0}'.format(prod.args.overlap),
        ])
        assert prod.get_title() == t

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == 'Spectrum: {0}'.format(
            prod.chan_list[0])
