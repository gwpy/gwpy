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

"""Unit tests for :mod:`gwpy.cli.timeseries`
"""

from ... import cli
from .base import (_TestTimeDomainProduct, update_namespace)


class TestCliTimeSeries(_TestTimeDomainProduct):
    TEST_CLASS = cli.TimeSeries
    ACTION = 'timeseries'

    def test_get_title(self, prod):
        update_namespace(prod.args, highpass=10, lowpass=100)
        t = f'Fs: (), duration: {prod.args.duration}, band pass (10.0-100.0)'
        assert prod.get_title() == t

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == f'Time series: {prod.chan_list[0]}'
