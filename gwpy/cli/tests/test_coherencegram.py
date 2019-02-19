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

"""Unit tests for :mod:`gwpy.cli.coherencegram`
"""

from ... import cli
from ...cli import cliproduct
from .test_spectrogram import TestCliSpectrogram as _TestCliSpectrogram
from .test_coherence import TestCliCoherence as _TestCliCoherence


class TestCliCoherencegram(_TestCliSpectrogram):
    TEST_CLASS = cli.Coherencegram
    ACTION = 'coherencegram'
    TEST_ARGS = _TestCliCoherence.TEST_ARGS

    def test_finalize_arguments(self, prod):
        from gwpy.cli.coherencegram import DEFAULT_CMAP as DEFAULT_COH_CMAP
        if DEFAULT_COH_CMAP is None:
            assert prod.args.cmap == cliproduct.DEFAULT_CMAP.name
        else:
            assert prod.args.cmap == DEFAULT_COH_CMAP.name
        assert prod.args.color_scale == 'linear'
        assert prod.args.imin == 0.
        assert prod.args.imax == 1.

    def test_get_suptitle(self, prod):
        t = 'Coherence spectrogram: {0} vs {1}'.format(*prod.chan_list)
        assert prod.get_suptitle() == t

    def test_init(self, prod):
        assert prod.chan_list == ['X1:TEST-CHANNEL', 'Y1:TEST-CHANNEL']
