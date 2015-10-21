# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Unit tests for the `gwpy.cli` module
"""

import os
import tempfile
import importlib

from compat import unittest

from gwpy import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class CliTestMixin(object):
    PRODUCT_NAME = 'gwpy.cli.cliproduct.CliProduct'

    def setUp(self):
        self.PRODUCT_TYPE = self._import_product()

    def _import_product(self):
        modname, objname = self.PRODUCT_NAME.rsplit('.', 1)
        mod = importlib.import_module(modname)
        return getattr(mod, objname)

    def test_init(self):
        product = self.PRODUCT_TYPE()


class CliTimeSeriesTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.timeseries.TimeSeries'


class CliSpectrumTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.spectrum.Spectrum'


class CliSpectrogramTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.spectrogram.Spectrogram'


class CliCoherenceTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.coherence.Coherence'


class CliCoherencegramTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.coherencegram.Coherencegram'


if __name__ == '__main__':
    unittest.main()
