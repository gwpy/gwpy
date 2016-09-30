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
import argparse

from numpy import random

from matplotlib import use
use('agg')

from compat import unittest

from gwpy.timeseries import TimeSeries
from gwpy.plotter import rcParams

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_GWF_FILE = os.path.join(os.path.split(__file__)[0], 'data',
                          'HLV-GW100916-968654552-1.gwf')
_, TEMP_PLOT_FILE = tempfile.mkstemp(prefix='GWPY-UNITTEST_', suffix='.png')


class CliTestMixin(object):
    PRODUCT_NAME = 'gwpy.cli.cliproduct.CliProduct'
    ACTION = None
    TEST_ARGS = ['--chan', 'H1:LDAS-STRAIN', '--start', '968654552',
                 '--framecache', TEST_GWF_FILE]

    def setUp(self):
        self.PRODUCT_TYPE = self._import_product()

    def _import_product(self):
        modname, objname = self.PRODUCT_NAME.rsplit('.', 1)
        mod = importlib.import_module(modname)
        return getattr(mod, objname)

    def test_init(self):
        product = self.PRODUCT_TYPE()

    def test_get_action(self):
        self.assertEqual(self.PRODUCT_TYPE().get_action(), self.ACTION)

    def test_init_cli(self):
        parser = argparse.ArgumentParser()
        product = self.PRODUCT_TYPE()
        product.init_cli(parser)
        self.assertGreater(len(parser._actions), 1)
        return product, parser

    def test_get_timeseries(self):
        product, parser = self.test_init_cli()
        args = parser.parse_args(self.TEST_ARGS + ['--out', TEMP_PLOT_FILE])
        try:
            try:
                product.getTimeSeries(args)
            except Exception as e:
                if 'No reader' in str(e):
                    raise RuntimeError(str(e))
                else:
                    raise
        except (RuntimeError, ImportError):
            product.timeseries = []
            product.time_groups = []
            product.start_list = []
            for s in map(int, args.start):
                product.start_list.append(s)
                product.time_groups.append([])
                for c in args.chan:
                    product.timeseries.append(
                        TimeSeries(random.random(1024 * 100), sample_rate=1024,
                                   channel=c, epoch=s))
                    product.time_groups[-1].append(len(product.timeseries)-1)
        return product, args

    def test_gen_plot(self):
        product, args = self.test_get_timeseries()
        product.config_plot(args)
        rcParams.update({'text.usetex': False,})
        product.gen_plot(args)
        return product, args

    def test_config_plot(self):
        product, parser = self.test_init_cli()
        product.config_plot(parser.parse_args(self.TEST_ARGS))

    def test_setup_xaxis(self):
        product, args = self.test_gen_plot()
        product.ax = product.plot.gca()
        product.setup_xaxis(args)

    def test_setup_yaxis(self):
        product, args = self.test_gen_plot()
        product.ax = product.plot.gca()
        product.setup_yaxis(args)

    def test_annotate_save_plot(self):
        product, args = self.test_gen_plot()
        try:
            product.annotate_save_plot(args)
        finally:
            if os.path.isfile(args.out):
                os.remove(args.out)


class CliTimeSeriesTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.timeseries.TimeSeries'
    ACTION = 'timeseries'


class CliSpectrumTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.spectrum.Spectrum'
    ACTION = 'spectrum'


class CliSpectrogramTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.spectrogram.Spectrogram'
    ACTION = 'spectrogram'


class CliCoherenceTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.coherence.Coherence'
    ACTION = 'coherence'
    TEST_ARGS = CliTestMixin.TEST_ARGS + [
        '--chan', 'L1:LDAS-STRAIN', '--secpfft', '0.25',
    ]


class CliCoherencegramTests(CliTestMixin, unittest.TestCase):
    PRODUCT_NAME = 'gwpy.cli.coherencegram.Coherencegram'
    ACTION = 'coherencegram'
    # XXX coherencegram fails to generate using 1-second of input data
    #     which is probably fair enough
    TEST_ARGS = ['--chan', 'X1:TEST-CHANNEL', '--chan', 'Y1:TEST-CHANNEL',
                 '--start', '968654552']


if __name__ == '__main__':
    unittest.main()
