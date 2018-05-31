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

"""Unit tests for :mod:`gwpy.cli`
"""

import os
import tempfile
import importlib
import argparse
import warnings

import pytest

from numpy import random

from matplotlib import (use, rcParams)
use('agg')  # nopep8

from gwpy.timeseries import TimeSeries

# local imports
from . import (utils, mocks)
from .mocks import mock

warnings.filterwarnings(
    'ignore', category=UserWarning, message=".*non-GUI backend.*")

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

_, TEMP_PLOT_FILE = tempfile.mkstemp(prefix='GWPY-UNITTEST_', suffix='.png')


class CliTestBase(object):
    PRODUCT_NAME = 'gwpy.cli.cliproduct.CliProduct'
    ACTION = None
    TEST_ARGS = ['--chan', 'X1:TEST-CHANNEL', '--start', '1000000000',
                 '--nds2-server', 'nds.test.gwpy']

    @classmethod
    def setup_class(cls):
        cls.PRODUCT_TYPE = cls._import_product()

    @classmethod
    def _import_product(cls):
        modname, objname = cls.PRODUCT_NAME.rsplit('.', 1)
        mod = importlib.import_module(modname)
        return getattr(mod, objname)

    def test_init(self):
        self.PRODUCT_TYPE()

    def test_get_action(self):
        assert self.PRODUCT_TYPE().get_action() == self.ACTION

    def test_init_cli(self):
        parser = argparse.ArgumentParser()
        product = self.PRODUCT_TYPE()
        product.init_cli(parser)
        assert len(parser._actions) > 1

        parser.add_argument('--verbose', default=1, action='count')
        parser.add_argument('--silent', default=False, action='store_true')
        return product, parser

    @utils.skip_missing_dependency('nds2')
    def test_get_timeseries(self):
        try:  # can't use decorator because this method gets called
            import nds2
        except ImportError as e:
            pytest.skip(str(e))
        product, parser = self.test_init_cli()
        args = parser.parse_args(self.TEST_ARGS + ['--out', TEMP_PLOT_FILE])
        product.post_arg(args)

        random.seed(0)
        xts = TimeSeries(random.rand(10240), t0=1000000000,
                         sample_rate=1024, name='X1:TEST-CHANNEL')
        yts = TimeSeries(random.rand(10240), t0=1000000000,
                         sample_rate=1024, name='Y1:TEST-CHANNEL')
        nds_connection = mocks.nds2_connection(buffers=[
            mocks.nds2_buffer_from_timeseries(xts),
            mocks.nds2_buffer_from_timeseries(yts),
        ])
        with mock.patch('nds2.connection') as mock_connection, \
                mock.patch('nds2.buffer'):
            mock_connection.return_value = nds_connection

            product.getTimeSeries(args)

        assert len(product.timeseries) == (len(product.chan_list) *
                                           len(product.start_list))

        return product, args

    def test_gen_plot(self):
        product, args = self.test_get_timeseries()
        product.config_plot(args)
        rcParams.update({'text.usetex': False})
        product.gen_plot(args)
        return product, args

    def test_config_plot(self):
        product, parser = self.test_init_cli()
        product.config_plot(parser.parse_args(self.TEST_ARGS))

    def test_setup_xaxis(self):
        product, args = self.test_gen_plot()
        product.ax = product.plot.gca()
        product.setup_xaxis(args)
    #

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

    @utils.skip_missing_dependency('nds2')
    def test_makePlot(self):
        product, parser = self.test_init_cli()
        args = parser.parse_args(self.TEST_ARGS + ['--out', TEMP_PLOT_FILE])
        product.post_arg(args)
        args.interactive = True

        random.seed(0)
        xts = TimeSeries(random.rand(10240), t0=1000000000,
                         sample_rate=1024, name='X1:TEST-CHANNEL')
        yts = TimeSeries(random.rand(10240), t0=1000000000,
                         sample_rate=1024, name='Y1:TEST-CHANNEL')
        nds_connection = mocks.nds2_connection(buffers=[
            mocks.nds2_buffer_from_timeseries(xts),
            mocks.nds2_buffer_from_timeseries(yts),
        ])

        with mock.patch('nds2.connection') as mock_connection, \
                mock.patch('nds2.buffer'):
            mock_connection.return_value = nds_connection

            try:
                product.makePlot(args, parser)
            finally:
                if os.path.isfile(args.out):
                    os.remove(args.out)

        assert product.is_interactive is True


class TestCliTimeSeries(CliTestBase):
    PRODUCT_NAME = 'gwpy.cli.timeseries.TimeSeries'
    ACTION = 'timeseries'


class TestCliSpectrum(CliTestBase):
    PRODUCT_NAME = 'gwpy.cli.spectrum.Spectrum'
    ACTION = 'spectrum'


class TestCliSpectrogram(CliTestBase):
    PRODUCT_NAME = 'gwpy.cli.spectrogram.Spectrogram'
    ACTION = 'spectrogram'


class TestCliCoherence(CliTestBase):
    PRODUCT_NAME = 'gwpy.cli.coherence.Coherence'
    ACTION = 'coherence'
    TEST_ARGS = CliTestBase.TEST_ARGS + [
        '--chan', 'Y1:TEST-CHANNEL', '--secpfft', '0.25',
    ]


class TestCliCoherencegram(TestCliCoherence):
    PRODUCT_NAME = 'gwpy.cli.coherencegram.Coherencegram'
    ACTION = 'coherencegram'


class TestCliQtransform(CliTestBase):
    PRODUCT_NAME = 'gwpy.cli.qtransform.Qtransform'
    ACTION = 'qtransform'
    TEST_ARGS = [
        '--chan', 'X1:TEST-CHANNEL', '--gps', '1000000005', '--search', '8',
        '--nds2-server', 'nds.test.gwpy', '--outdir', os.path.curdir,
    ]
