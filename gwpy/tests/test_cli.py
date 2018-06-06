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
import warnings
from argparse import ArgumentParser

import pytest

from numpy import random

from matplotlib import use
use('agg')  # nopep8
from matplotlib import pyplot

from astropy.time import Time

from gwpy import cli as gcli
from gwpy.cli import cliproduct
from gwpy.frequencyseries import FrequencySeries
from gwpy.timeseries import TimeSeries
from gwpy.plotter import Plot

# local imports
from . import (utils, mocks)
from .mocks import mock

warnings.filterwarnings(
    'ignore', category=UserWarning, message=".*non-GUI backend.*")

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

_, TEMP_PLOT_FILE = tempfile.mkstemp(prefix='GWPY-UNITTEST_', suffix='.png')


# -- utilities ----------------------------------------------------------------


def _random_data(*shape):
    random.seed(0)
    return random.rand(*shape)


def mock_nds2_connection():
    xts = TimeSeries(_random_data(10240), t0=0,
                     sample_rate=1024, name='X1:TEST-CHANNEL')
    yts = TimeSeries(_random_data(10240), t0=0,
                     sample_rate=1024, name='Y1:TEST-CHANNEL')
    data = [xts, yts]
    conn = mocks.nds2_connection(
        buffers=list(map(mocks.nds2_buffer_from_timeseries, data)),
    )
    return conn, data


def update_namespace(args, **params):
    for key in params:
        setattr(args, key, params[key])
    return args


# -- function tests -----------------------------------------------------------

def test_to_float():
    to_s = cliproduct.to_float('s')
    s = to_s(4)
    assert isinstance(s, float)
    assert s == 4.
    assert to_s('4ms') == 0.004


def test_unique():
    a = [1, 2, 4, 3, 5, 4, 5, 3]
    assert cliproduct.unique(a) == [1, 2, 4, 3, 5]


# -- class tests --------------------------------------------------------------

class _TestCliProduct(object):
    ACTION = None
    TEST_CLASS = cliproduct.CliProduct
    TEST_ARGS = [
        '--chan', 'X1:TEST-CHANNEL',
        '--start', 0,
        '--nds2-server', 'nds.test.gwpy',
        '--dpi', 100,
        '--geometry', '640x480',
    ]

    # -- fixtures -------------------------------

    @classmethod
    @pytest.fixture
    def args(cls):
        """Creates and parser arguments for a given `CliProduct`

        Returns the `argparse.Namespace`
        """
        parser = ArgumentParser()
        parser.add_argument('--verbose', action='count', default=1)
        parser.add_argument('--silent', action='store_true')
        cls.TEST_CLASS.init_cli(parser)
        return parser.parse_args(map(str, cls.TEST_ARGS))

    @classmethod
    @pytest.fixture
    def prod(cls, args):
        """Returns a `CliProduct`
        """
        return cls.TEST_CLASS(args)

    @classmethod
    @pytest.fixture
    def dataprod(cls, prod):
        """Returns a `CliProduct` with data
        """
        dur = prod.duration
        fs = 512

        for start in prod.start_list:
            for chan in prod.chan_list:
                ts = TimeSeries(_random_data(int(fs*dur)), t0=start,
                                sample_rate=512, name=chan)
                prod.timeseries.append(ts)
        return prod

    @classmethod
    @pytest.fixture
    def plotprod(cls, dataprod):
        dataprod.plot = pyplot.figure(FigureClass=Plot)
        return dataprod

    # -- tests ----------------------------------

    def test_init(self, args):
        prod = self.TEST_CLASS(args)
        assert prod.verbose == args.verbose
        assert prod.plot is None
        assert prod.plot_num == 0
        assert prod.start_list == [0]
        assert prod.duration == 10.
        assert prod.chan_list == ['X1:TEST-CHANNEL']
        assert prod.n_datasets == len(prod.start_list) * len(prod.chan_list)
        assert prod.timeseries == []
        assert prod.dpi == 100.
        assert prod.width == 640
        assert prod.height == 480
        assert prod.figsize == (6.4, 4.8)

    def test_action(self, prod):
        if self.ACTION is None:
            raise NotImplementedError
        assert prod.action is self.ACTION

    @pytest.mark.parametrize('level', (1, 2, 3))
    def test_log(self, prod, level, capsys):
        v = prod.verbose
        prod.log(level, 'Test')
        out, err = capsys.readouterr()
        if v >= level:
            assert out == 'Test\n'
        else:
            assert out == ''

    @utils.skip_missing_dependency('nds2')
    def test_get_data(self, prod):
        conn, data = mock_nds2_connection()
        with mock.patch('nds2.connection') as mocker:
            mocker.return_value = conn
            prod.get_data()

        assert prod.timeseries[0] == data[0]

    @pytest.mark.parametrize('ftype, filt', [
        ('highpass', {'highpass': 10}),
        ('lowpass', {'lowpass': 100}),
        ('bandpass', {'highpass': 10, 'lowpass': 100}),
    ])
    def test_filter_timeseries(self, dataprod, ftype, filt):
        ts = dataprod.timeseries[0]
        if ftype == 'bandpass':
            result = ts.bandpass(filt['highpass'], filt['lowpass'])
        else:
            result = getattr(ts, ftype)(filt[ftype])

        fts = dataprod._filter_timeseries(ts, **filt)
        utils.assert_quantity_sub_equal(fts, result)

    @pytest.mark.parametrize('params, title', [
        ({'highpass': 100}, 'high pass (100.0)'),
        ({'lowpass': 100}, 'low pass (100.0)'),
        ({'highpass': 100, 'lowpass': 200}, 'band pass (100.0-200.)'),
        ({'highpass': 100, 'notch': [60]}, 'high pass (100.0), notch(60.0)'),
    ])
    def test_get_title(self, prod, params, title):
        update_namespace(prod.args, **params)  # update parameters
        assert prod.get_title() == title

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == prod.chan_list[0]

    @pytest.mark.parametrize('params', [
        {},
        {'xscale': 'linear', 'xmin': 0, 'xmax': 5, 'xlabel': 'X-label',
         'yscale': 'log', 'ymin': 0, 'ymax': 50, 'ylabel': 'Y-label'},
    ])
    def test_set_plot_properties(self, plotprod, params):
        args = plotprod.args
        update_namespace(args, **params)  # update parameters

        if isinstance(plotprod, cliproduct.FrequencyDomainProduct):
            data = plotprod.spectra
        else:
            data = plotprod.timeseries
        xmin = min(series.xspan[0] for series in data)
        xmax = max(series.xspan[1] for series in data)

        plotprod.set_plot_properties()
        ax = plotprod.ax

        ymin, ymax = ax.get_ylim()

        assert ax.get_xscale() == params.get('xscale', args.xscale)
        assert ax.get_xlim() == (params.get('xmin', xmin),
                                 params.get('xmax', xmax))
        assert ax.get_xlabel() == params.get('xlabel', plotprod.get_xlabel())

        assert ax.get_yscale() == params.get('yscale', args.yscale or 'linear')
        assert ax.get_ylim() == (params.get('ymin', ymin),
                                 params.get('ymax', ymax))
        assert ax.get_ylabel() == params.get('ylabel', plotprod.get_ylabel())

    @utils.skip_missing_dependency('nds2')
    def test_run(self, prod):
        conn, _ = mock_nds2_connection()
        with mock.patch('nds2.connection') as mocker, \
                tempfile.NamedTemporaryFile(suffix='.png') as f:
            mocker.return_value = conn
            prod.args.out = f.name
            prod.run()
            assert os.path.isfile(f.name)
            assert prod.plot_num == 1
            assert not prod.has_more_plots()


class _TestImageProduct(_TestCliProduct):
    TEST_CLASS = cliproduct.ImageProduct

    @classmethod
    @pytest.fixture
    def plotprod(cls, dataprod):
        super(_TestImageProduct, cls).plotprod(dataprod)
        ax = dataprod.plot.gca(projection='timeseries')
        ax.plot_spectrogram(dataprod.result)
        return dataprod

    def test_extra_plot_options(self, args):
        for key in ('nocolorbar', 'cmap', 'imin', 'imax'):
            assert hasattr(args, key)

    def test_finalize_arguments(self, prod):
        # finalize_arguments() called by __init__
        assert prod.args.cmap == cliproduct.DEFAULT_CMAP.name

    @pytest.mark.parametrize('visible', [False, True])
    def test_set_plot_properties(self, plotprod, visible):
        update_namespace(plotprod.args, nocolorbar=not visible)
        plotprod.set_plot_properties()
        if visible:
            label = plotprod.get_color_label()
            assert plotprod.plot.colorbars[0].ax.get_ylabel() == label
        else:
            assert not plotprod.plot.colorbars


class _TestFFTMixin(object):
    pass


class _TestTimeDomainProduct(_TestCliProduct):
    pass


class _TestFrequencyDomainProduct(_TestCliProduct):
    @classmethod
    @pytest.fixture
    def dataprod(cls, prod):
        super(_TestFrequencyDomainProduct, cls).dataprod(prod)
        fftlength = prod.args.secpfft
        for ts in prod.timeseries:
            nsamp = int(fftlength * 512 / 2.) + 1
            fs = FrequencySeries(_random_data(nsamp), x0=0, dx=1/fftlength,
                                 channel=ts.channel, name=ts.name)
            prod.spectra.append(fs)
        return prod


class TestCliTimeSeries(_TestTimeDomainProduct):
    TEST_CLASS = gcli.TimeSeries
    ACTION = 'timeseries'

    def test_get_title(self, prod):
        update_namespace(prod.args, highpass=10, lowpass=100)
        t = 'Fs: (), duration: {0}, band pass (10.0-100.0)'.format(
            prod.args.duration)
        assert prod.get_title() == t

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == 'Time series: {0}'.format(
            prod.chan_list[0])


class TestCliSpectrum(_TestFrequencyDomainProduct):
    TEST_CLASS = gcli.Spectrum
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


class TestCliSpectrogram(_TestFFTMixin, _TestTimeDomainProduct,
                         _TestImageProduct):
    TEST_CLASS = gcli.Spectrogram
    ACTION = 'spectrogram'

    @classmethod
    @pytest.fixture
    def dataprod(cls, prod):
        super(TestCliSpectrogram, cls).dataprod(prod)
        prod.result = prod.get_spectrogram()
        return prod

    def test_get_title(self, prod):
        assert prod.get_title() == ', '.join([
            'fftlength={0}'.format(prod.args.secpfft),
            'overlap={0}'.format(prod.args.overlap),
        ])

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == 'Spectrogram: {0}'.format(
            prod.chan_list[0])


class TestCliCoherence(TestCliSpectrum):
    TEST_CLASS = gcli.Coherence
    ACTION = 'coherence'
    TEST_ARGS = _TestCliProduct.TEST_ARGS + [
        '--chan', 'Y1:TEST-CHANNEL', '--secpfft', '0.25',
    ]

    def test_init(self, prod):
        assert prod.chan_list == ['X1:TEST-CHANNEL', 'Y1:TEST-CHANNEL']
        assert prod.ref_chan == prod.chan_list[0]

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == 'Coherence: {0}'.format(
            prod.chan_list[0])


class TestCliCoherencegram(TestCliSpectrogram):
    TEST_CLASS = gcli.Coherencegram
    ACTION = 'coherencegram'
    TEST_ARGS = TestCliCoherence.TEST_ARGS

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


class TestCliQtransform(TestCliSpectrogram):
    TEST_CLASS = gcli.Qtransform
    ACTION = 'qtransform'
    TEST_ARGS = [
        '--chan', 'X1:TEST-CHANNEL', '--gps', '5', '--search', '10',
        '--nds2-server', 'nds.test.gwpy', '--outdir', os.path.curdir,
    ]

    def test_finalize_arguments(self, prod):
        assert prod.start_list == [prod.args.gps - prod.args.search/2.]
        assert prod.duration == prod.args.search
        assert prod.args.color_scale == 'linear'
        assert prod.args.xmin is None
        assert prod.args.xmax is None

    def test_init(self, args):
        update_namespace(args, qrange=(100., 110.))
        prod = self.TEST_CLASS(args)
        assert prod.qxfrm_args['gps'] == prod.args.gps
        assert prod.qxfrm_args['qrange'] == (100., 110.)

    def test_get_title(self, dataprod):
        t = ('Q=45.25, whitened, calc f-range=[36.01, 161.51], '
             'calc e-range=[-6.31, 18.98]')
        assert dataprod.get_title() == t

    def test_get_suptitle(self, prod):
        assert prod.get_suptitle() == 'Q-transform: {0}'.format(
            prod.chan_list[0])

    @utils.skip_missing_dependency('nds2')
    def test_run(self, prod):
        conn, _ = mock_nds2_connection()
        outf = 'X1-TEST_CHANNEL-5.0-0.5.png'
        with mock.patch('nds2.connection') as mocker:
            mocker.return_value = conn
            try:
                prod.run()
                assert os.path.isfile(outf)
            finally:
                if os.path.isfile(outf):
                    os.remove(outf)
            assert prod.plot_num == 1
            assert not prod.has_more_plots()
