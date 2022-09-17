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

"""Unit tests for :mod:`gwpy.cli`
"""

from argparse import ArgumentParser
from unittest import mock

import pytest

from numpy import random

from matplotlib import pyplot

from ...cli import cliproduct
from ...frequencyseries import FrequencySeries
from ...timeseries import TimeSeries
from ...plot import Plot
from ...testing import (utils, mocks)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- utilities ----------------------------------------------------------------

def _random_data(*shape):
    return random.rand(*shape)


def mock_nds2_connection():
    random.seed(0)
    xts = TimeSeries(_random_data(10240), t0=0,
                     sample_rate=1024, name='X1:TEST-CHANNEL')
    random.seed(1)  # use different seed to give coherence something to do
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
        prod = cls.TEST_CLASS(args)
        yield prod
        if prod.plot:
            prod.plot.close()

    @staticmethod
    def _prod_add_data(prod):
        # we need this method separate, rather than in dataprod, so that
        # we can have classes override the dataprod fixture with extra
        # stuff properly
        dur = prod.duration
        fs = 512

        i = 0
        for start in prod.start_list:
            for chan in prod.chan_list:
                random.seed(i)
                ts = TimeSeries(_random_data(int(fs*dur)), t0=start,
                                sample_rate=512, name=chan)
                prod.timeseries.append(ts)
                i += 1
        return prod

    @classmethod
    @pytest.fixture
    def dataprod(cls, prod):
        """Returns a `CliProduct` with data
        """
        return cls._prod_add_data(prod)

    @staticmethod
    def _plotprod_init(prod):
        prod.plot = pyplot.figure(FigureClass=Plot)
        return prod

    @classmethod
    @pytest.fixture
    def plotprod(cls, dataprod):
        return cls._plotprod_init(dataprod)

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

    @pytest.mark.requires("nds2")
    def test_get_data(self, prod):
        conn, data = mock_nds2_connection()
        with mock.patch('nds2.connection') as mocker:
            mocker.return_value = conn
            prod.get_data()

        utils.assert_quantity_sub_equal(
            prod.timeseries[0],
            data[0],
            exclude=("channel",),
        )

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

        if isinstance(plotprod, cliproduct.TransferFunctionProduct):
            data = plotprod.tfs
        elif isinstance(plotprod, cliproduct.FrequencyDomainProduct):
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

    @pytest.mark.requires("nds2")
    def test_run(self, tmp_path, prod):
        conn, _ = mock_nds2_connection()
        tmp = tmp_path / "plot.png"
        with mock.patch('nds2.connection') as mocker:
            mocker.return_value = conn
            prod.args.out = str(tmp)
            prod.run()
            assert tmp.is_file()
            assert prod.plot_num == 1
            assert not prod.has_more_plots()


class _TestImageProduct(_TestCliProduct):
    TEST_CLASS = cliproduct.ImageProduct

    @classmethod
    @pytest.fixture
    def plotprod(cls, dataprod):
        cls._plotprod_init(dataprod)
        dataprod.plot.gca().pcolormesh(dataprod.result)
        return dataprod

    def test_extra_plot_options(self, args):
        for key in ('nocolorbar', 'cmap', 'imin', 'imax'):
            assert hasattr(args, key)

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
        cls._prod_add_data(prod)
        fftlength = prod.args.secpfft
        for i, ts in enumerate(prod.timeseries):
            nsamp = int(fftlength * 512 / 2.) + 1
            random.seed(i)
            fs = FrequencySeries(_random_data(nsamp), x0=0, dx=1/fftlength,
                                 channel=ts.channel, name=ts.name)
            prod.spectra.append(fs)
        return prod


class _TestTransferFunctionProduct(_TestCliProduct):
    @classmethod
    @pytest.fixture
    def dataprod(cls, prod):
        cls._prod_add_data(prod)
        fftlength = prod.args.secpfft
        for i, ts in enumerate(prod.timeseries):
            nsamp = int(fftlength * 512 / 2.) + 1
            random.seed(i)
            if i % 2 == 0:
                name = (f'{prod.timeseries[i+1].name}---'
                        f'{prod.timeseries[i].name}')
                fs = FrequencySeries(
                    _random_data(nsamp) + 1j*_random_data(nsamp),
                    x0=0,
                    dx=1/fftlength,
                    channel=prod.timeseries[i+1].channel,
                    name=f'{name}',
                    dtype=complex,
                )
                prod.test_chan = prod.timeseries[i+1].name
            prod.tfs.append(fs)
        return prod
