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

"""Unit test for signal module
"""

from importlib import import_module

import pytest

import numpy

from scipy import signal

from astropy import units

try:
    import lal
except ImportError:
    pass

from gwpy.signal import (filter_design, window)
from gwpy.signal.fft import (get_default_fft_api,
                             basic as fft_basic,
                             scipy as fft_scipy,
                             lal as fft_lal,
                             pycbc as fft_pycbc,
                             utils as fft_utils,
                             registry as fft_registry, ui as fft_ui)
from gwpy.timeseries import TimeSeries

import utils

ONE_HZ = units.Quantity(1, 'Hz')

_nyq = 16384 / 2.
NOTCH_60HZ = signal.iirdesign(
    (59 / _nyq, 61 / _nyq),
    (59.9 / _nyq, 60.1 / _nyq),
    1, 10,
    analog=False, ftype='ellip', output='zpk',
)

_nyq = 1024 / 2.
LOWPASS_IIR_100HZ = signal.iirdesign(
    100 / _nyq,
    150 / _nyq,
    2, 30,
    analog=False, ftype='cheby1', output='zpk',
)
LOWPASS_FIR_100HZ = signal.firwin(
    30, 100, window='hamming', width=50., nyq=512.,
)

HIGHPASS_IIR_100HZ = signal.iirdesign(
    100 / _nyq,
    100 * 2/3. / _nyq,
    2, 30,
    analog=False, ftype='cheby1', output='zpk',
)
HIGHPASS_FIR_100HZ = signal.firwin(
    45, 100, window='hamming', pass_zero=False, width=-100/3., nyq=512.,
)

BANDPASS_IIR_100HZ_200HZ = signal.iirdesign(
    (100 / _nyq, 200 / _nyq),
    (100 * 2/3. / _nyq, 300 / _nyq),
    2, 30,
    analog=False, ftype='cheby1', output='zpk',
)
BANDPASS_FIR_100HZ_200HZ = signal.firwin(
    45, (100, 200.), window='hamming', pass_zero=False, nyq=512.,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- gwpy.signal.filter_design ------------------------------------------------

class TestSignalFilterDesign(object):
    """Tests for :mod:`gwpy.signal.filter_design`
    """
    def test_notch_design(self):
        """Test :func:`gwpy.signal.filter_design.notch`
        """
        # test simple notch
        zpk = filter_design.notch(60, 16384)
        utils.assert_zpk_equal(zpk, NOTCH_60HZ)

        # test Quantities
        zpk2 = filter_design.notch(60 * ONE_HZ, 16384 * ONE_HZ)
        utils.assert_zpk_equal(zpk, zpk2)

        # test FIR notch doesn't work
        with pytest.raises(NotImplementedError):
            filter_design.notch(60, 16384, type='fir')

    def test_lowpass(self):
        iir = filter_design.lowpass(100, 1024)
        utils.assert_zpk_equal(iir, LOWPASS_IIR_100HZ)
        fir = filter_design.lowpass(100, 1024, type='fir')
        utils.assert_allclose(fir, LOWPASS_FIR_100HZ)

    def test_highpass(self):
        iir = filter_design.highpass(100, 1024)
        utils.assert_zpk_equal(iir, HIGHPASS_IIR_100HZ)
        fir = filter_design.highpass(100, 1024, type='fir')
        utils.assert_allclose(fir, HIGHPASS_FIR_100HZ)

    def test_bandpass(self):
        iir = filter_design.bandpass(100, 200, 1024)
        utils.assert_zpk_equal(iir, BANDPASS_IIR_100HZ_200HZ)
        fir = filter_design.bandpass(100, 200, 1024, type='fir')
        utils.assert_allclose(fir, BANDPASS_FIR_100HZ_200HZ)

    def test_concatenate_zpks(self):
        zpk1 = ([1, 2, 3], [4, 5, 6], 1.)
        zpk2 = ([1, 2, 3, 4], [5, 6, 7, 8], 100)
        utils.assert_zpk_equal(
            filter_design.concatenate_zpks(zpk1, zpk2),
            ([1, 2, 3, 1, 2, 3, 4], [4, 5, 6, 5, 6, 7, 8], 100))

    def test_parse_filter(self):
        fir = numpy.arange(10)
        assert filter_design.parse_filter(fir) == ('ba', (fir, [1.]))
        zpk = ([1, 2, 3], [4, 5, 6], 1.)
        parsed = filter_design.parse_filter(zpk)
        assert parsed[0] == 'zpk'
        utils.assert_zpk_equal(parsed[1], zpk)


# -- gwpy.signal.window -------------------------------------------------------

class TestSignalWindow(object):
    """Tests for :mod:`gwpy.signal.window`
    """
    def test_canonical_name(self):
        """Test :func:`gwpy.signal.window.canonical_name`
        """
        assert window.canonical_name('Hanning') == 'hann'
        with pytest.raises(ValueError) as exc:
            window.canonical_name('blah')
        assert str(exc.value) == ('no window function in scipy.signal '
                                  'equivalent to \'blah\'')

    def test_recommended_overlap(self):
        """Test :func:`gwpy.signal.window.recommended_overlap`
        """
        assert window.recommended_overlap('hann') == .5
        assert window.recommended_overlap('Hamming') == .5
        assert window.recommended_overlap('barthann', nfft=128) == 64
        with pytest.raises(ValueError) as exc:
            window.recommended_overlap('kaiser')
        assert str(exc.value) == ('no recommended overlap for \'kaiser\' '
                                  'window')


# -- gwpy.signal.fft ----------------------------------------------------------

class TestSignalFft(object):

    def test_get_default_fft_api(self):
        api = get_default_fft_api()
        for lib in ('pycbc.psd', 'lal', 'scipy'):
            try:
                import_module(lib)
            except ImportError:
                continue
            assert api == lib
            return


# -- gwpy.signal.fft.registry -------------------------------------------------

class TestSignalFftRegistry(object):
    """Tests for :mod:`gwpy.signal.fft.registry`
    """
    @staticmethod
    def teardown():
        # remove test methods from registry
        # otherwise they will impact other tests, and test ordering
        # is annoying to predict
        for scaling in fft_registry.METHODS:
            fft_registry.METHODS[scaling].pop('fake_method', '')

    def test_registry(self):
        """Test :mod:`gwpy.signal.fft.registry`
        """
        def fake_method():
            pass

        # test register
        fft_registry.register_method(fake_method)
        assert 'fake_method' in fft_registry.METHODS['density']
        with pytest.raises(KeyError):
            fft_registry.register_method(fake_method)
        fft_registry.register_method(fake_method, force=True)
        assert 'fake_method' in fft_registry.METHODS['density']
        fft_registry.register_method(fake_method, scaling='spectrum')
        assert 'fake_method' in fft_registry.METHODS['spectrum']
        with pytest.raises(KeyError):
            fft_registry.register_method(fake_method, scaling='unknown')
        # test get
        f = fft_registry.get_method('fake_method')
        assert f is fake_method
        with pytest.raises(KeyError):
            fft_registry.get_method('unregistered')
        with pytest.raises(KeyError):
            fft_registry.get_method('fake_method', scaling='unknown')

    def test_update_doc(self):
        """Test :func:`gwpy.signal.fft.registry.update_doc`
        """
        def fake_caller():
            """Test method

            Notes
            -----"""
            pass

        # update docs
        fft_registry.update_doc(fake_caller)

        # simple tests
        doc = fake_caller.__doc__
        assert '            The available methods are:' in doc
        assert 'scipy_welch `gwpy.signal.fft.scipy.welch`' in doc

    @pytest.mark.parametrize('library', ['basic', 'pycbc', 'lal', 'scipy'])
    def test_register_library(self, library):
        apilib = import_module('gwpy.signal.fft.{}'.format(library))
        regname = str if library == 'basic' else ('%s_{}' % library).format
        for method in ('welch', 'bartlett', 'median', 'median_mean'):
            if method == 'median' and library == 'scipy':
                break
            assert (
                fft_registry.get_method(regname(method)) is
                getattr(apilib, method))


# -- gwpy.signal.fft.ui -------------------------------------------------------

class TestSignalFftUI(object):
    def test_seconds_to_samples(self):
        """Test :func:`gwpy.signal.fft.ui.seconds_to_samples`
        """
        assert fft_ui.seconds_to_samples(4, 256) == 1024
        assert fft_ui.seconds_to_samples(1 * units.minute, 16) == 960
        assert fft_ui.seconds_to_samples(
            4 * units.second, 16.384 * units.kiloHertz) == 65536

    def test_normalize_fft_params(self):
        """Test :func:`gwpy.signal.fft.ui.normalize_fft_params`
        """
        ftp = fft_ui.normalize_fft_params(
            TimeSeries(numpy.zeros(1024), sample_rate=256))
        assert ftp == {'nfft': 1024, 'noverlap': 0}
        ftp = fft_ui.normalize_fft_params(
            TimeSeries(numpy.zeros(1024), sample_rate=256),
            {'window': 'hann'})
        win = signal.get_window('hann', 1024)
        assert ftp.pop('nfft') == 1024
        assert ftp.pop('noverlap') == 512
        utils.assert_array_equal(ftp.pop('window'), win)
        assert not ftp

    def test_chunk_timeseries(self):
        """Test :func:`gwpy.signal.fft.ui._chunk_timeseries`
        """
        a = TimeSeries(numpy.arange(400))
        chunks = list(fft_ui._chunk_timeseries(a, 100, 50))
        assert chunks == [
            a[:150], a[75:225], a[175:325], a[275:400],
        ]

    def test_fft_library(self):
        """Test :func:`gwpy.signal.fft.ui._fft_library`
        """
        assert fft_ui._fft_library(fft_lal.welch) == 'lal'
        assert fft_ui._fft_library(fft_scipy.welch) == 'scipy'
        assert fft_ui._fft_library(fft_pycbc.welch) == 'pycbc'
        assert fft_ui._fft_library(fft_basic.welch) == (
            get_default_fft_api().split('.', 1)[0])


# -- gwpy.signal.fft.utils ----------------------------------------------------

class TestSignalFftUtils(object):
    def test_scale_timeseries_unit(self):
        """Test :func:`gwpy.signal.fft.utils.scale_timeseries_units`
        """
        scale_ = fft_utils.scale_timeseries_unit
        u = units.Unit('m')
        # check default
        assert scale_(u) == units.Unit('m^2/Hz')
        # check scaling='density'
        assert scale_(u, scaling='density') == units.Unit('m^2/Hz')
        # check scaling='spectrum'
        assert scale_(u, scaling='spectrum') == units.Unit('m^2')
        # check anything else raises an exception
        with pytest.raises(ValueError):
            scale_(u, scaling='other')
        # check null unit
        assert scale_(None) == units.Unit('Hz^-1')


# -- gwpy.signal.fft.basic ----------------------------------------------------

class TestSignalFftBasic(object):
    def test_map_fft_method(self):
        """Test :func:`gwpy.signal.fft.basic.map_fft_method`
        """
        # check that defining a new method that doesn't map to a
        # library method raises an exception
        @fft_basic.map_fft_method
        def blah(*args, **kwargs):
            pass

        with pytest.raises(RuntimeError) as exc:
            blah()
        assert str(exc.value).startswith('no underlying API method')

        # check that if we only have scipy, we get the same error for
        # median-mean
        if get_default_fft_api() == 'scipy':
            with pytest.raises(RuntimeError):
                fft_basic.median_mean(None, None)


# -- gwpy.signal.fft.lal ------------------------------------------------------

@utils.skip_missing_dependency('lal')
class TestSignalFftLal(object):
    def test_generate_window(self):
        """Test :func:`gwpy.signal.fft.lal.generate_window`
        """
        # test default arguments
        w = fft_lal.generate_window(128)
        assert isinstance(w, lal.REAL8Window)
        assert w.data.data.size == 128
        assert w.sum == 32.31817089602309
        # test generating the same window again returns the same object
        assert fft_lal.generate_window(128) is w
        # test dtype works
        w = fft_lal.generate_window(128, dtype='float32')
        assert isinstance(w, lal.REAL4Window)
        assert w.sum == 32.31817089602309
        # test errors
        with pytest.raises(ValueError):
            fft_lal.generate_window(128, 'unknown')
        with pytest.raises(AttributeError):
            fft_lal.generate_window(128, dtype=int)

    def test_generate_fft_plan(self):
        """Test :func:`gwpy.signal.fft.lal.generate_fft_plan`
        """
        # test default arguments
        plan = fft_lal.generate_fft_plan(128)
        assert isinstance(plan, lal.REAL8FFTPlan)
        # test generating the same fft_plan again returns the same object
        assert fft_lal.generate_fft_plan(128) is plan
        # test dtype works
        plan = fft_lal.generate_fft_plan(128, dtype='float32')
        assert isinstance(plan, lal.REAL4FFTPlan)
        # test forward/backward works
        rvrs = fft_lal.generate_fft_plan(128, forward=False)
        assert isinstance(rvrs, lal.REAL8FFTPlan)
        assert rvrs is not plan
        # test errors
        with pytest.raises(AttributeError):
            fft_lal.generate_fft_plan(128, dtype=int)
