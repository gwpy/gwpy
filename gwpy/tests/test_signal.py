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

import pytest

import numpy

from astropy import units

try:
    import lal
except ImportError:
    pass

from gwpy import signal as gwpy_signal
from gwpy.signal import window
from gwpy.signal.fft import (lal as fft_lal, utils as fft_utils,
                             registry as fft_registry, ui as fft_ui)
from gwpy.timeseries import TimeSeries

import utils

ONE_HZ = units.Quantity(1, 'Hz')

NOTCH_60HZ = (
    numpy.asarray([ 0.99973536+0.02300468j,  0.99973536-0.02300468j]),
    numpy.asarray([ 0.99954635-0.02299956j,  0.99954635+0.02299956j]),
    0.99981094420429639,
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
        zpk = gwpy_signal.notch(60, 16384)
        for a, b in zip(zpk, NOTCH_60HZ):
            utils.assert_array_almost_equal(a, b)
        # test Quantities
        zpk2 = gwpy_signal.notch(60 * ONE_HZ, 16384 * ONE_HZ)
        for a, b in zip(zpk, zpk2):
            utils.assert_array_almost_equal(a, b)


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
        assert window.recommended_overlap('ham') == .5
        assert window.recommended_overlap('Hanning') == .5
        assert window.recommended_overlap('bth', nfft=128) == 64
        with pytest.raises(ValueError) as exc:
            window.recommended_overlap('kaiser')
        assert str(exc.value) == ('no recommended overlap for \'kaiser\' '
                                  'window')


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
            pass

        assert fake_caller.__doc__ is None
        fft_registry.update_doc(fake_caller)
        print(fake_caller.__doc__)
        assert fake_caller.__doc__ == (
            'The available methods are:\n\n'
            '============ =================================\n'
            'Method name               Function            \n'
            '============ =================================\n'
            'lal_bartlett    `gwpy.signal.fft.lal.bartlett`\n'
            ' median_mean `gwpy.signal.fft.lal.median_mean`\n'
            '      median      `gwpy.signal.fft.lal.median`\n'
            '   lal_welch       `gwpy.signal.fft.lal.welch`\n'
            '    bartlett  `gwpy.signal.fft.scipy.bartlett`\n'
            '       welch     `gwpy.signal.fft.scipy.welch`\n'
            '============ =================================\n\n'
            'See :ref:`gwpy-signal-fft` for more details\n'
        )


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
        assert ftp == {'nfft': 1024, 'noverlap': 512, 'window': 'hann'}


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
        with pytest.raises(AttributeError):
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
