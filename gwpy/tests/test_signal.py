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
    numpy.asarray([0.99973535728792018+0.023004681879874127j,
                   0.99973535728792018-0.023004681879874127j]),
    numpy.asarray([0.99954635152445503-0.02299955570751059j,
                   0.99954635152445503+0.02299955570751059j]),
    0.99981094420429639,
)

LOWPASS_IIR_100HZ = (
    numpy.asarray([-1., -1., -1., -1., -1.]),
    numpy.asarray([0.79454998691808587+0.54184012440654583j,
                   0.83426821016564101+0.33107822350129129j,
                   0.87062783772611552+0j,
                   0.83426821016564101-0.33107822350129129j,
                   0.79454998691808587-0.54184012440654583j]),
    0.00018609967633116873,
)
LOWPASS_FIR_100HZ = numpy.asarray([
    0.0051744427146426849, 0.011232973712355994, 0.014490805709441739,
    0.012217513054165994, 0.0032895089810864677, -0.010748702513754024,
    -0.025474365116972923, -0.0345008948382233, -0.031372166848766113,
    -0.011873159836616267, 0.024051734689792249, 0.071534148822646818,
    0.12158364858558927, 0.16333161634903515, 0.18706289653557631,
    0.18706289653557631, 0.16333161634903515, 0.12158364858558927,
    0.071534148822646818, 0.024051734689792249, -0.011873159836616267,
    -0.031372166848766113, -0.0345008948382233, -0.025474365116972923,
    -0.010748702513754024, 0.0032895089810864677, 0.012217513054165994,
    0.014490805709441739, 0.011232973712355994, 0.0051744427146426849,
])

HIGHPASS_IIR_100HZ = (
    numpy.asarray([1., 1., 1., 1., 1.]),
    numpy.asarray([0.77771199811485181-0.56312862007490694j,
                   0.48361722536584878-0.6296353904448081j,
                   -0.18405708293888945+0j,
                   0.48361722536584878+0.6296353904448081j,
                   0.77771199811485181+0.56312862007490694j]),
    0.23031747786582513,
)
HIGHPASS_FIR_100HZ = numpy.asarray([
    -0.011787020858963437, -0.0048224630081723863, 0.0046859006581788168,
    0.013395609038641215, 0.017914430795942599, 0.016044695543909224,
    0.0077218015972893364, -0.0047157691553883999, -0.017086790204990016,
    -0.024647743663705372, -0.023727256706292138, -0.013196036922850287,
    0.0047371734073483581, 0.024735247428180736, 0.039580649196175911,
    0.04216448228370806, 0.027662895267402281, -0.0047500438569569054,
    -0.051203252199233426, -0.10371793059057478, -0.15198811030862544,
    -0.18589889130003465, 0.81616147030504715, -0.18589889130003465,
    -0.15198811030862544, -0.10371793059057478, -0.051203252199233426,
    -0.0047500438569569054, 0.027662895267402281, 0.04216448228370806,
    0.039580649196175911, 0.024735247428180736, 0.0047371734073483581,
    -0.013196036922850287, -0.023727256706292138, -0.024647743663705372,
    -0.017086790204990016, -0.0047157691553883999, 0.0077218015972893364,
    0.016044695543909224, 0.017914430795942599, 0.013395609038641215,
    0.0046859006581788168, -0.0048224630081723863, -0.011787020858963437,
])

BANDPASS_IIR_100HZ_200HZ = (
    numpy.asarray([1.+0.j, 1.+0.j,
                   1.+0.j, 1.+0.j,
                   -1.+0.j, -1.+0.j,
                   -1.+0.j, -1.+0.j]),
    numpy.asarray([0.79377356559644074-0.56946307660055717j,
                   0.67472242570127505-0.64218823896503052j,
                   0.67472242570127505+0.64218823896503052j,
                   0.79377356559644074+0.56946307660055717j,
                   0.33806768604370602+0.90190565001193446j,
                   0.485565803803577+0.77721050960919891j,
                   0.485565803803577-0.77721050960919891j,
                   0.33806768604370602-0.90190565001193446j]),
    0.00120035017968342,
)
BANDPASS_FIR_100HZ_200HZ = numpy.asarray([
    0.00017735000656136912, 0.0003607829651727832, -0.00041505363991587655,
    -0.00036879605008279009, 0.0024294085039510346, 0.0062777373511662639,
    0.0051592816116371298, -0.0039840786461563023, -0.013726673774661396,
    -0.012373516499560946, -0.00063293616134394326, 0.0055068851554455926,
    -0.0027574153134290701, -0.0072993625227341643, 0.017340191625737911,
    0.059162768865338396, 0.06209502522018371, -0.01240712848181694,
    -0.11874810490371347, -0.14978752521144881, -0.047813477255423419,
    0.11548530159729188, 0.1946602447884126, 0.11548530159729188,
    -0.047813477255423419, -0.14978752521144881, -0.11874810490371347,
    -0.01240712848181694, 0.06209502522018371, 0.059162768865338382,
    0.017340191625737907, -0.0072993625227341626, -0.0027574153134290701,
    0.0055068851554455926, -0.00063293616134394326, -0.012373516499560949,
    -0.013726673774661401, -0.0039840786461563041, 0.0051592816116371263,
    0.0062777373511662604, 0.0024294085039510329, -0.00036879605008279009,
    -0.00041505363991587655, 0.0003607829651727832, 0.00017735000656136912,
])

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
        utils.assert_zpk_equal(zpk, NOTCH_60HZ)

        # test Quantities
        zpk2 = gwpy_signal.notch(60 * ONE_HZ, 16384 * ONE_HZ)
        utils.assert_zpk_equal(zpk, zpk2)

        # test FIR notch doesn't work
        with pytest.raises(NotImplementedError):
            gwpy_signal.notch(60, 16384, type='fir')

    def test_lowpass(self):
        iir = gwpy_signal.lowpass(100, 1024)
        utils.assert_zpk_equal(iir, LOWPASS_IIR_100HZ)
        fir = gwpy_signal.lowpass(100, 1024, type='fir')
        utils.assert_allclose(fir, LOWPASS_FIR_100HZ)

    def test_highpass(self):
        iir = gwpy_signal.highpass(100, 1024)
        utils.assert_zpk_equal(iir, HIGHPASS_IIR_100HZ)
        fir = gwpy_signal.highpass(100, 1024, type='fir')
        print(fir)
        print(HIGHPASS_FIR_100HZ)
        print(fir - HIGHPASS_FIR_100HZ)
        utils.assert_allclose(fir, HIGHPASS_FIR_100HZ)

    def test_bandpass(self):
        iir = gwpy_signal.bandpass(100, 200, 1024)
        utils.assert_zpk_equal(iir, BANDPASS_IIR_100HZ_200HZ)
        fir = gwpy_signal.bandpass(100, 200, 1024, type='fir')
        utils.assert_allclose(fir, BANDPASS_FIR_100HZ_200HZ)

    def test_concatenate_zpks(self):
        zpk1 = ([1, 2, 3], [4, 5, 6], 1.)
        zpk2 = ([1, 2, 3, 4], [5, 6, 7, 8], 100)
        utils.assert_zpk_equal(
            gwpy_signal.concatenate_zpks(zpk1, zpk2),
            ([1, 2, 3, 1, 2, 3, 4], [4, 5, 6, 5, 6, 7, 8], 100))


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
