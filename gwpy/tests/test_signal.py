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

from compat import (unittest, HAS_LAL)

import numpy
from numpy import testing as nptest

from scipy import signal

from astropy import units

if HAS_LAL:
    import lal

from gwpy import signal as gwpy_signal
from gwpy.signal.fft import (lal as fft_lal, utils as fft_utils,
                             registry as fft_registry)

ONE_HZ = units.Quantity(1, 'Hz')

NOTCH_60HZ = (
    numpy.asarray([ 0.99973536+0.02300468j,  0.99973536-0.02300468j]),
    numpy.asarray([ 0.99954635-0.02299956j,  0.99954635+0.02299956j]),
    0.99981094420429639,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- gwpy.signal.filter_design ------------------------------------------------

class FilterDesignTestCase(unittest.TestCase):
    """`~unittest.TestCase` for the `gwpy.signal.filter_design` module
    """
    def test_notch_design(self):
        # test simple notch
        zpk = gwpy_signal.notch(60, 16384)
        for a, b in zip(zpk, NOTCH_60HZ):
            nptest.assert_array_almost_equal(a, b)
        # test Quantities
        zpk2 = gwpy_signal.notch(60 * ONE_HZ, 16384 * ONE_HZ)
        for a, b in zip(zpk, zpk2):
            nptest.assert_array_almost_equal(a, b)


# -- gwpy.signal.fft.registry -------------------------------------------------

class FFTRegistryTests(unittest.TestCase):
    def tearDown(self):
        # remove test methods from registry
        # otherwise they will impact other tests, and test ordering
        # is annoying to predict
        for scaling in fft_registry.METHODS:
            fft_registry.METHODS[scaling].pop('fake_method', '')

    def test_registry(self):
        def fake_method():
            pass

        # test register
        fft_registry.register_method(fake_method)
        self.assertIn('fake_method', fft_registry.METHODS['density'])
        self.assertRaises(KeyError, fft_registry.register_method, fake_method)
        fft_registry.register_method(fake_method, force=True)
        self.assertIn('fake_method', fft_registry.METHODS['density'])
        fft_registry.register_method(fake_method, scaling='spectrum')
        self.assertIn('fake_method', fft_registry.METHODS['spectrum'])
        self.assertRaises(KeyError, fft_registry.register_method,
                          fake_method, scaling='unknown')
        # test get
        f = fft_registry.get_method('fake_method')
        self.assertIs(f, fake_method)
        self.assertRaises(KeyError, fft_registry.get_method, 'unregistered')
        self.assertRaises(KeyError, fft_registry.get_method, 'fake_method',
                          scaling='unknown')

    def test_update_doc(self):
        def fake_caller():
            pass

        self.assertEqual(fake_caller.__doc__, None)
        fft_registry.update_doc(fake_caller)
        self.assertEqual(
            fake_caller.__doc__,
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
            'See :ref:`gwpy-signal-fft` for more details\n',
        )

# -- gwpy.signal.fft.utils ----------------------------------------------------

class FFTUtilsTests(unittest.TestCase):
    def test_scale_timeseries_unit(self):
        u = units.Unit('m')
        # check default
        self.assertEqual(fft_utils.scale_timeseries_unit(u),
                         units.Unit('m^2/Hz'))
        # check scaling='density'
        self.assertEqual(
            fft_utils.scale_timeseries_unit(u, scaling='density'),
            units.Unit('m^2/Hz'))
        # check scaling='spectrum'
        self.assertEqual(
            fft_utils.scale_timeseries_unit(u, scaling='spectrum'),
            units.Unit('m^2'))
        # check anything else raises an exception
        self.assertRaises(ValueError, fft_utils.scale_timeseries_unit,
                          u, scaling='other')
        # check null unit
        self.assertEqual(fft_utils.scale_timeseries_unit(None),
                         units.Unit('Hz^-1'))


# -- gwpy.signal.fft.lal ------------------------------------------------------

@unittest.skipUnless(HAS_LAL, 'No module named lal')
class LALFftTests(unittest.TestCase):
    def test_generate_window(self):
        # test default arguments
        w = fft_lal.generate_window(128)
        self.assertIsInstance(w, lal.REAL8Window)
        self.assertEqual(w.data.data.size, 128)
        self.assertEqual(w.sum, 32.31817089602309)
        # test generating the same window again returns the same object
        self.assertIs(fft_lal.generate_window(128), w)
        # test dtype works
        w = fft_lal.generate_window(128, dtype='float32')
        self.assertIsInstance(w, lal.REAL4Window)
        self.assertEqual(w.sum, 32.31817089602309)
        # test errors
        self.assertRaises(AttributeError, fft_lal.generate_window,
                          128, 'unknown')
        self.assertRaises(AttributeError, fft_lal.generate_window,
                          128, dtype=int)

    def test_generate_fft_plan(self):
        # test default arguments
        plan = fft_lal.generate_fft_plan(128)
        self.assertIsInstance(plan, lal.REAL8FFTPlan)
        # test generating the same fft_plan again returns the same object
        self.assertIs(fft_lal.generate_fft_plan(128), plan)
        # test dtype works
        plan = fft_lal.generate_fft_plan(128, dtype='float32')
        self.assertIsInstance(plan, lal.REAL4FFTPlan)
        # test forward/backward works
        rvrs = fft_lal.generate_fft_plan(128, forward=False)
        self.assertIsInstance(rvrs, lal.REAL8FFTPlan)
        self.assertIsNot(rvrs, plan)
        # test errors
        self.assertRaises(AttributeError, fft_lal.generate_fft_plan,
                          128, dtype=int)
