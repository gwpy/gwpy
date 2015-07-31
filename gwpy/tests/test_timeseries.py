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

"""Unit test for timeseries module
"""

import os
import os.path
import tempfile

from compat import unittest

import numpy
from numpy import testing as nptest

from astropy import units

from gwpy.time import Time

from gwpy import version
from gwpy.timeseries import (TimeSeries, StateVector)
from gwpy.spectrum import Spectrum
from gwpy.spectrogram import Spectrogram
from test_array import SeriesTestCase

SEED = 1
GPS_EPOCH = Time(0, format='gps', scale='utc')
ONE_HZ = units.Quantity(1, 'Hz')
ONE_SECOND = units.Quantity(1, 'second')

TEST_GWF_FILE = os.path.join(os.path.split(__file__)[0], 'data',
                          'HLV-GW100916-968654552-1.gwf')
TEST_HDF_FILE = '%s.hdf' % TEST_GWF_FILE[:-4]

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


# -----------------------------------------------------------------------------

class TimeSeriesTestMixin(object):
    """`~unittest.TestCase` for the `~gwpy.timeseries.TimeSeries` class
    """
    channel = 'L1:LDAS-STRAIN'
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_')
    TEST_CLASS = None

    def test_creation_with_metadata(self):
        self.ts = self.create()
        repr(self.ts)
        self.assertTrue(self.ts.epoch == GPS_EPOCH)
        self.assertTrue(self.ts.sample_rate == ONE_HZ)
        self.assertTrue(self.ts.dt == ONE_SECOND)

    def frame_read(self, format=None):
        ts = self.TEST_CLASS.read(
            TEST_GWF_FILE, self.channel, format=format)
        self.assertTrue(ts.epoch == Time(968654552, format='gps', scale='utc'))
        self.assertTrue(ts.sample_rate == units.Quantity(16384, 'Hz'))
        self.assertTrue(ts.unit == units.Unit('strain'))

    def test_epoch(self):
        array = self.create()
        self.assertEquals(array.epoch.gps, array.x0.value)

    def test_frame_read_lalframe(self):
        try:
            self.frame_read(format='lalframe')
        except ImportError as e:
            self.skipTest(str(e))

    def test_frame_read_framecpp(self):
        try:
            self.frame_read(format='framecpp')
        except ImportError as e:
            self.skipTest(str(e))

    def test_ascii_write(self, delete=True):
        self.ts = self.create()
        asciiout = self.tmpfile % 'txt'
        self.ts.write(asciiout)
        if delete and os.path.isfile(asciiout):
            os.remove(asciiout)
        return asciiout

    def test_ascii_read(self):
        fp = self.test_ascii_write(delete=False)
        try:
            self.TEST_CLASS.read(fp)
        finally:
            if os.path.isfile(fp):
                os.remove(fp)

    def test_hdf5_write(self, delete=True):
        self.ts = self.create(name=self.channel)
        hdfout = self.tmpfile % 'hdf'
        try:
            self.ts.write(hdfout)
        except ImportError as e:
            self.skipTest(str(e))
        finally:
            if delete and os.path.isfile(hdfout):
                os.remove(hdfout)
        return hdfout

    def test_hdf5_read(self):
        try:
            hdfout = self.test_hdf5_write(delete=False)
        except ImportError as e:
            self.skipTest(str(e))
        else:
            try:
                self.TEST_CLASS.read(hdfout, self.channel)
            finally:
                if os.path.isfile(hdfout):
                    os.remove(hdfout)

    def test_resample(self):
        """Test the `TimeSeries.resample` method
        """
        ts1 = self.create(sample_rate=100)
        ts2 = ts1.resample(10)
        self.assertEquals(ts2.sample_rate, ONE_HZ*10)

    def test_to_from_lal(self):
        ts = self.create()
        try:
            lalts = ts.to_lal()
        except (NotImplementedError, ImportError) as e:
            self.skipTest(str(e))
        ts2 = type(ts).from_lal(lalts)
        self.assertEqual(ts, ts2)


# -----------------------------------------------------------------------------

class TimeSeriesTestCase(TimeSeriesTestMixin, SeriesTestCase):
    TEST_CLASS = TimeSeries

    def _read(self):
        return self.TEST_CLASS.read(TEST_HDF_FILE, self.channel)

    def test_fft(self):
        ts = self._read()
        fs = ts.fft()
        self.assertEqual(fs.size, ts.size//2+1)
        fs = ts.fft(nfft=256)
        self.assertEqual(fs.size, 129)
        self.assertIsInstance(fs, Spectrum)
        self.assertEqual(fs.x0, 0*units.Hertz)
        self.assertEqual(fs.dx, 1*units.Hertz)
        self.assertIs(ts.channel, fs.channel)

    def test_average_fft(self):
        ts = self._read()
        # test all defaults
        fs = ts.average_fft()
        self.assertEqual(fs.size, ts.size//2+1)
        self.assertEqual(fs.f0, 0 * units.Hertz)
        self.assertIsInstance(fs, Spectrum)
        self.assertIs(fs.channel, ts.channel)
        # test fftlength
        fs = ts.average_fft(fftlength=0.5)
        self.assertEqual(fs.size, 0.5 * ts.sample_rate.value // 2 + 1)
        self.assertEqual(fs.df, 2 * units.Hertz)
        # test overlap
        fs = ts.average_fft(fftlength=0.4, overlap=0.2)

    def test_psd(self):
        ts = self._read()
        # test all defaults
        fs = ts.psd()
        self.assertEqual(fs.size, ts.size//2+1)
        self.assertEqual(fs.f0, 0*units.Hertz)
        self.assertEqual(fs.df, 1 / ts.duration)
        self.assertIsInstance(fs, Spectrum)
        self.assertIs(fs.channel, ts.channel)
        self.assertEqual(fs.unit, ts.unit ** 2 / units.Hertz)
        # test fftlength
        fs = ts.psd(fftlength=0.5)
        self.assertEqual(fs.size, 0.5 * ts.sample_rate.value // 2 + 1)
        self.assertEqual(fs.df, 2 * units.Hertz)
        # test overlap
        ts.psd(fftlength=0.4, overlap=0.2)
        # test methods
        ts.psd(fftlength=0.4, overlap=0.2, method='welch')
        try:
            ts.psd(fftlength=0.4, overlap=0.2, method='lal-welch')
        except ImportError as e:
            pass
        else:
            ts.psd(fftlength=0.4, overlap=0.2, method='median-mean')
            ts.psd(fftlength=0.4, overlap=0.2, method='median')
            # test check for at least two averages
            self.assertRaises(ValueError, ts.psd, method='median-mean')

    def test_asd(self):
        ts = self._read()
        fs = ts.asd()
        self.assertEqual(fs.unit, ts.unit / units.Hertz ** (1/2.))

    def test_spectrogram(self):
        ts = self._read()
        # test defaults
        sg = ts.spectrogram(1)
        self.assertEqual(sg.shape, (1, ts.size//2+1))
        self.assertEqual(sg.f0, 0*units.Hertz)
        self.assertEqual(sg.df, 1 / ts.duration)
        self.assertIsInstance(sg, Spectrogram)
        self.assertIs(sg.channel, ts.channel)
        self.assertEqual(sg.unit, ts.unit ** 2 / units.Hertz)
        self.assertEqual(sg.epoch, ts.epoch)
        self.assertEqual(sg.span, ts.span)
        # check the same result as PSD
        psd = ts.psd()
        nptest.assert_array_equal(sg.data[0], psd.data)
        # test fftlength
        sg = ts.spectrogram(1, fftlength=0.5)
        self.assertEqual(sg.shape, (1, 0.5 * ts.size//2+1))
        self.assertEqual(sg.df, 2 * units.Hertz)
        self.assertEqual(sg.dt, 1 * units.second)
        # test overlap
        sg = ts.spectrogram(0.5, fftlength=0.2, overlap=0.1)
        self.assertEqual(sg.shape, (2, 0.2 * ts.size//2 + 1))
        self.assertEqual(sg.df, 5 * units.Hertz)
        self.assertEqual(sg.dt, 0.5 * units.second)
        # test multiprocessing
        sg2 = ts.spectrogram(0.5, fftlength=0.2, overlap=0.1, nproc=2)
        self.assertArraysEqual(sg, sg2)

    def test_spectrogram2(self):
        ts = self._read()
        # test defaults
        sg = ts.spectrogram2(1)
        self.assertEqual(sg.shape, (1, ts.size//2+1))
        self.assertEqual(sg.f0, 0*units.Hertz)
        self.assertEqual(sg.df, 1 / ts.duration)
        self.assertIsInstance(sg, Spectrogram)
        self.assertIs(sg.channel, ts.channel)
        self.assertEqual(sg.unit, ts.unit ** 2 / units.Hertz)
        self.assertEqual(sg.epoch, ts.epoch)
        self.assertEqual(sg.span, ts.span)
        # test the same result as spectrogam
        sg1 = ts.spectrogram(1)
        nptest.assert_array_equal(sg.data, sg1.data)
        # test fftlength
        sg = ts.spectrogram2(0.5)
        self.assertEqual(sg.shape, (2, 0.5 * ts.size//2+1))
        self.assertEqual(sg.df, 2 * units.Hertz)
        self.assertEqual(sg.dt, 0.5 * units.second)
        # test overlap
        sg = ts.spectrogram2(fftlength=0.2, overlap=0.19)
        self.assertEqual(sg.shape, (99, 0.2 * ts.size//2 + 1))
        self.assertEqual(sg.df, 5 * units.Hertz)
        # note: bizarre stride length because 16384/100 gets rounded
        self.assertEqual(sg.dt, 0.010009765625 * units.second)


class StateVectorTestCase(TimeSeriesTestMixin, SeriesTestCase):
    """`~unittest.TestCase` for the `~gwpy.timeseries.StateVector` object
    """
    TEST_CLASS = StateVector

    def setUp(self):
        super(StateVectorTestCase, self).setUp(dtype='uint32')


if __name__ == '__main__':
    unittest.main()
