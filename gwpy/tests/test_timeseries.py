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
import pytest
import tempfile

from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError

from compat import unittest

import numpy
from numpy import testing as nptest

from scipy import signal

from matplotlib import use
use('agg')

from astropy import units
from astropy.io.registry import (get_reader, register_reader)

from gwpy.time import (Time, LIGOTimeGPS)

from gwpy.timeseries import (TimeSeries, StateVector, TimeSeriesDict,
                             StateVectorDict, TimeSeriesList)
from gwpy.segments import (Segment, DataQualityFlag, DataQualityDict)
from gwpy.frequencyseries import (FrequencySeries, SpectralVariance)
from gwpy.types import Array2D
from gwpy.spectrogram import Spectrogram
from gwpy.io.cache import Cache
from gwpy.plotter import (TimeSeriesPlot, SegmentPlot)

from test_array import SeriesTestCase
import common

SEED = 1
numpy.random.seed(SEED)
GPS_EPOCH = Time(0, format='gps', scale='utc')
ONE_HZ = units.Quantity(1, 'Hz')
ONE_SECOND = units.Quantity(1, 'second')

TEST_GWF_FILE = os.path.join(os.path.split(__file__)[0], 'data',
                          'HLV-GW100916-968654552-1.gwf')
TEST_HDF_FILE = '%s.hdf' % TEST_GWF_FILE[:-4]
TEST_SEGMENT = Segment(968654552, 968654553)


FIND_CHANNEL = 'L1:LDAS-STRAIN'
FIND_GPS = 968654552
FIND_FRAMETYPE = 'L1_LDAS_C02_L2'

LOSC_HDF_FILE = ("https://losc.ligo.org/archive/data/S6/930086912/"
                 "L-L1_LOSC_4_V1-931069952-4096.hdf5")
LOSC_DQ_BITS = [
    'Science data available',
    'Category-1 checks passed for CBC high-mass search',
    'Category-2 and 1 checks passed for CBC high-mass search',
    'Category-3 and 2 and 1 checks passed for CBC high-mass search',
    'Category-4,3,2,1 checks passed for CBC high-mass search',
    'Category-1 checks passed for CBC low-mass search',
    'Category-2 and 1 checks passed for CBC low-mass search',
    'Category-3 and 2 and 1 checks passed for CBC low-mass search',
    'Category-4, veto active for CBC low-mass search',
    'Category-1 checks passed for burst search',
    'Category-2 and 1 checks passed for burst search',
    'Category-3 and 2 and 1 checks passed for burst search',
    'Category-4, 3 and 2 and 1 checks passed for burst search',
    'Category-3 and 2 and 1 and hveto checks passed for burst search',
    'Category-4, 3 and 2 and 1 and hveto checks passed for burst search',
    'Category-1 checks passed for continuous-wave search',
    'Category-1 checks passed for stochastic search',
]
LOSC_GW150914 = 1126259462

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -----------------------------------------------------------------------------

class TimeSeriesTestMixin(object):
    """`~unittest.TestCase` for the `~gwpy.timeseries.TimeSeries` class
    """
    channel = 'L1:LDAS-STRAIN'

    def test_creation_with_metadata(self):
        self.ts = self.create()
        repr(self.ts)
        self.assertEqual(self.ts.epoch, GPS_EPOCH)
        self.assertEqual(self.ts.sample_rate, ONE_HZ)
        self.assertEqual(self.ts.dt, ONE_SECOND)

    def frame_read(self, format=None):
        ts = self.TEST_CLASS.read(
            TEST_GWF_FILE, self.channel, format=format)
        self.assertEqual(ts.epoch, Time(968654552, format='gps', scale='utc'))
        self.assertEqual(ts.sample_rate, units.Quantity(16384, 'Hz'))
        self.assertEqual(ts.unit, units.Unit('strain'))
        # check that channel carries the correct parameters
        self.assertEqual(ts.channel.sample_rate, ts.sample_rate)
        self.assertEqual(ts.channel.unit, ts.unit)
        return ts

    def test_ligotimegps(self):
        # test that LIGOTimeGPS works
        array = self.create(t0=LIGOTimeGPS(0))
        self.assertEqual(array.t0.value, 0)
        array.t0 = LIGOTimeGPS(10)
        self.assertEqual(array.t0.value, 10)
        array.x0 = LIGOTimeGPS(1000000000)
        self.assertEqual(array.t0.value, 1000000000)
        # check epoch access
        array.epoch = LIGOTimeGPS(10)
        self.assertEqual(array.t0.value, 10)

    def test_epoch(self):
        array = self.create()
        self.assertEquals(array.epoch.gps, array.x0.value)

    def _test_frame_read_format(self, format):
        # test with specific format
        try:
            self.frame_read(format=format)
        except ImportError as e:
            self.skipTest(str(e))
        else:
            # test again with no format argument
            # but we need to move other readers out of the way first
            try:
                read_ = get_reader('gwf', TimeSeries)
            except Exception:
                pass
            else:
                register_reader('gwf', TimeSeries,
                                get_reader(format, TimeSeries),
                                force=True)
                try:
                    self.frame_read()
                finally:
                    register_reader('gwf', TimeSeries, read_, force=True)
            # test empty Cache()
            self.assertRaises(ValueError, self.TEST_CLASS.read, Cache(),
                              self.channel, format=format)

            # test cache method with `nproc=2`
            c = Cache.from_urls([TEST_GWF_FILE])
            ts = self.TEST_CLASS.read(c, self.channel, nproc=2, format=format)

    def test_frame_read_lalframe(self):
        return self._test_frame_read_format('lalframe')

    def test_frame_read_framecpp(self):
        return self._test_frame_read_format('framecpp')

    def test_frame_read_cache(self):
        try:
            a = self.TEST_CLASS.read(TEST_GWF_FILE, self.channel)
        except Exception as e:  # don't care why this fails for this test
            self.skipTest(str(e))
        c = Cache.from_urls([TEST_GWF_FILE])
        with tempfile.NamedTemporaryFile(suffix='.lcf', delete=False) as f:
            c.tofile(f)
            f.delete = True
            b = self.TEST_CLASS.read(f.name, self.channel)
            self.assertArraysEqual(a, b)
            b = self.TEST_CLASS.read(open(f.name), self.channel)
            self.assertArraysEqual(a, b)

    def frame_write(self, format=None):
        try:
            ts = self.TEST_CLASS.read(TEST_GWF_FILE, self.channel,
                                      format=format)
        except ImportError as e:
            self.skipTest(str(e))
        except Exception as e:
            if 'No reader' in str(e):
                self.skipTest(str(e))
            else:
                raise
        else:
            with tempfile.NamedTemporaryFile(suffix='.gwf') as f:
                ts.write(f.name, format=format)
                ts2 = self.TEST_CLASS.read(f.name, self.channel)
            self.assertArraysEqual(ts, ts2)
            for ctype in ['sim', 'proc', 'adc']:
                ts.channel._ctype = ctype
                with tempfile.NamedTemporaryFile(suffix='.gwf') as f:
                    ts.write(f.name, format=format)

    def test_frame_write(self):
        try:
            self.frame_write(format='gwf')
        except Exception as e:
            if 'No writer' in str(e):
                self.skipTest(str(e))

    def test_frame_write_framecpp(self):
        self.frame_write(format='framecpp')

    def test_find(self):
        try:
            ts = self.TEST_CLASS.find(FIND_CHANNEL, FIND_GPS, FIND_GPS+1,
                                      frametype=FIND_FRAMETYPE)
        except (ImportError, RuntimeError) as e:
            self.skipTest(str(e))
        else:
            self.assertEqual(ts.x0.value, FIND_GPS)
            self.assertEqual(abs(ts.span), 1)
            try:
                comp = self.frame_read()
            except ImportError:
                pass
            else:
                nptest.assert_array_almost_equal(ts.value, comp.value)
            # test observatory
            ts2 = self.TEST_CLASS.find(FIND_CHANNEL, FIND_GPS, FIND_GPS+1,
                                      frametype=FIND_FRAMETYPE,
                                      observatory=FIND_CHANNEL[0])
            self.assertArraysEqual(ts, ts2)
            self.assertRaises(RuntimeError, self.TEST_CLASS.find, FIND_CHANNEL,
                              FIND_GPS, FIND_GPS+1, frametype=FIND_FRAMETYPE,
                              observatory='X')

    def test_find_best_frametype(self):
        from gwpy.io import datafind
        # check we can actually run this test here
        try:
            os.environ['LIGO_DATAFIND_SERVER']
        except KeyError as e:
            self.skipTest(str(e))
        # test a few (channel, frametype) pairs
        for channel, target in [
                ('H1:GDS-CALIB_STRAIN',
                    ['H1_HOFT_C00', 'H1_ER_C00_L1']),
                ('L1:IMC-ODC_CHANNEL_OUT_DQ',
                    ['L1_R']),
                ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend',
                    ['H1_T']),
                ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend',
                    ['H1_M'])]:
            ft = datafind.find_best_frametype(
                channel, 1143504017, 1143504017+100)
            self.assertIn(ft, target)

        # test that this works end-to-end as part of a TimeSeries.find
        try:
            ts = self.TEST_CLASS.find(FIND_CHANNEL, FIND_GPS, FIND_GPS+1)
        except (ImportError, RuntimeError) as e:
            self.skipTest(str(e))

    def test_get(self):
        try:
            ts = self.TEST_CLASS.get(FIND_CHANNEL, FIND_GPS, FIND_GPS+1)
        except (ImportError, RuntimeError) as e:
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

    def test_resample(self):
        """Test the `TimeSeries.resample` method
        """
        ts1 = self.create(sample_rate=100)
        ts2 = ts1.resample(10, ftype='iir')
        self.assertEquals(ts2.sample_rate, ONE_HZ*10)
        self.assertEqual(ts1.unit, ts2.unit)
        ts1.resample(10, ftype='fir', n=10)

    def test_to_from_lal(self):
        ts = self.create()
        try:
            lalts = ts.to_lal()
        except (NotImplementedError, ImportError) as e:
            self.skipTest(str(e))
        ts2 = type(ts).from_lal(lalts)
        self.assertEqual(ts, ts2)
        # test copy=False
        ts2 = type(ts).from_lal(lalts, copy=False)
        self.assertEqual(ts, ts2)
        # test no unit
        ts.override_unit(None)
        ts2 = type(ts).from_lal(lalts, copy=False)
        self.assertIs(ts2.unit, units.dimensionless_unscaled)

    def test_io_identify(self):
        common.test_io_identify(self.TEST_CLASS, ['txt', 'hdf', 'gwf'])

    def fetch_open_data(self):
        try:
            return self._open_data
        except AttributeError:
            try:
                type(self)._open_data = self.TEST_CLASS.fetch_open_data(
                    self.channel[:2], *TEST_SEGMENT)
            except URLError as e:
                self.skipTest(str(e))
            else:
                return self.fetch_open_data()

    def test_fetch_open_data(self):
        ts = self.fetch_open_data()
        self.assertEqual(ts.unit, units.Unit('strain'))
        self.assertEqual(ts.sample_rate, 4096 * units.Hz)
        self.assertEqual(ts.span, TEST_SEGMENT)
        nptest.assert_allclose(
            ts.value[:10],
            [-2.86995824e-17, -2.77331804e-17, -2.67514139e-17,
             -2.57456587e-17, -2.47232689e-17, -2.37029998e-17,
             -2.26415858e-17, -2.15710360e-17, -2.04492206e-17,
             -1.93265041e-17])
        # try GW150914 data at 16 kHz
        try:
            ts = self.TEST_CLASS.fetch_open_data(
                self.channel[:2], LOSC_GW150914-16, LOSC_GW150914+16,
                sample_rate=16384)
        except URLError as e:
            self.skipTest(str(e))
        else:
            self.assertEqual(ts.sample_rate, 16384 * units.Hz)

        # make sure errors get thrown
        self.assertRaises(ValueError, self.TEST_CLASS.fetch_open_data,
                          self.channel[:2], 0, 1)

    def test_losc(self):
        _, tmpfile = tempfile.mkstemp(prefix='GWPY-TEST_LOSC_', suffix='.hdf')
        try:
            response = urlopen(LOSC_HDF_FILE)
            with open(tmpfile, 'w') as f:
                f.write(response.read())
            self._test_losc_inner(tmpfile)  # actually run test here
        except (URLError, ImportError) as e:
            self.skipTest(str(e))
        finally:
            if os.path.isfile(tmpfile):
                os.remove(tmpfile)

    def _test_losc_inner(self):
        self.skipTest("LOSC inner test method has not been written yet")

    def test_plot(self):
        ts = self.create()
        plot = ts.plot()
        self.assertIsInstance(plot, TimeSeriesPlot)
        return plot


# -----------------------------------------------------------------------------

class TimeSeriesTestCase(TimeSeriesTestMixin, SeriesTestCase):
    TEST_CLASS = TimeSeries

    @classmethod
    def setUpClass(cls, dtype=None):
        super(TimeSeriesTestCase, cls).setUpClass(dtype=dtype)
        cls.random = cls.TEST_CLASS(
            numpy.random.normal(loc=1, size=16384 * 10), sample_rate=16384,
            epoch=-5)

    def _read(self):
        return self.TEST_CLASS.read(TEST_HDF_FILE, self.channel)

    def test_fft(self):
        ts = self._read()
        fs = ts.fft()
        self.assertEqual(fs.size, ts.size//2+1)
        self.assertIsInstance(fs, FrequencySeries)
        self.assertEqual(fs.x0, 0*units.Hertz)
        self.assertEqual(fs.dx, 1*units.Hertz)
        self.assertIs(ts.channel, fs.channel)
        # test with nfft arg
        fs = ts.fft(nfft=256)
        self.assertEqual(fs.size, 129)
        self.assertEqual(fs.dx, ts.sample_rate / 256)

    def test_average_fft(self):
        ts = self._read()
        # test all defaults
        fs = ts.average_fft()
        self.assertEqual(fs.size, ts.size//2+1)
        self.assertEqual(fs.f0, 0 * units.Hertz)
        self.assertIsInstance(fs, FrequencySeries)
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
        self.assertIsInstance(fs, FrequencySeries)
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

    def test_csd(self):
        ts = self._read()
        # test all defaults
        fs = ts.csd(ts)
        self.assertEqual(fs.size, ts.size//2+1)
        self.assertEqual(fs.f0, 0*units.Hertz)
        self.assertEqual(fs.df, 1 / ts.duration)
        self.assertIsInstance(fs, FrequencySeries)
        self.assertIs(fs.channel, ts.channel)
        self.assertEqual(fs.unit, ts.unit ** 2 / units.Hertz)
        # test that self-CSD is equal to PSD
        sp = ts.psd()
        nptest.assert_array_equal(fs.value, sp.value)
        # test fftlength
        fs = ts.csd(ts, fftlength=0.5)
        self.assertEqual(fs.size, 0.5 * ts.sample_rate.value // 2 + 1)
        self.assertEqual(fs.df, 2 * units.Hertz)
        # test overlap
        ts.csd(ts, fftlength=0.4, overlap=0.2)

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
        nptest.assert_array_equal(sg.value[0], psd.value)
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
        # test methods
        ts.spectrogram(0.5, fftlength=0.2, method='bartlett')

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
        nptest.assert_array_equal(sg.value, sg1.value)
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

    def test_spectral_variance(self):
        ts = self._read()
        variance = ts.spectral_variance(.5)
        self.assertIsInstance(variance, SpectralVariance)

    def test_whiten(self):
        # create noise with a glitch in it at 1000 Hz
        noise = self.random.zpk([], [0], 1)
        glitchtime = 0.5
        glitch = signal.gausspulse(noise.times.value + glitchtime,
                                   bw=100) * 1e-4
        data = noise + glitch
        # whiten and test that the max amplitude is recovered at the glitch
        tmax = data.times[data.argmax()]
        self.assertNotAlmostEqual(tmax.value, -glitchtime)
        whitened = data.whiten(2, 1)
        self.assertEqual(noise.size, whitened.size)
        self.assertAlmostEqual(whitened.mean().value, 0.0, places=4)
        tmax = whitened.times[whitened.argmax()]
        self.assertAlmostEqual(tmax.value, -glitchtime)

    def test_detrend(self):
        self.assertNotAlmostEqual(self.random.mean(), 0.0)
        detrended = self.random.detrend()
        self.assertAlmostEqual(detrended.mean(), 0.0)

    def test_csd_spectrogram(self):
        ts = self._read()
        # test defaults
        sg = ts.csd_spectrogram(ts, 1)
        self.assertEqual(sg.shape, (1, ts.size//2+1))
        self.assertEqual(sg.f0, 0*units.Hertz)
        self.assertEqual(sg.df, 1 / ts.duration)
        self.assertIsInstance(sg, Spectrogram)
        self.assertIs(sg.channel, ts.channel)
        self.assertEqual(sg.unit, ts.unit ** 2 / units.Hertz)
        self.assertEqual(sg.epoch, ts.epoch)
        self.assertEqual(sg.span, ts.span)
        # check the same result as CSD
        csd = ts.csd(ts)
        nptest.assert_array_equal(sg.value[0], csd.value)
        # test fftlength
        sg = ts.csd_spectrogram(ts, 1, fftlength=0.5)
        self.assertEqual(sg.shape, (1, 0.5 * ts.size//2+1))
        self.assertEqual(sg.df, 2 * units.Hertz)
        self.assertEqual(sg.dt, 1 * units.second)
        # test overlap
        sg = ts.csd_spectrogram(ts, 0.5, fftlength=0.2, overlap=0.1)
        self.assertEqual(sg.shape, (2, 0.2 * ts.size//2 + 1))
        self.assertEqual(sg.df, 5 * units.Hertz)
        self.assertEqual(sg.dt, 0.5 * units.second)
        # test multiprocessing
        sg2 = ts.csd_spectrogram(ts, 0.5, fftlength=0.2, overlap=0.1, nproc=2)
        self.assertArraysEqual(sg, sg2)
        # test method not 'welch' raises warning
        with pytest.warns(UserWarning):
           ts.csd_spectrogram(ts, 0.5, method='median-mean')

    def test_notch(self):
        # test notch runs end-to-end
        ts = self.create(sample_rate=256)
        notched = ts.notch(10)
        # test breaks when you try and 'fir' notch
        self.assertRaises(NotImplementedError, ts.notch, 10, type='fir')

    def _test_losc_inner(self, loscfile):
        ts = self.TEST_CLASS.read(loscfile, 'Strain', format='losc')
        self.assertEqual(ts.x0, units.Quantity(931069952, 's'))
        self.assertEqual(ts.dx, units.Quantity(0.000244140625, 's'))
        self.assertEqual(ts.name, 'Strain')

    def test_q_transform(self):
        gps = 968654558
        duration = 32
        start = int(round(gps - duration/2.))
        end = start + duration
        try:
            ts = self.TEST_CLASS.fetch_open_data('H1', start, end)
        except (ImportError, RuntimeError) as e:
            self.skipTest(str(e))
        else:
            qspecgram = ts.q_transform(method='welch')
            self.assertIsInstance(qspecgram, Spectrogram)
            self.assertTupleEqual(qspecgram.shape, (32000, 2560))
            self.assertAlmostEqual(qspecgram.q, 11.31370849898476)
            self.assertAlmostEqual(qspecgram.value.max(), 37.035843858490509)


class StateVectorTestCase(TimeSeriesTestMixin, SeriesTestCase):
    """`~unittest.TestCase` for the `~gwpy.timeseries.StateVector` object
    """
    TEST_CLASS = StateVector

    @classmethod
    def setUpClass(cls, dtype='uint32'):
        super(StateVectorTestCase, cls).setUpClass(dtype=dtype)

    def _test_losc_inner(self, loscfile):
        ts = self.TEST_CLASS.read(loscfile, 'quality/simple', format='losc')
        self.assertEqual(ts.x0, units.Quantity(931069952, 's'))
        self.assertEqual(ts.dx, units.Quantity(1.0, 's'))
        self.assertListEqual(list(ts.bits), LOSC_DQ_BITS)

    def test_fetch_open_data(self):
        ts = self.fetch_open_data()
        self.assertEqual(ts.sample_rate, 1 * units.Hz)
        self.assertEqual(ts.span, TEST_SEGMENT)
        self.assertListEqual(list(ts.bits), LOSC_DQ_BITS)
        self.assertEqual(ts.value[0], 131071)  # sanity check data

    def test_to_dqflags(self):
        sv = self.fetch_open_data()
        dqdict = sv.to_dqflags()
        self.assertIsInstance(dqdict, DataQualityDict)
        for i, (key, flag) in enumerate(dqdict.items()):
            self.assertIsInstance(flag, DataQualityFlag)
            self.assertEqual(flag.name, sv.bits[i])
            self.assertListEqual(flag.known, [sv.span])

    def test_plot(self):
        data = self.fetch_open_data()
        # test segment plotting
        plot = data.plot()
        self.assertIsInstance(plot, SegmentPlot)
        self.assertEqual(len(plot.gca().collections), len(data.bits) * 2)
        plot.close()
        # test timeseries plotting
        plot = data.plot(format='timeseries')
        self.assertIsInstance(plot, TimeSeriesPlot)
        self.assertEqual(len(plot.gca().lines), 1)
        plot.close()

    def test_boolean(self):
        data = self.fetch_open_data()
        b = data.boolean
        self.assertIsInstance(b, Array2D)
        self.assertTupleEqual(b.shape, (data.size, len(data.bits)))

    def test_resample(self):
        ts1 = self.create(sample_rate=100)
        ts2 = ts1.resample(10)
        self.assertEqual(ts2.sample_rate, ONE_HZ*10)
        self.assertEqual(ts1.unit, ts2.unit)


# -- TimeSeriesDict tests ------------------------------------------------------

class TimeSeriesDictTestCase(unittest.TestCase):
    channels = ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN']
    TEST_CLASS = TimeSeriesDict

    def read(self):
        try:
            return self._test_data.copy()
        except AttributeError:
            try:
                self._test_data = self.TEST_CLASS.read(
                    TEST_GWF_FILE, self.channels)
            except ImportError as e:
                self.skipTest(str(e))
            except Exception as e:
                if 'No reader' in str(e):
                    self.skipTest(str(e))
                else:
                    raise
            else:
                return self.read()

    def test_init(self):
        tsd = self.TEST_CLASS()

    def test_frame_read(self):
        try:
            return self.TEST_CLASS.read(TEST_GWF_FILE, self.channels)
        except Exception as e:
            if 'No reader' in str(e):
                self.skipTest(str(e))
            else:
                raise

    def test_frame_write(self):
        try:
            tsd = self.test_frame_read()
        except ImportError as e:
            self.skipTest(str(e))
        except Exception as e:
            if 'No reader' in str(e):
                self.skipTest(str(e))
            else:
                raise
        else:
            with tempfile.NamedTemporaryFile(suffix='.gwf') as f:
                try:
                    tsd.write(f.name)
                except Exception as e:
                    if 'No writer' in str(e):
                        self.skipTest(str(e))
                tsd2 = self.TEST_CLASS.read(f.name, tsd.keys())
            self.assertDictEqual(tsd, tsd2)

    def test_plot(self):
        tsd = self.read()
        plot = tsd.plot()
        self.assertIsInstance(plot, TimeSeriesPlot)
        self.assertEqual(len(plot.gca().lines), 2)

    def test_resample(self):
        tsd = self.read()
        tsd.resample(2048)
        for key in tsd:
            self.assertEqual(tsd[key].sample_rate, 2048 * units.Hertz)

    def test_crop(self):
        tsd = self.read()
        tsd.crop(968654552, 968654552.5)
        for key in tsd:
            self.assertEqual(tsd[key].span, Segment(968654552, 968654552.5))

    def test_append(self):
        a = self.read()
        a.crop(968654552, 968654552.5, copy=True)
        b = self.read()
        b.crop(968654552.5, 968654553)
        a.append(b)
        for key in a:
            self.assertEqual(a[key].span, Segment(968654552, 968654553))

    def test_prepend(self):
        a = self.read()
        a.crop(968654552, 968654552.5)
        b = self.read()
        b.crop(968654552.5, 968654553, copy=True)
        b.prepend(a)
        for key in b:
            self.assertEqual(b[key].span, Segment(968654552, 968654553))


class StateVectorDictTestCase(TimeSeriesDictTestCase):
    TEST_CLASS = StateVectorDict

    def test_plot(self):
        tsd = self.read()
        plot = tsd.plot()
        self.assertIsInstance(plot, TimeSeriesPlot)


# -- TimeSeriesList tests -----------------------------------------------------

class TimeSeriesListTestCase(unittest.TestCase):
    TEST_CLASS = TimeSeriesList

    def create(self):
        out = self.TEST_CLASS()
        for epoch in [0, 100, 400]:
            data = (numpy.random.random(100) * 1e5).astype(float)
            out.append(out.EntryClass(data, epoch=epoch, sample_rate=1))
        return out

    def test_segments(self):
        tsl = self.create()
        segs = tsl.segments
        self.assertListEqual(tsl.segments, [(0, 100), (100, 200), (400, 500)])

    def test_coalesce(self):
        tsl = self.create()
        tsl2 = self.create().coalesce()
        self.assertEqual(tsl2[0], tsl[0].append(tsl[1], inplace=False))


if __name__ == '__main__':
    unittest.main()
