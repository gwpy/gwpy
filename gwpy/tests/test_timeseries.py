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

from compat import (unittest, mock)

import pytest

import numpy
from numpy import testing as nptest

from scipy import signal

from matplotlib import use
use('agg')

from astropy import units
from astropy.io.registry import (get_reader, register_reader)

from gwpy.detector import Channel
from gwpy.time import (Time, LIGOTimeGPS)
from gwpy.timeseries import (TimeSeries, StateVector, TimeSeriesDict,
                             StateVectorDict, TimeSeriesList, StateTimeSeries)
from gwpy.segments import (Segment, DataQualityFlag, DataQualityDict)
from gwpy.frequencyseries import (FrequencySeries, SpectralVariance)
from gwpy.types import Array2D
from gwpy.spectrogram import Spectrogram
from gwpy.io.cache import Cache
from gwpy.plotter import (TimeSeriesPlot, SegmentPlot)

from test_array import SeriesTestCase
import common
import mockutils

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

    # -- test I/O -------------------------------

    def _test_read_cache(self, format, extension=None, exclude=['channel']):
        if extension is None:
            extension = format
        # make array
        a = self.create(name='test', t0=0, sample_rate=self.data.shape[0])
        exta = '-%d-%d.%s' % (a.span[0], a.span[1], extension)

        # write it to a file, so we can read it again later
        with tempfile.NamedTemporaryFile(prefix='tmp-', suffix=exta,
                                         delete=False) as f1:
            a.write(f1.name)

        # test reading it from the cache
        cache = Cache.from_urls([f1.name], coltype=int)
        b = self.TEST_CLASS.read(cache, a.name)
        self.assertArraysEqual(a, b, exclude=exclude)

        # write a cache file and read that
        try:
            with tempfile.NamedTemporaryFile(suffix='.lcf', delete=False) as f:
                cache.tofile(f)
                b = self.TEST_CLASS.read(f.name, a.name)
                self.assertArraysEqual(a, b, exclude=exclude)
                b = self.TEST_CLASS.read(open(f.name), a.name)
                self.assertArraysEqual(a, b, exclude=exclude)
        finally:
            if os.path.isfile(f.name):
                os.remove(f.name)

        # create second array with a gap
        b = self.create(name='test', t0=a.xspan[1]+1, dt=a.dt)
        extb = '-%d-%d.%s' % (b.span[0], b.span[1], extension)
        try:
            with tempfile.NamedTemporaryFile(prefix='tmp-', suffix=extb,
                                             delete=False) as f2:
                # write tmp file
                b.write(f2.name)
                # make cache of file names
                cache = Cache.from_urls([f1.name, f2.name], coltype=int)
                # assert gap='raise' actually raises by default
                self.assertRaises(ValueError, self.TEST_CLASS.read,
                                  cache, a.name)
                # read from cache
                ts = self.TEST_CLASS.read(cache, a.name, gap='pad', pad=0)
                nptest.assert_array_equal(
                    ts.value,
                    numpy.concatenate((a.value,
                                       numpy.zeros(int(a.sample_rate.value)),
                                       b.value)))
                # read with multi-processing
                ts2 = self.TEST_CLASS.read(cache, a.name, nproc=2,
                                           gap='pad', pad=0)
                self.assertArraysEqual(ts, ts2)
        finally:
            # clean up
            for f in (f1, f2):
                if os.path.exists(f.name):
                    os.remove(f.name)

    def test_read_write_gwf(self):
        # test basic read
        try:
            self._test_read_write('gwf', exclude=['channel'])
        except ImportError as e:
            self.skipTest(str(e))
        # test cache read
        self._test_read_cache('gwf')
        # check reading with start/end works
        start, end = TEST_SEGMENT.contract(.25)
        t = self.TEST_CLASS.read(TEST_GWF_FILE, self.channel, format='gwf',
                                 start=start, end=end)
        self.assertTupleEqual(t.span, (start, end))
        t = self.TEST_CLASS.read(TEST_GWF_FILE, self.channel, format='gwf',
                                 start=start)
        self.assertTupleEqual(t.span, (start, TEST_SEGMENT[1]))
        t = self.TEST_CLASS.read(TEST_GWF_FILE, self.channel, format='gwf',
                                 end=end)
        self.assertTupleEqual(t.span, (TEST_SEGMENT[0], end))
        # check errors
        self.assertRaises((ValueError, RuntimeError), self.TEST_CLASS.read,
                          TEST_GWF_FILE, self.channel, format='gwf',
                          start=TEST_SEGMENT[1])
        self.assertRaises((ValueError, RuntimeError), self.TEST_CLASS.read,
                          TEST_GWF_FILE, self.channel, format='gwf',
                          end=TEST_SEGMENT[0])

    def read_write_gwf_api(self, api):
        fmt = 'gwf.%s' % api
        try:
            self._test_read_write(fmt, extension='gwf', exclude=['channel'],
                                  auto=True)#False)
        except ImportError as e:
            self.skipTest(str(e))
        self._test_read_cache(fmt, extension='gwf')
        # check old format prints a deprecation warning
        with pytest.warns(DeprecationWarning):
            self.TEST_CLASS.read(TEST_GWF_FILE, self.channel, format=api)

    def test_read_write_gwf_lalframe(self):
        return self.read_write_gwf_api('lalframe')

    def test_read_write_gwf_framecpp(self):
        return self.read_write_gwf_api('framecpp')

    def test_read_write_hdf5(self):
        # test basic read
        try:
            self._test_read_write('hdf5', exclude=['channel'], auto=False)
        except ImportError as e:
            self.skipTest(str(e))
        self._test_read_write('hdf5', exclude=['channel'], auto=True,
                              writekwargs={'overwrite': True})
        # check reading with start/end works
        start, end = TEST_SEGMENT.contract(.25)
        t = self.TEST_CLASS.read(TEST_HDF_FILE, self.channel, format='hdf5',
                                 start=start, end=end)
        self.assertTupleEqual(t.span, (start, end))

    def test_read_write_ascii(self):
        return self._test_read_write_ascii(format='txt')

    def test_read_write_csv(self):
        return self._test_read_write_ascii(format='csv')

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

    # -- methods --------------------------------

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
        import lal
        ts2 = type(ts).from_lal(lalts)
        self.assertEqual(ts, ts2)
        # test copy=False
        ts2 = type(ts).from_lal(lalts, copy=False)
        self.assertEqual(ts, ts2)
        # test bad unit
        ts.override_unit('undef')
        with pytest.warns(UserWarning):
            lalts = ts.to_lal()
        self.assertEqual(lalts.sampleUnits, lal.DimensionlessUnit)
        ts2 = self.TEST_CLASS.from_lal(lalts)
        self.assertIs(ts2.unit, units.dimensionless_unscaled)

    def test_io_identify(self):
        common.test_io_identify(self.TEST_CLASS, ['txt', 'hdf5', 'gwf'])

    def test_fetch(self):
        try:
            nds_buffer = mockutils.mock_nds2_buffer(
                'X1:TEST', self.data, 1000000000, self.data.shape[0], 'm')
        except ImportError as e:
            self.skipTest(str(e))
        nds_connection = mockutils.mock_nds2_connection([nds_buffer])
        with mock.patch('nds2.connection') as mock_connection, \
             mock.patch('nds2.buffer', nds_buffer):
            mock_connection.return_value = nds_connection
            # use verbose=True to hit more lines
            ts = TimeSeries.fetch('X1:TEST', 1000000000, 1000000001,
                                  verbose=True)
        nptest.assert_array_equal(ts.value, self.data)
        self.assertEqual(ts.sample_rate, self.data.shape[0] * units.Hz)
        self.assertTupleEqual(ts.span, (1000000000, 1000000001))
        self.assertEqual(ts.unit, units.meter)

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
            with open(tmpfile, 'wb') as f:
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
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            plot.save(f.name)
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
        self.assertNotAlmostEqual(self.random.value.mean(), 0.0)
        detrended = self.random.detrend()
        self.assertAlmostEqual(detrended.value.mean(), 0.0)

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

    def test_boolean_statetimeseries(self):
        comp = self.TEST_ARRAY >= 100 * self.TEST_ARRAY.unit
        self.assertIsInstance(comp, StateTimeSeries)
        self.assertEqual(comp.unit, units.Unit(''))
        self.assertEqual(
            comp.name,
            '%s >= 100.0 %s' % (self.TEST_ARRAY.name, self.TEST_ARRAY.unit))

    def test_rms(self):
        rms = self.TEST_ARRAY.rms(1.)
        self.assertQuantityEqual(rms.sample_rate, 1 * units.Hertz)


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

    def test_read_write_gwf(self):
        try:
            a = self.TEST_CLASS.read(TEST_GWF_FILE, self.channels)
        except ImportError as e:
            self.skipTest(str(e))
        for c in self.channels:
            self.assertIn(c, a)
        channels = list(map(Channel, self.channels))
        a = self.TEST_CLASS.read(TEST_GWF_FILE, channels)
        for c in channels:
            self.assertIn(c, a)
        # test write
        try:
            with tempfile.NamedTemporaryFile(suffix='.gwf', delete=False) as f:
                a.write(f.name)
                b = self.TEST_CLASS.read(f.name, a.keys())
        finally:
            if os.path.exists(f.name):
                os.remove(f.name)
        self.assertDictEqual(a, b)

    def test_plot(self):
        tsd = self.read()
        plot = tsd.plot()
        self.assertIsInstance(plot, TimeSeriesPlot)
        self.assertEqual(len(plot.gca().lines), 2)
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            plot.save(f.name)

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
        with tempfile.NamedTemporaryFile(suffix='.png') as f:
            plot.save(f.name)


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
