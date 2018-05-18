# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Unit tests for `gwpy.table`
"""

import os.path
import shutil
import tempfile
from ssl import SSLError

from six import PY2
from six.moves import StringIO
from six.moves.urllib.error import URLError

import pytest

import sqlparse

from numpy import (random, isclose, dtype)

import h5py

from matplotlib import use, rc_context
use('agg')  # nopep8

from astropy import units
from astropy.io.ascii import InconsistentTableError
from astropy.table import vstack

from gwpy.frequencyseries import FrequencySeries
from gwpy.io import ligolw as io_ligolw
from gwpy.segments import (Segment, SegmentList)
from gwpy.table import (Table, EventTable, filters, GravitySpyTable)
from gwpy.table.filter import filter_table
from gwpy.table.io.hacr import (HACR_COLUMNS, get_hacr_triggers)
from gwpy.time import LIGOTimeGPS
from gwpy.timeseries import (TimeSeries, TimeSeriesDict)
from gwpy.plotter import (EventTablePlot, EventTableAxes, TimeSeriesPlot,
                          HistogramPlot)

import utils
from mocks import mock

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')
TEST_XML_FILE = os.path.join(
    TEST_DATA_DIR, 'H1-LDAS_STRAIN-968654552-10.xml.gz')
TEST_OMEGA_FILE = os.path.join(TEST_DATA_DIR, 'omega.txt')
TEST_JSON_RESPONSE_FILE = os.path.join(TEST_DATA_DIR, 'test_json_query.json')


# -- mocks --------------------------------------------------------------------

def mock_hacr_connection(table, start, stop):
    """Mock a pymysql connection object to test HACR fetching
    """
    # create cursor
    cursor = mock.MagicMock()

    def execute(qstr):
        cursor._query = sqlparse.parse(qstr)[0]
        return len(table)

    cursor.execute = execute

    def fetchall():
        if cursor._query.get_real_name() == 'job':
            return [(1, start, stop)]
        if cursor._query.get_real_name() == 'mhacr':
            columns = list(map(
                str, list(cursor._query.get_sublists())[0].get_identifiers()))
            selections = list(map(
                str, list(cursor._query.get_sublists())[2].get_sublists()))
            return filter_table(table, selections[3:])[columns]

    cursor.fetchall = fetchall

    # create connection
    conn = mock.MagicMock()
    conn.cursor.return_value = cursor
    return conn


# -- gwpy.table.Table (astropy.table.Table) -----------------------------------

class TestTable(object):
    TABLE = Table

    @classmethod
    def create(cls, n, names, dtypes=None):
        data = []
        for i, name in enumerate(names):
            random.seed(i)
            if dtypes:
                dtype = dtypes[i]
            else:
                dtype = None
            data.append((random.rand(n) * 1000).astype(dtype))
        return cls.TABLE(data, names=names)

    @classmethod
    @pytest.fixture()
    def table(cls):
        return cls.create(100, ['time', 'snr', 'frequency'])

    # -- test I/O -------------------------------

    @utils.skip_missing_dependency('glue.ligolw.lsctables')
    @pytest.mark.parametrize('ext', ['xml', 'xml.gz'])
    def test_read_write_ligolw(self, ext):
        table = self.create(
            100, ['peak_time', 'peak_time_ns', 'snr', 'central_freq'],
            ['i4', 'i4', 'f4', 'f4'])
        with tempfile.NamedTemporaryFile(suffix='.{}'.format(ext),
                                         delete=False) as f:
            def _read(*args, **kwargs):
                kwargs.setdefault('format', 'ligolw')
                kwargs.setdefault('tablename', 'sngl_burst')
                return self.TABLE.read(f, *args, **kwargs)

            def _write(*args, **kwargs):
                kwargs.setdefault('format', 'ligolw')
                kwargs.setdefault('tablename', 'sngl_burst')
                return table.write(f.name, *args, **kwargs)

            # check simple write (using open file descriptor, not file path)
            table.write(f, format='ligolw', tablename='sngl_burst')

            # check simple read
            t2 = _read()
            utils.assert_table_equal(table, t2, almost_equal=True)
            assert t2.meta.get('tablename', None) == 'sngl_burst'

            # check auto-discovery of 'time' columns works
            from glue.ligolw.lsctables import LIGOTimeGPS
            with pytest.warns(DeprecationWarning):
                t3 = _read(columns=['time'])
            assert 'time' in t3.columns
            assert isinstance(t3[0]['time'], LIGOTimeGPS)
            utils.assert_array_equal(
                t3['time'], table['peak_time'] + table['peak_time_ns'] * 1e-9)

            # check numpy type casting works
            with pytest.warns(DeprecationWarning):
                t3 = _read(columns=['time'], use_numpy_dtypes=True)
            assert t3['time'].dtype == dtype('float64')
            utils.assert_array_equal(
                t3['time'], table['peak_time'] + table['peak_time_ns'] * 1e-9)

            # check reading multiple tables works
            try:
                t3 = self.TABLE.read([f.name, f.name], format='ligolw',
                                     tablename='sngl_burst')
            except NameError as e:
                if not PY2:  # ligolw not patched for python3 just yet
                    pytest.xfail(str(e))
                raise
            utils.assert_table_equal(vstack((t2, t2)), t3)

            # check writing to existing file raises IOError
            with pytest.raises(IOError) as exc:
                _write()
            assert str(exc.value) == 'File exists: %s' % f.name

            # check overwrite=True, append=False rewrites table
            try:
                _write(overwrite=True)
            except TypeError as e:
                # ligolw is not python3-compatbile, so skip if it fails
                if not PY2 and (
                        str(e) == 'write() argument must be str, not bytes'):
                    pytest.xfail(str(e))
                raise
            t3 = _read()
            utils.assert_table_equal(t2, t3)

            # check append=True duplicates table
            _write(append=True)
            t3 = _read()
            utils.assert_table_equal(vstack((t2, t2)), t3)

            # check overwrite=True, append=True rewrites table
            _write(append=True, overwrite=True)
            t3 = _read()
            utils.assert_table_equal(t2, t3)

            # write another table and check we can still get back the first
            insp = self.create(10, ['end_time', 'snr', 'chisq_dof'])
            insp.write(f.name, format='ligolw', tablename='sngl_inspiral',
                       append=True)
            t3 = _read()
            utils.assert_table_equal(t2, t3)

            # write another table with append=False and check the first table
            # is gone
            insp.write(f.name, format='ligolw', tablename='sngl_inspiral',
                       append=False, overwrite=True)
            with pytest.raises(ValueError) as exc:
                _read()
            assert str(exc.value) == ('document must contain exactly '
                                      'one sngl_burst table')

            # -- deprecations
            # check deprecations print warnings where expected

            with pytest.warns(DeprecationWarning):
                table.write(f.name, format='ligolw.sngl_burst', overwrite=True)
            with pytest.warns(DeprecationWarning):
                _read(format='ligolw.sngl_burst')
            with pytest.warns(DeprecationWarning):
                _read(get_as_columns=True)
            with pytest.warns(DeprecationWarning):
                _read(on_attributeerror='anything')

    @utils.skip_missing_dependency('glue.ligolw.lsctables')
    def test_read_write_ligolw_property_columns(self):
        table = self.create(100, ['peak', 'snr', 'central_freq'],
                            ['f8', 'f4', 'f4'])
        with tempfile.NamedTemporaryFile(suffix='.xml') as f:
            # write table
            table.write(f, format='ligolw', tablename='sngl_burst')

            # read raw ligolw and check gpsproperty was unpacked properly
            llw = io_ligolw.read_table(f, tablename='sngl_burst')
            for col in ('peak_time', 'peak_time_ns'):
                assert col in llw.columnnames
            with io_ligolw.patch_ligotimegps():
                utils.assert_array_equal(llw.get_peak(), table['peak'])

            # read table and assert gpsproperty was repacked properly
            t2 = self.TABLE.read(f, columns=table.colnames,
                                 use_numpy_dtypes=True)
            utils.assert_table_equal(t2, table, almost_equal=True)

    @utils.skip_missing_dependency('root_numpy')
    def test_read_write_root(self, table):
        tempdir = tempfile.mkdtemp()
        try:
            fp = tempfile.mktemp(suffix='.root', dir=tempdir)

            # check write
            table.write(fp)

            def _read(*args, **kwargs):
                return type(table).read(fp, *args, **kwargs)

            # check read gives back same table
            utils.assert_table_equal(table, _read())

            # check that reading table from file with multiple trees without
            # specifying fails
            table.write(fp, treename='test')
            with pytest.raises(ValueError) as exc:
                _read()
            assert str(exc.value).startswith('Multiple trees found')

            # test selections work
            segs = SegmentList([Segment(100, 200), Segment(400, 500)])
            t2 = _read(treename='test',
                       selection=['200 < frequency < 500',
                                  ('time', filters.in_segmentlist, segs)])
            utils.assert_table_equal(
                t2, filter_table(table,
                                 'frequency > 200',
                                 'frequency < 500',
                                 ('time', filters.in_segmentlist, segs)),
            )

        finally:
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir)

    def test_read_write_gwf(self):
        table = self.create(100, ['time', 'blah', 'frequency'])
        columns = table.dtype.names
        tempdir = tempfile.mkdtemp()
        try:
            fp = tempfile.mktemp(suffix='.gwf', dir=tempdir)

            # check write
            table.write(fp, 'test_read_write_gwf')

            # check read gives back same table
            t2 = self.TABLE.read(fp, 'test_read_write_gwf', columns=columns)
            utils.assert_table_equal(table, t2, meta=False, almost_equal=True)

            # check selections works
            t3 = self.TABLE.read(fp, 'test_read_write_gwf',
                                 columns=columns, selection='frequency>500')
            utils.assert_table_equal(
                filter_table(t2, 'frequency>500'), t3)

        except ImportError as e:
            pytest.skip(str(e))
        finally:
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir)


class TestEventTable(TestTable):
    TABLE = EventTable

    def test_filter(self, table):
        # check simple filter
        lowf = table.filter('frequency < 100')
        assert isinstance(lowf, type(table))
        assert len(lowf) == 11
        assert isclose(lowf['frequency'].max(), 96.5309156606)

        # check filtering everything returns an empty table
        assert len(table.filter('snr>5', 'snr<=5')) == 0

        # check compounding works
        loud = table.filter('snr > 100')
        lowfloud = table.filter('frequency < 100', 'snr > 100')
        brute = type(table)(rows=[row for row in lowf if row in loud],
                            names=table.dtype.names)
        utils.assert_table_equal(brute, lowfloud)

        # check double-ended filter
        midf = table.filter('100 < frequency < 1000')
        utils.assert_table_equal(
            midf, table.filter('frequency > 100').filter('frequency < 1000'))

        # check unicode parsing (PY2)
        loud2 = table.filter(u'snr > 100')

    def test_filter_in_segmentlist(self, table):
        print(table)
        # check filtering on segments works
        segs = SegmentList([Segment(100, 200), Segment(400, 500)])
        inseg = table.filter(('time', filters.in_segmentlist, segs))
        brute = type(table)(rows=[row for row in table if row['time'] in segs],
                            names=table.colnames)
        utils.assert_table_equal(inseg, brute)

        # check empty segmentlist is handled well
        utils.assert_table_equal(
            table.filter(('time', filters.in_segmentlist, SegmentList())),
            type(table)(names=table.colnames))

        # check inverse works
        notsegs = SegmentList([Segment(0, 1000)]) - segs
        utils.assert_table_equal(
            inseg, table.filter(('time', filters.not_in_segmentlist, notsegs)))
        utils.assert_table_equal(
            table,
            table.filter(('time', filters.not_in_segmentlist, SegmentList())))

    def test_event_rates(self, table):
        rate = table.event_rate(1)
        assert isinstance(rate, TimeSeries)
        assert rate.sample_rate == 1 * units.Hz

        # repeat with object dtype
        try:
            from lal import LIGOTimeGPS
        except ImportError:
            return
        lgps = list(map(LIGOTimeGPS, table['time']))
        t2 = type(table)(data=[lgps], names=['time'])
        rate2 = t2.event_rate(1, start=table['time'].min())
        utils.assert_quantity_sub_equal(rate, rate2)

    def test_binned_event_rates(self, table):
        rates = table.binned_event_rates(100, 'snr', [10, 100],
                                         timecolumn='time')
        assert isinstance(rates, TimeSeriesDict)
        assert list(rates.keys()), [10, 100]
        assert rates[10].max() == 0.14 * units.Hz
        assert rates[10].name == 'snr >= 10'
        assert rates[100].max() == 0.13 * units.Hz
        assert rates[100].name == 'snr >= 100'
        table.binned_event_rates(100, 'snr', [10, 100], operator='in')
        table.binned_event_rates(100, 'snr', [(0, 10), (10, 100)])

    def test_plot(self, table):
        with rc_context(rc={'text.usetex': False}):
            plot = table.plot('time', 'frequency', color='snr')
            assert isinstance(plot, EventTablePlot)
            assert isinstance(plot, TimeSeriesPlot)
            assert isinstance(plot.gca(), EventTableAxes)
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)

    def test_hist(self, table):
        with rc_context(rc={'text.usetex': False}):
            plot = table.hist('snr')
            assert isinstance(plot, HistogramPlot)
            assert len(plot.gca().patches) == 10
            with tempfile.NamedTemporaryFile(suffix='.png') as f:
                plot.save(f.name)

    def test_get_column(self, table):
        utils.assert_array_equal(table.get_column('snr'), table['snr'])

    # -- test I/O -------------------------------

    @pytest.mark.parametrize('fmtname', ('Omega', 'cWB'))
    def test_read_write_ascii(self, table, fmtname):
        fmt = 'ascii.%s' % fmtname.lower()
        with tempfile.NamedTemporaryFile(suffix='.txt', mode='w') as f:
            print(f.name)
            # check write/read returns the same table
            table.write(f, format=fmt)
            f.seek(0)
            utils.assert_table_equal(table, self.TABLE.read(f, format=fmt),
                                     almost_equal=True)

        with tempfile.NamedTemporaryFile(suffix='.txt') as f:
            # assert reading blank file doesn't work with column name error
            with pytest.raises(InconsistentTableError) as exc:
                self.TABLE.read(f, format=fmt)
            assert str(exc.value) == ('No column names found in %s header'
                                      % fmtname)

    def test_read_pycbc_live(self):
        table = self.create(
            100, names=['a', 'b', 'c', 'chisq', 'd', 'e', 'f',
                        'mass1', 'mass2', 'snr'])
        loudest = (table['snr'] > 500).nonzero()[0]
        psd = FrequencySeries(random.randn(1000), df=1)
        fp = os.path.join(tempfile.mkdtemp(), 'X1-Live-0-0.hdf')
        try:
            # write table in pycbc_live format (by hand)
            with h5py.File(fp, 'w') as h5f:
                group = h5f.create_group('X1')
                for col in table.columns:
                    group.create_dataset(data=table[col], name=col)
                group.create_dataset('loudest', data=loudest)
                group.create_dataset('psd', data=psd.value)
                group['psd'].attrs['delta_f'] = psd.df.to('Hz').value

            # check that we can read
            t2 = self.TABLE.read(fp)
            utils.assert_table_equal(table, t2)
            # and check metadata was recorded correctly
            assert t2.meta['ifo'] == 'X1'

            # check keyword arguments result in same table
            t2 = self.TABLE.read(fp, format='hdf5.pycbc_live')
            utils.assert_table_equal(table, t2)
            t2 = self.TABLE.read(fp, format='hdf5.pycbc_live', ifo='X1')

            # assert loudest works
            t2 = self.TABLE.read(fp, loudest=True)
            utils.assert_table_equal(table.filter('snr > 500'), t2)

            # check extended_metadata=True works (default)
            t2 = self.TABLE.read(fp, extended_metadata=True)
            utils.assert_table_equal(table, t2)
            utils.assert_array_equal(t2.meta['loudest'], loudest)
            utils.assert_quantity_sub_equal(
                t2.meta['psd'], psd,
                exclude=['name', 'channel', 'unit', 'epoch'])

            # check extended_metadata=False works
            t2 = self.TABLE.read(fp, extended_metadata=False)
            assert t2.meta == {'ifo': 'X1'}

            # double-check that loudest and extended_metadata=False work
            t2 = self.TABLE.read(fp, loudest=True, extended_metadata=False)
            utils.assert_table_equal(table.filter('snr > 500'), t2)
            assert t2.meta == {'ifo': 'X1'}

            # add another IFO, then assert that reading the table without
            # specifying the IFO fails
            with h5py.File(fp) as h5f:
                h5f.create_group('Z1')
            with pytest.raises(ValueError) as exc:
                self.TABLE.read(fp)
            assert str(exc.value).startswith(
                'PyCBC live HDF5 file contains dataset groups')

            # but check that we can still read the original
            t2 = self.TABLE.read(fp, format='hdf5.pycbc_live', ifo='X1')
            utils.assert_table_equal(table, t2)

            # assert processed colums works
            t2 = self.TABLE.read(fp, ifo='X1', columns=['mchirp', 'new_snr'])
            mchirp = (table['mass1'] * table['mass2']) ** (3/5.) / (
                table['mass1'] + table['mass2']) ** (1/5.)
            utils.assert_array_equal(t2['mchirp'], mchirp)

            # test with selection
            t2 = self.TABLE.read(fp, format='hdf5.pycbc_live',
                                 ifo='X1', selection='snr>.5')
            utils.assert_table_equal(filter_table(table, 'snr>.5'), t2)
        finally:
            if os.path.isdir(os.path.dirname(fp)):
                shutil.rmtree(os.path.dirname(fp))

    def test_fetch_hacr(self):
        table = self.create(100, names=HACR_COLUMNS)
        try:
            from pymysql import connect
        except ImportError:
            mockee = 'gwpy.table.io.hacr.connect'
        else:
            mockee = 'pymysql.connect'
        with mock.patch(mockee) as mock_connect:
            mock_connect.return_value = mock_hacr_connection(
                table, 123, 456)

            # test simple query returns the full table
            t2 = self.TABLE.fetch('hacr', 'X1:TEST-CHANNEL', 123, 456)
            utils.assert_table_equal(table, t2)

            # test column selection works
            t2 = self.TABLE.fetch('hacr', 'X1:TEST-CHANNEL', 123, 456,
                                  columns=['gps_start', 'snr'])
            utils.assert_table_equal(table['gps_start', 'snr'], t2)

            # test column selection works
            t2 = self.TABLE.fetch('hacr', 'X1:TEST-CHANNEL', 123, 456,
                                  columns=['gps_start', 'snr'],
                                  selection='freq_central>500')
            utils.assert_table_equal(
                filter_table(table, 'freq_central>500')['gps_start', 'snr'],
                t2)


@utils.skip_minimum_version('astropy', '2.0.4')
class TestGravitySpyTable(TestEventTable):
    TABLE = GravitySpyTable

    def test_search(self):
        try:
            t2 = self.TABLE.search(uniqueID="8FHTgA8MEu", howmany=1)
        except (URLError, SSLError) as e:
            pytest.skip(str(e))

        import json
        with open(TEST_JSON_RESPONSE_FILE) as f:
            table = GravitySpyTable(json.load(f))

        utils.assert_table_equal(table, t2)
