# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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
from io import BytesIO
from ssl import SSLError

from six import PY2
from six.moves.urllib.error import URLError

import pytest

import sqlparse

from numpy import (random, isclose, dtype, asarray, all)
from numpy.testing import assert_array_equal
from numpy.ma.core import MaskedConstant

import h5py

from astropy import units
from astropy.io.ascii import InconsistentTableError
from astropy.table import vstack

from ...frequencyseries import FrequencySeries
from ...io import ligolw as io_ligolw
from ...segments import (Segment, SegmentList)
from ...testing import utils
from ...testing.compat import mock
from ...time import LIGOTimeGPS
from ...timeseries import (TimeSeries, TimeSeriesDict)
from .. import (Table, EventTable, filters)
from ..filter import filter_table
from ..io.hacr import HACR_COLUMNS

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')
TEST_XML_FILE = os.path.join(
    TEST_DATA_DIR, 'H1-LDAS_STRAIN-968654552-10.xml.gz')
TEST_OMEGA_FILE = os.path.join(TEST_DATA_DIR, 'omega.txt')


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
            return map(tuple, filter_table(table, selections[3:])[columns])

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
                dtp = dtypes[i]
            else:
                dtp = None
            # use map() to support non-primitive types
            if dtype(dtp).name == 'object':
                data.append(list(map(dtp, random.rand(n) * 1000)))
            else:
                data.append((random.rand(n) * 1000).astype(dtp))
        return cls.TABLE(data, names=names)

    @classmethod
    @pytest.fixture()
    def table(cls):
        return cls.create(100, ['time', 'snr', 'frequency'])

    @classmethod
    @pytest.fixture()
    def clustertable(cls):
        return cls.TABLE(data=[[11, 1, 1, 10, 1, 1, 9],
                               [0.0, 1.9, 1.95, 2.0, 2.05, 2.1, 4.0]],
                         names=['amplitude', 'time'])

    @staticmethod
    @pytest.fixture()
    def llwtable():
        from ligo.lw.lsctables import (New, SnglBurstTable)
        llwtab = New(SnglBurstTable, columns=["peak_frequency", "snr"])
        for i in range(10):
            row = llwtab.RowType()
            row.peak_frequency = float(i)
            row.snr = float(i)
            llwtab.append(row)
        return llwtab

    # -- test I/O -------------------------------

    @utils.skip_missing_dependency('ligo.lw.lsctables')
    def test_ligolw(self, llwtable):
        tab = self.TABLE(llwtable)
        assert set(tab.colnames) == {"peak_frequency", "snr"}
        assert_array_equal(tab["snr"], llwtable.getColumnByName("snr"))

    @utils.skip_missing_dependency('ligo.lw.lsctables')
    def test_ligolw_rename(self, llwtable):
        tab = self.TABLE(llwtable, rename={"peak_frequency": "frequency"})
        assert set(tab.colnames) == {"frequency", "snr"}
        assert_array_equal(
            tab["frequency"],
            llwtable.getColumnByName("peak_frequency"),
        )

    @utils.skip_missing_dependency('ligo.lw.lsctables')
    @pytest.mark.parametrize('ext', ['xml', 'xml.gz'])
    def test_read_write_ligolw(self, ext):
        table = self.create(
            100, ['peak_time', 'peak_time_ns', 'snr', 'central_freq'],
            ['i4', 'i4', 'f4', 'f4'])
        with utils.TemporaryFilename(suffix='.{}'.format(ext)) as tmp:
            def _read(*args, **kwargs):
                kwargs.setdefault('format', 'ligolw')
                kwargs.setdefault('tablename', 'sngl_burst')
                return self.TABLE.read(tmp, *args, **kwargs)

            def _write(*args, **kwargs):
                kwargs.setdefault('format', 'ligolw')
                kwargs.setdefault('tablename', 'sngl_burst')
                return table.write(tmp, *args, **kwargs)

            # check simple write (using open file descriptor, not file path)
            with open(tmp, 'w+b') as f:
                table.write(f, format='ligolw', tablename='sngl_burst')

            # check simple read
            t2 = _read()
            utils.assert_table_equal(table, t2, almost_equal=True)
            assert t2.meta.get('tablename', None) == 'sngl_burst'

            # check numpy type casting works
            from ligo.lw.lsctables import LIGOTimeGPS as LigolwGPS
            t3 = _read(columns=['peak'])
            assert isinstance(t3['peak'][0], LigolwGPS)
            t3 = _read(columns=['peak'], use_numpy_dtypes=True)
            assert t3['peak'].dtype == dtype('float64')
            utils.assert_array_equal(
                t3['peak'], table['peak_time'] + table['peak_time_ns'] * 1e-9)

            # check reading multiple tables works
            try:
                t3 = self.TABLE.read([tmp, tmp], format='ligolw',
                                     tablename='sngl_burst')
            except NameError as e:
                if not PY2:  # ligolw not patched for python3 just yet
                    pytest.xfail(str(e))
                raise
            utils.assert_table_equal(vstack((t2, t2)), t3)

            # check writing to existing file raises IOError
            with pytest.raises(IOError) as exc:
                _write()
            assert str(exc.value) == 'File exists: %s' % tmp

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
            insp.write(tmp, format='ligolw', tablename='sngl_inspiral',
                       append=True)
            t3 = _read()
            utils.assert_table_equal(t2, t3)

            # write another table with append=False and check the first table
            # is gone
            insp.write(tmp, format='ligolw', tablename='sngl_inspiral',
                       append=False, overwrite=True)
            with pytest.raises(ValueError) as exc:
                _read()
            assert str(exc.value) == ('document must contain exactly '
                                      'one sngl_burst table')

    @utils.skip_missing_dependency('glue.ligolw.lsctables')
    def test_read_write_ligolw_ilwdchar_compat(self):
        from glue.ligolw.ilwd import get_ilwdchar_class
        from glue.ligolw.lsctables import SnglBurstTable

        eid_type = get_ilwdchar_class("sngl_burst", "event_id")

        table = self.create(
            100,
            ["peak", "snr", "central_freq", "event_id"],
            ["f8", "f4", "f4", "i8"],
        )
        with tempfile.NamedTemporaryFile(suffix=".xml") as tmp:
            # write table with ilwdchar_compat=True
            table.write(tmp, format="ligolw", tablename="sngl_burst",
                        ilwdchar_compat=True)

            # read raw ligolw and check type is correct
            llw = io_ligolw.read_table(tmp, tablename="sngl_burst",
                                       ilwdchar_compat=True)
            assert type(llw.getColumnByName("event_id")[0]) is eid_type

            # reset IDs to 0
            SnglBurstTable.reset_next_id()

            # read without explicit use of ilwdchar_compat
            t2 = self.TABLE.read(tmp, columns=table.colnames)
            assert type(t2[0]["event_id"]) is eid_type

            # read again with explicit use of ilwdchar_compat
            SnglBurstTable.reset_next_id()
            utils.assert_table_equal(
                self.TABLE.read(tmp, columns=table.colnames,
                                ilwdchar_compat=True),
                t2,
            )

            # and check that ilwdchar_compat=True, use_numpy_dtypes=True works
            SnglBurstTable.reset_next_id()
            utils.assert_table_equal(
                self.TABLE.read(tmp, columns=table.colnames,
                                ilwdchar_compat=True, use_numpy_dtypes=True),
                table,
                almost_equal=True,
            )

    @utils.skip_missing_dependency('ligo.lw.lsctables')
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
                utils.assert_array_equal(
                    asarray([row.peak for row in llw]),
                    table['peak'],
                )

            # read table and assert gpsproperty was repacked properly
            t2 = self.TABLE.read(f, columns=table.colnames,
                                 use_numpy_dtypes=True)
            utils.assert_table_equal(t2, table, almost_equal=True)

    @utils.skip_missing_dependency('root_numpy')
    def test_read_write_root(self, table):
        with utils.TemporaryFilename(suffix='.root') as tmp:
            # check write
            table.write(tmp)

            def _read(*args, **kwargs):
                return type(table).read(tmp, *args, **kwargs)

            # check read gives back same table
            utils.assert_table_equal(table, _read())

            # check that reading table from file with multiple trees without
            # specifying fails
            table.write(tmp, treename='test')
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

    def test_read_write_gwf(self):
        table = self.create(100, ['time', 'blah', 'frequency'])
        columns = table.dtype.names
        with utils.TemporaryFilename(suffix='.gwf') as tmp:
            # check write
            try:
                table.write(tmp, 'test_read_write_gwf')
            except ImportError as exc:  # no frameCPP
                pytest.skip(str(exc))
            except TypeError as exc:  # frameCPP broken (2.6.7)
                if 'ParamList' in str(exc):
                    pytest.skip("bug in python-ldas-tools-framecpp")
                raise

            # check read gives back same table
            t2 = self.TABLE.read(tmp, 'test_read_write_gwf', columns=columns)
            utils.assert_table_equal(table, t2, meta=False, almost_equal=True)

            # check selections works
            t3 = self.TABLE.read(tmp, 'test_read_write_gwf',
                                 columns=columns, selection='frequency>500')
            utils.assert_table_equal(
                filter_table(t2, 'frequency>500'), t3)


class TestEventTable(TestTable):
    TABLE = EventTable

    def test_get_time_column(self, table):
        with pytest.raises(ValueError) as exc:
            self.TABLE()._get_time_column()
        assert str(exc.value).startswith('cannot identify time column')

        # table from fixture has 'time' column
        assert table._get_time_column() == 'time'

        # check that single GPS column can be identified
        t = self.create(10, ('blah', 'blah2'), dtypes=(float, LIGOTimeGPS))
        assert t._get_time_column() == 'blah2'

        # check that two GPS columns causes issues
        try:
            t.add_column(t['blah2'], name='blah3')
        except TypeError:  # astropy < 2.0 (or something like that)
            pass
        else:
            with pytest.raises(ValueError):
                t._get_time_column()

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
        brute = type(table)(
            rows=[tuple(row) for row in lowf if row in loud],
            names=table.dtype.names,
        )
        utils.assert_table_equal(brute, lowfloud)

        # check double-ended filter
        midf = table.filter('100 < frequency < 1000')
        utils.assert_table_equal(
            midf, table.filter('frequency > 100').filter('frequency < 1000'))

        # check unicode parsing (PY2)
        table.filter(u'snr > 100')

    def test_filter_in_segmentlist(self, table):
        # check filtering on segments works
        segs = SegmentList([Segment(100, 200), Segment(400, 500)])
        inseg = table.filter(('time', filters.in_segmentlist, segs))
        brute = type(table)(
            rows=[tuple(row) for row in table if row['time'] in segs],
            names=table.colnames,
        )
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
            from lal import LIGOTimeGPS as LalGps
        except ImportError:
            return
        lgps = list(map(LalGps, table['time']))
        t2 = type(table)(data=[lgps], names=['time'])
        rate2 = t2.event_rate(1, start=table['time'].min())
        utils.assert_quantity_sub_equal(rate, rate2)

        # check that method can function without explicit time column
        # (and no data) if and only if start/end are both given
        t2 = self.create(10, names=['a', 'b'])
        with pytest.raises(ValueError) as exc:
            t2.event_rate(1)
        assert 'please give `timecolumn` keyword' in str(exc.value)
        with pytest.raises(ValueError):
            t2.event_rate(1, start=0)
        with pytest.raises(ValueError):
            t2.event_rate(1, end=1)
        t2.event_rate(1, start=0, end=10)

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

        # check that method can function without explicit time column
        # (and no data) if and only if start/end are both given
        t2 = self.create(0, names=['a', 'b'])
        with pytest.raises(ValueError) as exc:
            t2.binned_event_rates(1, 'a', (10, 100))
        assert 'please give `timecolumn` keyword' in str(exc.value)
        with pytest.raises(ValueError):
            t2.binned_event_rates(1, 'a', (10, 100), start=0)
        with pytest.raises(ValueError):
            t2.binned_event_rates(1, 'a', (10, 100), end=1)
        t2.binned_event_rates(1, 'a', (10, 100), start=0, end=10)

    def test_plot(self, table):
        with pytest.deprecated_call():
            plot = table.plot('time', 'frequency', color='snr')
            plot.close()

    def test_scatter(self, table):
        plot = table.scatter('time', 'frequency', color='snr')
        plot.save(BytesIO(), format='png')
        plot.close()

    def test_hist(self, table):
        plot = table.hist('snr')
        assert len(plot.gca().patches) == 10
        plot.save(BytesIO(), format='png')
        plot.close()

    def test_get_column(self, table):
        utils.assert_array_equal(table.get_column('snr'), table['snr'])

    def test_cluster(self, clustertable):
        # check that the central data points are all clustered away,
        # the original table is unchanged, and all points return their
        # intended values
        t = clustertable.cluster('time', 'amplitude', 0.6)
        assert len(t) == 3
        assert len(clustertable) == 7
        assert all(t['amplitude'] == [11, 10, 9])
        assert all(t['time'] == [0.0, 2.0, 4.0])

    def test_single_point_cluster(self, clustertable):
        # check that a large cluster window returns at least one data point
        t = clustertable.cluster('time', 'amplitude', 10)
        assert len(t) == 1
        assert all(t['amplitude'] == [11])
        assert all(t['time'] == [0.0])

    def test_cluster_window(self, clustertable):
        # check that a non-positive window throws an appropriate ValueError
        with pytest.raises(ValueError) as exc:
            clustertable.cluster('time', 'amplitude', 0)
        assert str(exc.value) == 'Window must be a positive value'

    # -- test I/O -------------------------------

    def test_read_write_hdf5(self, table):
        # check that our overrides of astropy's H5 reader
        # didn't break everything
        with utils.TemporaryFilename(suffix=".h5") as tmp:
            table.write(tmp, path="/data")
            t2 = self.TABLE.read(tmp, path="/data")
            utils.assert_table_equal(t2, table)

            t2 = self.TABLE.read(
                tmp,
                path="/data",
                selection="frequency>500",
                columns=["time", "snr"],
            )
            utils.assert_table_equal(
                t2,
                filter_table(table, "frequency>500")[("time", "snr")],
            )

    @pytest.mark.parametrize('fmtname', ('Omega', 'cWB'))
    def test_read_write_ascii(self, table, fmtname):
        fmt = 'ascii.{}'.format(fmtname.lower())
        with utils.TemporaryFilename(suffix='.txt') as tmp:
            # check write/read returns the same table
            with open(tmp, 'w') as fobj:
                table.write(fobj, format=fmt)
            t2 = self.TABLE.read(tmp, format=fmt)
            utils.assert_table_equal(table, t2, almost_equal=True)

            # check that we can use selections and column filtering
            t2 = self.TABLE.read(
                tmp,
                format=fmt,
                selection="frequency>500",
                columns=["time", "snr"],
            )
            utils.assert_table_equal(
                t2,
                filter_table(table, "frequency>500")[("time", "snr")],
                almost_equal=True,
            )

    @pytest.mark.parametrize('fmtname', ('Omega', 'cWB'))
    def test_read_write_ascii_error(self, table, fmtname):
        with utils.TemporaryFilename(suffix='.txt') as tmp:
            with open(tmp, 'w'):
                pass  # write empty file
            # assert reading blank file doesn't work with column name error
            with pytest.raises(InconsistentTableError) as exc:
                self.TABLE.read(
                    tmp,
                    format="ascii.{}".format(fmtname.lower()),
                )
            assert str(exc.value) == (
                'No column names found in {} header'.format(fmtname)
            )

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
            t2 = self.TABLE.read(fp, format="hdf5.pycbc_live")
            utils.assert_table_equal(table, t2)
            # and check metadata was recorded correctly
            assert t2.meta['ifo'] == 'X1'

            # check keyword arguments result in same table
            t2 = self.TABLE.read(fp, format='hdf5.pycbc_live', ifo='X1')
            utils.assert_table_equal(table, t2)

            # assert loudest works
            t2 = self.TABLE.read(fp, format="hdf5.pycbc_live", loudest=True)
            utils.assert_table_equal(table.filter('snr > 500'), t2)

            # check extended_metadata=True works (default)
            t2 = self.TABLE.read(
                fp,
                format="hdf5.pycbc_live",
                extended_metadata=True,
            )
            utils.assert_table_equal(table, t2)
            utils.assert_array_equal(t2.meta['loudest'], loudest)
            utils.assert_quantity_sub_equal(
                t2.meta['psd'], psd,
                exclude=['name', 'channel', 'unit', 'epoch'])

            # check extended_metadata=False works
            t2 = self.TABLE.read(
                fp,
                format="hdf5.pycbc_live",
                extended_metadata=False,
            )
            assert t2.meta == {'ifo': 'X1'}

            # double-check that loudest and extended_metadata=False work
            t2 = self.TABLE.read(
                fp,
                format="hdf5.pycbc_live",
                loudest=True,
                extended_metadata=False,
            )
            utils.assert_table_equal(table.filter('snr > 500'), t2)
            assert t2.meta == {'ifo': 'X1'}

            # add another IFO, then assert that reading the table without
            # specifying the IFO fails
            with h5py.File(fp) as h5f:
                h5f.create_group('Z1')
            with pytest.raises(ValueError) as exc:
                self.TABLE.read(fp, format="hdf5.pycbc_live")
            assert str(exc.value).startswith(
                'PyCBC live HDF5 file contains dataset groups')

            # but check that we can still read the original
            t2 = self.TABLE.read(fp, format='hdf5.pycbc_live', ifo='X1')
            utils.assert_table_equal(table, t2)

            # assert processed colums works
            t2 = self.TABLE.read(
                fp,
                format="hdf5.pycbc_live",
                ifo="X1",
                columns=["mchirp", "new_snr"],
            )
            mchirp = (table['mass1'] * table['mass2']) ** (3/5.) / (
                table['mass1'] + table['mass2']) ** (1/5.)
            utils.assert_array_equal(t2['mchirp'], mchirp)

            # test with selection and columns
            t2 = self.TABLE.read(
                fp,
                format='hdf5.pycbc_live',
                ifo='X1',
                selection='snr>.5',
                columns=("a", "b", "mass1"),
            )
            utils.assert_table_equal(
                t2,
                filter_table(table, 'snr>.5')[("a", "b", "mass1")],
            )

            # regression test: gwpy/gwpy#1081
            t2 = self.TABLE.read(
                fp,
                format='hdf5.pycbc_live',
                ifo='X1',
                selection='snr>.5',
                columns=("a", "b", "snr"),
            )
            utils.assert_table_equal(
                t2,
                filter_table(table, 'snr>.5')[("a", "b", "snr")],
            )

        finally:
            if os.path.isdir(os.path.dirname(fp)):
                shutil.rmtree(os.path.dirname(fp))

    def test_read_snax(self):
        table = self.create(
            100, names=['time', 'snr', 'frequency'])
        fp = os.path.join(tempfile.mkdtemp(), 'SNAX-0-0.h5')
        try:
            # write table in snax format (by hand)
            with h5py.File(fp, 'w') as h5f:
                group = h5f.create_group('H1:FAKE')
                group.create_dataset(data=table, name='0.0_20.0')

            # check that we can read
            t2 = self.TABLE.read(fp, 'H1:FAKE', format='hdf5.snax')
            utils.assert_table_equal(table, t2)

            # test with selection and columns
            t2 = self.TABLE.read(
                fp,
                'H1:FAKE',
                format='hdf5.snax',
                selection='snr>.5',
                columns=('time', 'snr'),
            )
            utils.assert_table_equal(
                t2,
                filter_table(table, 'snr>.5')[('time', 'snr')],
            )

        finally:
            if os.path.isdir(os.path.dirname(fp)):
                shutil.rmtree(os.path.dirname(fp))

    def test_fetch_hacr(self):
        table = self.create(100, names=HACR_COLUMNS)
        try:
            from pymysql import connect  # noqa: F401
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

    def test_fetch_open_data(self):
        try:
            table = self.TABLE.fetch_open_data("GWTC-1-confident")
        except (URLError, SSLError) as exc:
            pytest.skip(str(exc))
        assert len(table)
        assert {"L_peak", "distance", "mass1"}.intersection(table.colnames)
        # check unit parsing worked
        assert table["distance"].unit == "Mpc"
        # check that masking worked (needs table to be sorted)
        gw170818 = table.loc["GW170818"]
        assert isinstance(gw170818["snr_pycbc"], MaskedConstant)

    def test_fetch_open_data_kwargs(self):
        try:
            table = self.TABLE.fetch_open_data(
                "GWTC-1-confident",
                selection="mass1 < 5",
                columns=["name", "mass1", "mass2", "distance"])
        except (URLError, SSLError) as exc:
            pytest.skip(str(exc))
        assert len(table) == 1
        assert table[0]["name"] == "GW170817"
        assert set(table.colnames) == {"name", "mass1", "mass2", "distance"}
