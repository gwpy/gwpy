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

"""Unit tests for `gwpy.table`
"""

import os.path
import re
from io import BytesIO
from unittest import mock

import pytest

from numpy import (random, isclose, dtype, asarray, all)

import h5py

from astropy import units
from astropy.io.ascii import InconsistentTableError
from astropy.table import vstack

from ...frequencyseries import FrequencySeries
from ...io import ligolw as io_ligolw
from ...segments import (Segment, SegmentList)
from ...testing import utils
from ...testing.errors import pytest_skip_network_error
from ...time import LIGOTimeGPS
from ...timeseries import (TimeSeries, TimeSeriesDict)
from .. import (Table, EventTable, filters)
from ..filter import filter_table
from ..io.hacr import HACR_COLUMNS
from ..io.root import _import_uproot_that_can_write_root_files

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')
TEST_XML_FILE = os.path.join(
    TEST_DATA_DIR, 'H1-LDAS_STRAIN-968654552-10.xml.gz')
TEST_OMEGA_FILE = os.path.join(TEST_DATA_DIR, 'omega.txt')

try:
    uproot = _import_uproot_that_can_write_root_files()
except ImportError:
    HAVE_UPROOT_RW = False
else:
    HAVE_UPROOT_RW = True
SKIP_UPROOT_RW = pytest.mark.skipif(
    not HAVE_UPROOT_RW,
    reason="uproot>4 doesn't support r/w and no uproot3",
)


# -- mocks --------------------------------------------------------------------

def mock_hacr_connection(table, start, stop):
    """Mock a pymysql connection object to test HACR fetching
    """
    # create cursor
    cursor = mock.MagicMock()

    def execute(qstr):
        cursor._query = qstr
        return len(table)

    cursor.execute = execute
    column_regex = re.compile(r"\Aselect (.*) from ", re.I)
    select_regex = re.compile(r"where (.*) (order by .*)?\Z", re.I)

    def fetchall():
        q = cursor._query
        if "from job" in q:
            return [(1, start, stop)]
        if "from mhacr" in q:
            columns = column_regex.match(q).groups()[0].split(", ")
            selections = (
                select_regex.search(q).groups()[0].strip().split(" and ")
            )
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
    def emptytable(cls):
        return cls.create(0, ['time', 'snr', 'frequency'])

    @classmethod
    @pytest.fixture()
    def clustertable(cls):
        return cls.TABLE(data=[[11, 1, 1, 10, 1, 1, 9],
                               [0.0, 1.9, 1.95, 2.0, 2.05, 2.1, 4.0]],
                         names=['amplitude', 'time'])

    # -- test I/O -------------------------------

    @utils.skip_missing_dependency('ligo.lw.lsctables')
    @pytest.mark.parametrize('ext', ['xml', 'xml.gz'])
    def test_read_write_ligolw(self, tmp_path, ext):
        table = self.create(
            100, ['peak_time', 'peak_time_ns', 'snr', 'central_freq'],
            ['i4', 'i4', 'f4', 'f4'])
        tmp = tmp_path / "table.{}".format(ext)

        def _read(*args, **kwargs):
            kwargs.setdefault('format', 'ligolw')
            kwargs.setdefault('tablename', 'sngl_burst')
            return self.TABLE.read(tmp, *args, **kwargs)

        def _write(*args, **kwargs):
            kwargs.setdefault('format', 'ligolw')
            kwargs.setdefault('tablename', 'sngl_burst')
            return table.write(tmp, *args, **kwargs)

        # check simple write (using open file descriptor, not file path)
        with tmp.open("w+b") as f:
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
        t3 = self.TABLE.read([tmp, tmp], format='ligolw',
                             tablename='sngl_burst')
        utils.assert_table_equal(vstack((t2, t2)), t3)

        # check writing to existing file raises IOError
        with pytest.raises(IOError) as exc:
            _write()
        assert str(exc.value) == 'File exists: %s' % tmp

        # check overwrite=True, append=False rewrites table
        _write(overwrite=True)
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

    @utils.skip_missing_dependency('ligo.lw.lsctables')
    def test_read_write_ligolw_ilwdchar_compat(self, tmp_path):
        from glue.ligolw.ilwd import get_ilwdchar_class
        from glue.ligolw.lsctables import SnglBurstTable

        eid_type = get_ilwdchar_class("sngl_burst", "event_id")

        table = self.create(
            100,
            ["peak", "snr", "central_freq", "event_id"],
            ["f8", "f4", "f4", "i8"],
        )

        # write table with ilwdchar_compat=True
        tmp = tmp_path / "tmp.xml"
        table.write(
            tmp,
            format="ligolw",
            tablename="sngl_burst",
            ilwdchar_compat=True,
        )

        # read raw ligolw and check type is correct
        llw = io_ligolw.read_table(
            tmp,
            tablename="sngl_burst",
            ilwdchar_compat=True,
        )
        assert type(llw.getColumnByName("event_id")[0]) is eid_type

        # reset IDs to 0
        SnglBurstTable.reset_next_id()

        # read without explicit use of ilwdchar_compat
        t2 = self.TABLE.read(tmp, columns=table.colnames)
        assert type(t2[0]["event_id"]) is eid_type

        # read again with explicit use of ilwdchar_compat
        SnglBurstTable.reset_next_id()
        utils.assert_table_equal(
            self.TABLE.read(
                tmp,
                columns=table.colnames,
                ilwdchar_compat=True,
            ),
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
    def test_read_write_ligolw_property_columns(self, tmp_path):
        table = self.create(100, ['peak', 'snr', 'central_freq'],
                            ['f8', 'f4', 'f4'])

        # write table
        tmp = tmp_path / "test.xml"
        table.write(tmp, format='ligolw', tablename='sngl_burst')

        # read raw ligolw and check gpsproperty was unpacked properly
        llw = io_ligolw.read_table(tmp, tablename='sngl_burst')
        for col in ('peak_time', 'peak_time_ns'):
            assert col in llw.columnnames
        with io_ligolw.patch_ligotimegps():
            utils.assert_array_equal(
                asarray([row.peak for row in llw]),
                table['peak'],
            )

        # read table and assert gpsproperty was repacked properly
        t2 = self.TABLE.read(
            tmp,
            columns=table.colnames,
            use_numpy_dtypes=True,
        )
        utils.assert_table_equal(t2, table, almost_equal=True)

    @utils.skip_missing_dependency('ligo.lw.lsctables')
    @pytest.mark.parametrize('ilwdchar_compat', (False, True))
    def test_read_ligolw_get_as_exclude(self, tmp_path, ilwdchar_compat):
        table = self.TABLE(
            rows=[
                ("H1", 0.0, 4, 0),
                ("L1", 0.62831, 4, 0),
                ("V1", 0.31415, 4, 0),
            ],
            names=("instrument", "offset", "process_id", "time_slide_id"),
        )

        # write a file
        tmp = tmp_path / "test.xml"
        table.write(
            tmp,
            format="ligolw",
            tablename="time_slide",
            ilwdchar_compat=ilwdchar_compat,
        )

        # read it back and assert that we the `instrument` table is
        # read properly
        t2 = table.read(
            tmp,
            tablename="time_slide",
            use_numpy_dtypes=ilwdchar_compat,
        )
        t2.sort("instrument")
        utils.assert_table_equal(t2, table)

    @SKIP_UPROOT_RW
    def test_read_write_root(self, table, tmp_path):
        tmp = tmp_path / "table.root"

        # check write
        table.write(tmp)

        # check read gives back same table
        t2 = self.TABLE.read(tmp)
        utils.assert_table_equal(table, t2)

        # test selections work
        segs = SegmentList([Segment(100, 200), Segment(400, 500)])
        t2 = self.TABLE.read(
            tmp,
            selection=['200 < frequency < 500',
                       ('time', filters.in_segmentlist, segs)],
        )
        utils.assert_table_equal(
            t2,
            filter_table(
                table,
                'frequency > 200',
                'frequency < 500',
                ('time', filters.in_segmentlist, segs),
            ),
        )

    @SKIP_UPROOT_RW
    def test_write_root_overwrite(self, table, tmp_path):
        tmp = tmp_path / "table.root"
        table.write(tmp)

        # assert failure with overwrite=False (default)
        with pytest.raises(OSError):
            table.write(tmp)

        # assert works with overwrite=True
        table.write(tmp, overwrite=True)

    @SKIP_UPROOT_RW
    def test_read_root_multiple_trees(self, table, tmp_path):
        tmp = tmp_path / "table.root"
        with uproot.create(tmp) as root:
            root["a"] = uproot.newtree({"branch": "int32"})
            root["a"].extend({"branch": asarray([1, 2, 3, 4, 5])})
            root["b"] = uproot.newtree()
        with pytest.raises(ValueError) as exc:
            self.TABLE.read(tmp)
        assert str(exc.value).startswith('Multiple trees found')
        self.TABLE.read(tmp, treename="a")

    @SKIP_UPROOT_RW
    def test_read_write_root_append(self, table, tmp_path):
        tmp = tmp_path / "table.root"
        # append hasn't been implemented in uproot 3 yet
        with pytest.raises(NotImplementedError):
            table.write(tmp, treename="test2", append=True)

    @utils.skip_missing_dependency('LDAStools.frameCPP')
    def test_read_write_gwf(self, tmp_path):
        table = self.create(100, ['time', 'blah', 'frequency'])
        columns = table.dtype.names
        tmp = tmp_path / "table.gwf"

        # check write
        try:
            table.write(tmp, 'test_read_write_gwf')
        except TypeError as exc:  # pragma: no-cover
            msg = str(exc).splitlines()[0]
            for err, ref in [
                ("ParamList_type", 57),
                ("Unable to translate parameter to Parameters_type", 146),
            ]:
                if err in msg:
                    pytest.skip(
                        f"bug in python-ldas-tools-framecpp: '{msg}', "
                        "see https://git.ligo.org/computing/ldastools/"
                        f"LDAS_Tools/-/issues/{ref}",
                    )
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
        """Check that `_get_time_colum` works on name
        """
        assert table._get_time_column() == "time"

    def test_get_time_column_case(self, table):
        """Check that `_get_time_colum` works on name case-insensitively
        """
        table.rename_column("time", "TiMe")
        assert table._get_time_column() == 'TiMe'

    def test_get_time_column_gps_type(self):
        """Check that `_get_time_column` works on dtype
        """
        # check that single GPS column can be identified
        t = self.create(1, ("a", "b"), dtypes=(float, LIGOTimeGPS))
        assert t._get_time_column() == "b"

    def test_get_time_column_error_no_match(self, table):
        """Check that `_get_time_column` raises the right exception
        when no matches are found
        """
        t = self.create(1, ("a",))
        with pytest.raises(ValueError) as exc:
            t._get_time_column()
        assert " no columns named 'time'" in str(exc.value)

    def test_get_time_column_error_multiple_match(self):
        # check that two GPS columns causes issues
        t = self.create(
            10,
            ("a", "b", "c"),
            dtypes=(float, LIGOTimeGPS, LIGOTimeGPS),
        )
        with pytest.raises(ValueError) as exc:
            t._get_time_column()
        assert " multiple columns named 'time'" in str(exc.value)

    def test_get_time_column_error_empty(self):
        """Check that `_get_time_column` errors properly on an empty table
        """
        t = self.create(0, ("a",))
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
        """Test :meth:`gwpy.table.EventTable.event_rate`
        """
        rate = table.event_rate(1)
        assert isinstance(rate, TimeSeries)
        assert rate.sample_rate == 1 * units.Hz

    @utils.skip_missing_dependency('lal')
    def test_event_rates_gpsobject(self, table):
        """Test that `EventTable.event_rate` can handle object dtypes
        """
        rate = table.event_rate(1)

        from lal import LIGOTimeGPS as LalGps
        lgps = list(map(LalGps, table['time']))
        t2 = type(table)(data=[lgps], names=['time'])
        rate2 = t2.event_rate(1, start=table['time'].min())

        utils.assert_quantity_sub_equal(rate, rate2)

    def test_event_rates_start_end(self):
        """Check that `EventTable.event_rate` can function without explicit time column
        (and no data) if and only if start/end are both given
        """
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

    def test_cluster_multiple(self, clustertable):
        # check that after clustering a table, clustering the table a
        # second time with the same parameters returns the same result
        t_clustered = clustertable.cluster('time', 'amplitude', 0.6)
        utils.assert_table_equal(
            t_clustered,
            t_clustered.cluster('time', 'amplitude', 0.6),
        )

    def test_cluster_empty(self, emptytable):
        # check that clustering an empty table is a no-op
        t = emptytable.cluster('time', 'amplitude', 0.6)
        utils.assert_table_equal(t, emptytable)

    # -- test I/O -------------------------------

    def test_read_write_hdf5(self, table, tmp_path):
        # check that our overrides of astropy's H5 reader
        # didn't break everything
        tmp = tmp_path / "table.h5"
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
    def test_read_write_ascii(self, table, tmp_path, fmtname):
        fmt = 'ascii.{}'.format(fmtname.lower())
        tmp = tmp_path / "table.txt"

        # check write/read returns the same table
        with tmp.open("w") as fobj:
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
    def test_read_write_ascii_error(self, table, tmp_path, fmtname):
        tmp = tmp_path / "table.txt"
        with tmp.open("w"):
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

    @pytest.fixture
    def pycbclivetable(self):
        return self.create(
            100,
            names=[
                'a',
                'b',
                'c',
                'chisq',
                'd',
                'e',
                'f',
                'mass1',
                'mass2',
                'snr',
            ],
        )

    @staticmethod
    @pytest.fixture
    def pycbclivepsd():
        return FrequencySeries(random.randn(1000), df=1)

    @pytest.fixture
    def pycbclivefile(self, tmp_path, pycbclivetable, pycbclivepsd):
        # create table
        loudest = (pycbclivetable['snr'] > 500).nonzero()[0]

        # manually create pycbc_live-format HDF5 file
        tmp = tmp_path / "X1-Live-0-0.hdf"
        # write table in pycbc_live format (by hand)
        with h5py.File(tmp, "w") as h5f:
            group = h5f.create_group('X1')
            for col in pycbclivetable.columns:
                group.create_dataset(data=pycbclivetable[col], name=col)
            group.create_dataset('loudest', data=loudest)
            group.create_dataset('psd', data=pycbclivepsd.value)
            group['psd'].attrs['delta_f'] = pycbclivepsd.df.to('Hz').value
        return tmp

    def test_read_pycbc_live(self, pycbclivetable, pycbclivefile):
        """Check that we can read a PyCBC-Live file
        """
        table = self.TABLE.read(pycbclivefile, format="hdf5.pycbc_live")
        utils.assert_table_equal(pycbclivetable, table)
        assert table.meta['ifo'] == 'X1'

    def test_read_pycbc_live_kwargs(self, pycbclivetable, pycbclivefile):
        """Check that we can read a PyCBC-Live file using keywords
        """
        table = self.TABLE.read(
            pycbclivefile,
            format='hdf5.pycbc_live',
            ifo='X1',
        )
        utils.assert_table_equal(pycbclivetable, table)

    def test_read_pycbc_live_loudest(self, pycbclivetable, pycbclivefile):
        """Check that we can read the `loudest` table from a PyCBC-Live file
        """
        table = self.TABLE.read(
            pycbclivefile,
            format="hdf5.pycbc_live",
            loudest=True,
        )
        utils.assert_table_equal(pycbclivetable.filter('snr > 500'), table)

    def test_read_pycbc_live_extended_metadata(
            self,
            pycbclivetable,
            pycbclivepsd,
            pycbclivefile,
    ):
        """Check that we can read extended metadata from a PyCBC-Live file
        """
        table = self.TABLE.read(
            pycbclivefile,
            format="hdf5.pycbc_live",
            extended_metadata=True,  # default
        )
        utils.assert_table_equal(pycbclivetable, table)
        utils.assert_array_equal(
            table.meta['loudest'],
            (pycbclivetable['snr'] > 500).nonzero()[0],
        )
        utils.assert_quantity_sub_equal(
            table.meta['psd'],
            pycbclivepsd,
            exclude=['name', 'channel', 'unit', 'epoch'])

    def test_read_pycbc_live_extended_metadata_false(
            self,
            pycbclivetable,
            pycbclivefile,
    ):
        """Check that we can read a PyCBC-Live file without extended metadata
        """
        # check extended_metadata=False works
        table = self.TABLE.read(
            pycbclivefile,
            format="hdf5.pycbc_live",
            extended_metadata=False,
        )
        assert table.meta == {'ifo': 'X1'}

    def test_read_pycbc_live_multiple_ifos(
            self,
            pycbclivetable,
            pycbclivefile,
    ):
        """Check that we can handle multiple IFOs in a PyCBC-Live file
        """
        with h5py.File(pycbclivefile, "r+") as h5f:
            h5f.create_group('Z1')
        with pytest.raises(ValueError) as exc:
            self.TABLE.read(pycbclivefile, format="hdf5.pycbc_live")
        assert str(exc.value).startswith(
            'PyCBC live HDF5 file contains dataset groups')

        # but check that we can still read the original
        table = self.TABLE.read(
            pycbclivefile,
            format='hdf5.pycbc_live',
            ifo='X1',
        )
        utils.assert_table_equal(pycbclivetable, table)

    def test_read_pycbc_live_processed_columns(
            self,
            pycbclivetable,
            pycbclivefile,
    ):
        """Check that we can read processed columns from a PyCBC-Live file
        """
        # assert processed colums works
        table = self.TABLE.read(
            pycbclivefile,
            format="hdf5.pycbc_live",
            ifo="X1",
            columns=["mchirp", "new_snr"],
        )
        mchirp = (
            (pycbclivetable['mass1'] * pycbclivetable['mass2']) ** (3/5.)
            / (pycbclivetable['mass1'] + pycbclivetable['mass2']) ** (1/5.)
        )
        utils.assert_array_equal(table['mchirp'], mchirp)

    def test_read_pycbc_live_selection_columns(
            self,
            pycbclivetable,
            pycbclivefile,
    ):
        """Check that the selection and columns kwargs work when
        reading from a PyCBC-Live file
        """
        # test with selection and columns
        table = self.TABLE.read(
            pycbclivefile,
            format='hdf5.pycbc_live',
            ifo='X1',
            selection='snr>.5',
            columns=("a", "b", "mass1"),
        )
        utils.assert_table_equal(
            table,
            filter_table(pycbclivetable, 'snr>.5')[("a", "b", "mass1")],
        )

    def test_read_pycbc_live_regression_1081(
            self,
            pycbclivetable,
            pycbclivefile,
    ):
        """Check against regression of gwpy/gwpy#1081
        """
        table = self.TABLE.read(
            pycbclivefile,
            format='hdf5.pycbc_live',
            ifo='X1',
            selection='snr>.5',
            columns=("a", "b", "snr"),
        )
        utils.assert_table_equal(
            table,
            filter_table(pycbclivetable, 'snr>.5')[("a", "b", "snr")],
        )

    @pytest.fixture
    def snaxtable(self):
        channel = 'H1:FAKE'
        table = self.create(
            100,
            names=[
                'time',
                'snr',
                'frequency',
            ],
        )
        table["channel"] = channel
        return table

    @staticmethod
    @pytest.fixture
    def snaxfile(snaxtable, tmp_path):
        tmp = tmp_path / "SNAX-0-0.h5"
        channel = snaxtable[0]["channel"]
        tmptable = snaxtable.copy()
        tmptable.remove_column("channel")
        with h5py.File(tmp, "w") as h5f:
            group = h5f.create_group(channel)
            group.create_dataset(data=tmptable, name='0.0_20.0')
        return tmp

    def test_read_snax(self, snaxtable, snaxfile):
        """Check that we can read a SNAX-format HDF5 file
        """
        table = self.TABLE.read(snaxfile, format='hdf5.snax')
        utils.assert_table_equal(snaxtable, table)

    def test_read_snax_channel(self, snaxtable, snaxfile):
        """Check that we can read a SNAX-format HDF5 file specifying
        the channel
        """
        table = self.TABLE.read(
            snaxfile,
            format='hdf5.snax',
            channels="H1:FAKE",
        )
        utils.assert_table_equal(snaxtable, table)

    def test_read_snax_selection_columns(self, snaxtable, snaxfile):
        """Check that the selection and columns kwargs work when
        reading from a SNAX-format file
        """
        # test with selection and columns
        table = self.TABLE.read(
            snaxfile,
            channels="H1:FAKE",
            format='hdf5.snax',
            selection='snr>.5',
            columns=('time', 'snr'),
        )
        utils.assert_table_equal(
            table,
            filter_table(snaxtable, 'snr>.5')[('time', 'snr')],
        )

    def test_read_snax_compact(self, snaxtable, snaxfile):
        """Check that the selection and columns kwargs work when
        reading from a SNAX-format file
        """
        # test compact representation of channel column
        t2 = self.TABLE.read(snaxfile, compact=True, format='hdf5.snax')

        # group by channel and drop channel column
        tables = {}
        t2 = t2.group_by('channel')
        t2.remove_column('channel')
        for key, group in zip(t2.groups.keys, t2.groups):
            channel = t2.meta['channel_map'][key['channel']]
            tables[channel] = self.TABLE(group, copy=True)

        # verify table groups are identical
        t_ref = snaxtable.copy().group_by('channel')
        t_ref.remove_column('channel')
        for key, group in zip(t_ref.groups.keys, t_ref.groups):
            channel = key['channel']
            utils.assert_table_equal(group, tables[channel])

    def test_read_snax_errors(self, snaxtable, snaxfile):
        """Check error handling when reading from a SNAX-format file
        """
        missing = ["H1:FAKE", 'H1:MISSING']
        with pytest.raises(ValueError):
            self.TABLE.read(snaxfile, channels=missing, format='hdf5.snax')

        with pytest.warns(UserWarning):
            table = self.TABLE.read(
                snaxfile,
                channels=missing,
                format='hdf5.snax',
                on_missing='warn',
            )
        utils.assert_table_equal(snaxtable, table)

    @pytest.fixture(scope="module")
    def hacr_table(self):
        """Create a table of HACR-like data, and patch
        `pymysql.connect` to return it
        """
        table = self.create(100, names=HACR_COLUMNS)
        connect = mock.patch(
            "pymysql.connect",
            return_value=mock_hacr_connection(table, 123, 456),
        )
        try:
            connect.start()
        except ImportError as exc:  # pragma: no-cover
            pytest.skip(str(exc))
        yield table
        connect.stop()

    def test_fetch_hacr(self, hacr_table):
        t2 = self.TABLE.fetch('hacr', 'X1:TEST-CHANNEL', 123, 456)

        # check type matches
        assert type(t2) is self.TABLE

        # check response is correct
        utils.assert_table_equal(hacr_table, t2)

    def test_fetch_hacr_columns(self, hacr_table):
        t2 = self.TABLE.fetch('hacr', 'X1:TEST-CHANNEL', 123, 456,
                              columns=['gps_start', 'snr'],
                              selection='freq_central>500')
        utils.assert_table_equal(
            filter_table(hacr_table, 'freq_central>500')['gps_start', 'snr'],
            t2,
        )

    @pytest_skip_network_error
    def test_fetch_open_data(self):
        table = self.TABLE.fetch_open_data("GWTC-1-confident")
        assert len(table)
        assert {
            "mass_1_source",
            "luminosity_distance",
            "chi_eff"
        }.intersection(table.colnames)
        # check unit parsing worked
        assert table["luminosity_distance"].unit == "Mpc"

    @pytest_skip_network_error
    def test_fetch_open_data_kwargs(self):
        table = self.TABLE.fetch_open_data(
            "GWTC-1-confident",
            selection="mass_1_source < 5",
            columns=[
                "name",
                "mass_1_source",
                "mass_2_source",
                "luminosity_distance"
            ],
        )
        assert len(table) == 1
        assert table[0]["name"] == "GW170817-v3"
        assert set(table.colnames) == {
            "name",
            "mass_1_source",
            "mass_2_source",
            "luminosity_distance"
        }
