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

"""Unit tests for `gwpy.table`.

Note that tests of the I/O integrations are separately maintained
in the ``test_io_{format}.py`` modules alongside this one.
"""

import os.path
from io import BytesIO

import pytest
from astropy import units
from numpy.testing import assert_array_equal

from ...segments import (
    Segment,
    SegmentList,
)
from ...testing import utils
from ...time import LIGOTimeGPS
from ...timeseries import (
    TimeSeries,
    TimeSeriesDict,
)
from .. import (
    EventTable,
    Table,
    filters,
)
from .utils import random_table

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')
TEST_XML_FILE = os.path.join(
    TEST_DATA_DIR, 'H1-LDAS_STRAIN-968654552-10.xml.gz')
TEST_OMEGA_FILE = os.path.join(TEST_DATA_DIR, 'omega.txt')


# -- gwpy.table.Table (astropy.table.Table) -----------------------------------

class TestTable:
    TABLE = Table

    @classmethod
    def create(cls, n, names, dtypes=None):
        return random_table(
            names=names,
            length=n,
            dtypes=dtypes,
            table_class=cls.TABLE,
        )

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
        with pytest.raises(
            ValueError,
            match=" no columns named 'time'",
        ):
            t._get_time_column()

    def test_get_time_column_error_multiple_match(self):
        # check that two GPS columns causes issues
        t = self.create(
            10,
            ("a", "b", "c"),
            dtypes=(float, LIGOTimeGPS, LIGOTimeGPS),
        )
        with pytest.raises(
            ValueError,
            match=" multiple columns named 'time'",
        ):
            t._get_time_column()

    def test_get_time_column_error_empty(self):
        """Check that `_get_time_column` errors properly on an empty table
        """
        t = self.create(0, ("a",))
        with pytest.raises(ValueError):
            t._get_time_column()

    def test_filter(self, table):
        """Test that `EventTable.filter` works with a simple filter statement
        """
        # check simple filter
        lowf = table.filter('frequency < 100')
        assert isinstance(lowf, type(table))
        assert len(lowf) == 11
        assert lowf['frequency'].max() == pytest.approx(96.5309156606)

    def test_filter_empty(self, table):
        """Test that `EventTable.filter` works with an empty table
        """
        assert len(table.filter('snr>5', 'snr<=5')) == 0

    def test_filter_chaining(self, table):
        """Test that chaining filters works with `EventTable.filter`
        """
        loud = table.filter('snr > 100')
        lowf = table.filter('frequency < 100')
        lowfloud = table.filter('frequency < 100', 'snr > 100')
        brute = type(table)(
            rows=[tuple(row) for row in lowf if row in loud],
            names=table.dtype.names,
        )
        utils.assert_table_equal(brute, lowfloud)

    def test_filter_range(self, table):
        """Test that `EventTable.filter` works with a range statement
        """
        # check double-ended filter
        midf = table.filter('100 < frequency < 1000')
        utils.assert_table_equal(
            midf,
            table.filter('frequency > 100').filter('frequency < 1000'),
        )

    def test_filter_function(self, table):
        """Test that `EventTable.filter` works with a filter function
        """
        def my_filter(column, threshold):
            return column < threshold

        lowf = table.filter(("frequency", my_filter, 100))
        assert len(lowf) == 11

    def test_filter_function_multiple(self, table):
        """Test that `EventTable.filter` works with a filter function
        that requires multiple columns
        """
        def my_filter(table, threshold):
            return table["snr"] * table["frequency"] > threshold

        filtered = table.filter((("snr", "frequency"), my_filter, 100000))
        assert len(filtered) == 64

    def test_filter_in_segmentlist(self, table):
        """Test `EventTable.filter` with `in_segmentlist`
        """
        # check filtering on segments works
        segs = SegmentList([Segment(100, 200), Segment(400, 500)])
        inseg = table.filter(('time', filters.in_segmentlist, segs))
        brute = type(table)(
            rows=[tuple(row) for row in table if row['time'] in segs],
            names=table.colnames,
        )
        utils.assert_table_equal(inseg, brute)

    def test_filter_in_segmentlist_empty(self, table):
        """Test `EventTable.filter` with `in_segmentlist` and an empty table
        """
        # check empty segmentlist is handled well
        utils.assert_table_equal(
            table.filter(('time', filters.in_segmentlist, SegmentList())),
            type(table)(names=table.colnames),
        )

    def test_filter_not_in_segmentlist(self, table):
        """Test `EventTable.filter` with `not_in_segmentlist`
        """
        segs = SegmentList([Segment(100, 200), Segment(400, 500)])
        notsegs = SegmentList([Segment(0, 1000)]) - segs
        inseg = table.filter(('time', filters.in_segmentlist, segs))
        utils.assert_table_equal(
            inseg, table.filter(('time', filters.not_in_segmentlist, notsegs)),
        )
        utils.assert_table_equal(
            table,
            table.filter(('time', filters.not_in_segmentlist, SegmentList())),
        )

    def test_event_rates(self, table):
        """Test :meth:`gwpy.table.EventTable.event_rate`
        """
        rate = table.event_rate(1)
        assert isinstance(rate, TimeSeries)
        assert rate.sample_rate == 1 * units.Hz

    @pytest.mark.requires("lal")
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
        """Check that `EventTable.event_rate` can function without explicit
        time column (and no data) if and only if start/end are both given.
        """
        t2 = self.create(10, names=['a', 'b'])
        with pytest.raises(
            ValueError,
            match="please give `timecolumn` keyword",
        ):
            t2.event_rate(1)
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
        with pytest.raises(
            ValueError,
            match="please give `timecolumn` keyword",
        ):
            t2.binned_event_rates(1, 'a', (10, 100))
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
        assert_array_equal(t['amplitude'], [11, 10, 9])
        assert_array_equal(t['time'], [0.0, 2.0, 4.0])

    def test_single_point_cluster(self, clustertable):
        # check that a large cluster window returns at least one data point
        t = clustertable.cluster('time', 'amplitude', 10)
        assert len(t) == 1
        assert_array_equal(t['amplitude'], [11])
        assert_array_equal(t['time'], [0.0])

    def test_cluster_window(self, clustertable):
        # check that a non-positive window throws an appropriate ValueError
        with pytest.raises(
            ValueError,
            match="^Window must be a positive value$",
        ):
            clustertable.cluster('time', 'amplitude', 0)

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
