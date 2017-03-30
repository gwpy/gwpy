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

from numpy import (may_share_memory, testing as nptest, random)

from matplotlib import use
use('agg')

from astropy import units

from gwpy.table import (Table, EventTable)
from gwpy.timeseries import (TimeSeries, TimeSeriesDict)

import common
from compat import unittest

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')
TEST_XML_FILE = os.path.join(
    TEST_DATA_DIR, 'H1-LDAS_STRAIN-968654552-10.xml.gz')
TEST_OMEGA_FILE = os.path.join(TEST_DATA_DIR, 'omega.txt')


class TableTests(unittest.TestCase):
    TABLE_CLASS = Table

    def assertTableEqual(self, a, b, copy=None, meta=False):
        assert a.colnames == b.colnames
        nptest.assert_array_equal(a.as_array(), b.as_array())
        if meta:
            assert a.meta == b.meta
        for col, col2 in zip(a.columns.values(), b.columns.values()):
            if copy:
                assert not may_share_memory(col, col2)
            elif copy is False:
                assert may_share_memory(col, col2)

    def assertTableEqualTypeless(self, a, b, copy=None, meta=False):
        assert a.colnames == b.colnames
        for col, col2 in zip(a.columns.values(), b.columns.values()):
            nptest.assert_array_equal(col, col2.astype(col.dtype))
        if meta:
            assert a.meta == b.meta

    def test_read_write_ligolw(self):
        table = self.TABLE_CLASS.read(TEST_XML_FILE,
                                      format='ligolw.sngl_burst')
        self.assertIsInstance(table, self.TABLE_CLASS)
        self.assertIsInstance(table['snr'], self.TABLE_CLASS.Column)
        self.assertEqual(len(table), 2052)
        self.assertAlmostEqual(table[0]['snr'], 0.69409615)
        # try multiple files
        table2 = self.TABLE_CLASS.read([TEST_XML_FILE, TEST_XML_FILE],
                                       format='ligolw.sngl_burst')
        self.assertEqual(len(table2), 4104)
        self.assertEqual(table2[0]['snr'], table2[2052]['snr'])
        # try with columns
        table4 = self.TABLE_CLASS.read(
            TEST_XML_FILE, format='ligolw.sngl_burst',
            columns=['time', 'snr', 'central_freq'])
        self.assertListEqual(sorted(table4.dtype.names),
                             ['central_freq', 'snr', 'time'])
        self.assertEqual(
            table[0]['peak_time'] + table[0]['peak_time_ns'] * 1e-9,
            table4[0]['time'])

        # test write
        tempdir = tempfile.mkdtemp()
        try:
            fp = tempfile.mktemp(suffix='.xml', dir=tempdir)
            # write fresh
            table.write(fp, format='ligolw.sngl_burst')
            table5 = self.TABLE_CLASS.read(fp, format='ligolw.sngl_burst')
            self.assertTableEqual(table, table5)
            # assert existing file raises IOError
            with self.assertRaises(IOError) as exc:
                table.write(fp, format='ligolw.sngl_burst')
            self.assertEqual(str(exc.exception), 'File exists: %s' % fp)
            # overwrite=True, append=False
            table.write(fp, format='ligolw.sngl_burst', overwrite=True)
            table5 = self.TABLE_CLASS.read(fp, format='ligolw.sngl_burst')
            self.assertTableEqual(table, table5)
            # overwrite=False, append=True
            table.write(fp, format='ligolw.sngl_burst', append=True)
            table5 = self.TABLE_CLASS.read(fp, format='ligolw.sngl_burst')
            self.assertTableEqual(table2, table5)
            # overwrite=True, append=True
            table.write(fp, format='ligolw.sngl_burst', append=True,
                        overwrite=True)
            table5 = self.TABLE_CLASS.read(fp, format='ligolw.sngl_burst')
            self.assertTableEqual(table, table5)
            # append a different table and check we still have the first
            p = self.TABLE_CLASS.read(TEST_XML_FILE, format='ligolw.process')
            p.write(fp, format='ligolw.process', append=True)
            table5 = self.TABLE_CLASS.read(fp, format='ligolw.sngl_burst')
            self.assertTableEqual(table, table5)
            # append=False and check we don't still have the first
            p.write(fp, format='ligolw.process', append=False, overwrite=True)
            with self.assertRaises(ValueError) as exc:
                self.TABLE_CLASS.read(fp, format='ligolw.sngl_burst')
            self.assertEqual(
                str(exc.exception),
                'document must contain exactly one sngl_burst table')
        finally:
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir)

    def test_read_write_root(self):
        table = self.TABLE_CLASS.read(
            TEST_XML_FILE, format='ligolw.sngl_burst',
            columns=['peak_time', 'peak_time_ns', 'snr', 'peak_frequency'])
        tempdir = tempfile.mkdtemp()
        try:
            fp = tempfile.mktemp(suffix='.root', dir=tempdir)
            # test read
            table.write(fp)
            # test read gives back same table
            table2 = self.TABLE_CLASS.read(fp)
            self.assertTableEqual(table, table2, meta=False)
            # test writing a second table then reading without tree= raises
            # ValueError
            table.write(fp, treename='test')
            with self.assertRaises(ValueError) as exc:
                self.TABLE_CLASS.read(fp)
            self.assertTrue(str(exc.exception).startswith(
                "Multiple trees found"))
        except ImportError as e:
            self.skipTest(str(e))
        finally:
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir)

    def test_read_write_gwf(self):
        table = self.TABLE_CLASS.read(
            TEST_XML_FILE, format='ligolw.sngl_burst',
            columns=['peak_time', 'peak_time_ns', 'snr', 'peak_frequency'])
        # add expected columns for GWF
        time = table['peak_time'] + table['peak_time_ns'] * 1e-9
        time.name = 'time'
        table.add_column(time)
        # test read/write
        columns = table.dtype.names
        tempdir = tempfile.mkdtemp()
        try:
            fp = tempfile.mktemp(suffix='.gwf', dir=tempdir)
            # test read
            table.write(fp, 'test_read_write_gwf')
            # test read gives back same table
            table2 = self.TABLE_CLASS.read(fp, 'test_read_write_gwf',
                                           columns=columns)
            self.assertTableEqualTypeless(table, table2, meta=False)
        except ImportError as e:
            self.skipTest(str(e))
        finally:
            if os.path.isdir(tempdir):
                shutil.rmtree(tempdir)

class EventTableTests(TableTests):
    TABLE_CLASS = EventTable

    def test_read_write_ligolw(self):
        table = super(EventTableTests, self).test_read_write_ligolw()
        # try reading with nproc
        table = self.TABLE_CLASS.read([TEST_XML_FILE, TEST_XML_FILE],
                                      format='ligolw.sngl_burst')
        table2 = self.TABLE_CLASS.read([TEST_XML_FILE, TEST_XML_FILE],
                                       nproc=2, format='ligolw.sngl_burst')
        self.assertTableEqual(table, table2)

    def test_read_omega(self):
        table = self.TABLE_CLASS.read(TEST_OMEGA_FILE, format='ascii.omega')
        self.assertIsInstance(table, self.TABLE_CLASS)
        self.assertIsInstance(table['frequency'], self.TABLE_CLASS.Column)
        self.assertEqual(len(table), 92)
        self.assertAlmostEqual(table[0]['frequency'], 962.609375)

    def test_event_rates(self):
        # test event_rate
        table = self.TABLE_CLASS.read(
            TEST_XML_FILE, format='ligolw.sngl_burst',
            columns=['time', 'snr'])
        rate = table.event_rate(1)
        self.assertIsInstance(rate, TimeSeries)
        self.assertEqual(rate.sample_rate, 1 * units.Hz)
        # test binned_event_rates
        rates = table.binned_event_rates(1, 'snr', [2, 4, 6])
        self.assertIsInstance(rates, TimeSeriesDict)
        table.binned_event_rates(1, 'snr', [2, 4, 6], operator='in')
        table.binned_event_rates(1, 'snr', [(0, 2), (2, 4), (4, 6)])

    def test_plot(self):
        table = self.TABLE_CLASS.read(TEST_OMEGA_FILE, format='ascii.omega')
        plot = table.plot('time', 'frequency', color='normalizedEnergy')

    def test_hist(self):
        table = self.TABLE_CLASS.read(TEST_OMEGA_FILE, format='ascii.omega')
        table.hist('normalizedEnergy')

    def test_get_column(self):
        table = self.TABLE_CLASS.read(TEST_OMEGA_FILE, format='ascii.omega')
        nptest.assert_array_equal(table.get_column('normalizedEnergy'),
                                  table['normalizedEnergy'])

    def test_read_hdf5_mp(self):
        try:
            import h5py
        except ImportError as e:
            self.skipTest(str(e))
        t = self.TABLE_CLASS(random.random((10, 10)))
        fp = tempfile.mktemp(suffix='.hdf')
        try:
            t.write(fp, format='hdf5', path='/test')
            h5file = h5py.File(fp, 'r')
            h5dset = h5file['/test']
            t2 = self.TABLE_CLASS.read(h5dset, format='hdf5')
        finally:
            if os.path.exists(fp):
                os.remove(fp)

    def test_read_pycbc_live(self):
        try:
            import h5py
        except ImportError as e:
            self.skipTest(str(e))
        table = self.TABLE_CLASS(random.random((10, 10)),
                                 names=['a', 'b', 'c', 'chisq', 'd', 'e', 'f',
                                        'mass1', 'mass2', 'snr'])
        table.meta['ifo'] = 'X1'
        fp = os.path.join(tempfile.mkdtemp(), 'X1-Live-0-0.hdf')
        try:
            # write table in pycbc_live format
            h5file = h5py.File(fp, 'w')
            group = h5file.create_group('X1')
            for col in table.columns:
                dataset = group.create_dataset(data=table[col], name=col)
            h5file.close()
            # assert reading works
            table2 = self.TABLE_CLASS.read(fp)
            self.assertTableEqual(table, table2)
            # assert keyword arguments result in same table
            table2 = self.TABLE_CLASS.read(fp, format='hdf5.pycbc_live')
            self.assertTableEqual(table, table2)
            table2 = self.TABLE_CLASS.read(fp, format='hdf5.pycbc_live',
                                           ifo='X1')
            self.assertTableEqual(table, table2)
            # add another IFO, then assert that reading the table without
            # specifying the IFO fails
            h5file = h5py.File(fp)
            h5file.create_group('Z1')
            with self.assertRaises(ValueError) as exc:
                self.TABLE_CLASS.read(fp)
            self.assertTrue(str(exc.exception).startswith(
                'PyCBC live HDF5 file contains dataset groups'))
            table2 = self.TABLE_CLASS.read(fp, format='hdf5.pycbc_live',
                                           ifo='X1')
            # assert processed colums works
            table2 = self.TABLE_CLASS.read(fp, ifo='X1',
                                           columns=['mchirp', 'new_snr'])
            mchirp = (table['mass1'] * table['mass2']) ** (3/5.) / (
                table['mass1'] + table['mass2']) ** (1/5.)
            nptest.assert_array_equal(table2['mchirp'], mchirp)
        finally:
            if os.path.exists(fp):
                os.remove(fp)
