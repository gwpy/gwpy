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

"""Unit test for table module
"""

import os.path
import tempfile

import pytest

import numpy
from numpy import testing as nptest

from astropy import units

from gwpy.time import LIGOTimeGPS
from gwpy.table import lsctables
from gwpy.table.io import (omega, trigfind)
from gwpy.timeseries import (TimeSeries, TimeSeriesDict)

import common
from compat import unittest

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_DATA_DIR = os.path.join(os.path.split(__file__)[0], 'data')


class TableTestMixin(object):
    TABLE_CLASS = None

    def test_has_gwpy_methods(self):
        for method in ['read', 'plot']:
            self.assertTrue(hasattr(self.TABLE_CLASS, method))


class SnglBurstTableTestCase(TableTestMixin, unittest.TestCase):
    """`TestCase` for `SnglBurstTable`
    """
    TABLE_CLASS = lsctables.SnglBurstTable
    TEST_XML_FILE = os.path.join(
        TEST_DATA_DIR, 'H1-LDAS_STRAIN-968654552-10.xml.gz')
    TEST_OMEGA_FILE = os.path.join(TEST_DATA_DIR, 'omega.txt')
    TEST_OMEGADQ_FILE = os.path.join(TEST_DATA_DIR, 'omegadq.txt')

    def test_read_ligolw(self):
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE)
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE, format='ligolw')
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE, format='sngl_burst')
        self.assertEquals(len(table), 2052)

    def test_write_ligolw(self):
        # read table
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE)
        tmpxml = tempfile.mktemp(suffix='.xml.gz')
        # write table
        try:
            table.write(open(tmpxml, 'w'))
            table2 = self.TABLE_CLASS.read(tmpxml)
        finally:
            if os.path.isfile(tmpxml):
                os.remove(tmpxml)
        self.assertEquals(len(table), len(table2))
        self.assertEquals(table[0].get_peak(), table2[0].get_peak())
        self.assertEquals(table[0].snr, table2[0].snr)

    def test_io_identify(self):
        common.test_io_identify(self.TABLE_CLASS,
                                ['xml', 'xml.gz', 'omicron.root'])

    def test_read_ascii(self):
        # read table
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE)
        # write to ASCII
        tmpascii = tempfile.mktemp(suffix='.txt')
        numpy.savetxt(tmpascii, zip(table.get_peak(), table.get_column('snr'),
                                    table.get_column('central_freq')),
                      fmt=['%s', '%.18e', '%.18e'])
        # read from ASCII
        try:
            table2 = self.TABLE_CLASS.read(
                tmpascii, columns=['time', 'snr', 'central_freq'])
        finally:
            if os.path.isfile(tmpascii):
                os.remove(tmpascii)
        self.assertEquals(len(table), len(table2))
        nptest.assert_array_equal(table.get_peak(), table2.get_peak())
        nptest.assert_array_equal(table.get_column('snr'),
                                  table2.get_column('snr'))
        nptest.assert_array_equal(table.get_column('central_freq'),
                                  table2.get_column('central_freq'))

    def test_trigfind(self):
        # test error
        self.assertRaises(ValueError, trigfind.find_trigger_urls,
                          'X1:CHANNEL', 'doesnt-exist', 0, 1)

    def test_read_omega(self):
        self.assertRaises(TypeError, self.TABLE_CLASS.read,
                          self.TEST_OMEGA_FILE)
        table = self.TABLE_CLASS.read(self.TEST_OMEGA_FILE, format='omega')
        self.assertEquals(len(table), 92)
        self.assertEquals(table[0].snr, 147.66187483916312)
        self.assertEquals(table[50].get_start(),
                          LIGOTimeGPS(966211219, 530621317))
        self.assertEquals(table.columnnames, omega.OMEGA_LIGOLW_COLUMNS)

    def test_read_omegadq(self):
        self.assertRaises(TypeError, self.TABLE_CLASS.read,
                          self.TEST_OMEGADQ_FILE)
        table = self.TABLE_CLASS.read(self.TEST_OMEGADQ_FILE, format='omegadq')
        self.assertEquals(len(table), 9)
        self.assertEquals(table[0].snr, 5.304714883949938)
        self.assertEquals(table[6].get_start(),
                          LIGOTimeGPS(966211228,123722672))
        self.assertEquals(table.columnnames, omega.OMEGADQ_LIGOLW_COLUMNS)

    def test_rates(self):
        # test event_rate
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE)
        rate = table.event_rate(1)
        self.assertIsInstance(rate, TimeSeries)
        self.assertEquals(rate.sample_rate, 1 * units.Hz)
        # test binned_event_rates
        rates = table.binned_event_rates(1, 'snr', [2, 4, 6])
        self.assertIsInstance(rates, TimeSeriesDict)
        table.binned_event_rates(1, 'snr', [2, 4, 6], operator='in')
        table.binned_event_rates(1, 'snr', [(0, 2), (2, 4), (4, 6)])

    def test_to_recarray(self):
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE)
        arr = table.to_recarray()
        self.assertListEqual(list(arr.dtype.names), table.columnnames)
        nptest.assert_array_equal(table.get_column('snr'), arr['snr'])
        nptest.assert_array_equal(
            table.get_peak().astype(float),
            arr['peak_time'] + arr['peak_time_ns'] * 1e-9)
        # test with errors/warning
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE, columns=['snr'])
        self.assertRaises(AttributeError, table.to_recarray)
        with pytest.warns(UserWarning):
            table.to_recarray(on_attributeerror='warn')

    def test_from_recarray(self):
        table = self.TABLE_CLASS.read(self.TEST_XML_FILE)
        arr = table.to_recarray()
        table2 = self.TABLE_CLASS.from_recarray(arr)
        for column in table.columnnames:
            if column == 'process_id':  # broken
                continue
            if table.validcolumns[column] == 'ilwd:char':  # ID columns
                nptest.assert_array_equal(table.getColumnByName(column),
                                          table2.getColumnByName(column))
            else:
                nptest.assert_array_equal(
                    table.getColumnByName(column).asarray(),
                    table2.getColumnByName(column).asarray())
