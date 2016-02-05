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
import unittest

import numpy
from numpy import testing as nptest

from gwpy import version
from gwpy.table import lsctables

import common

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


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
        os.path.split(__file__)[0], 'data',
        'H1-LDAS_STRAIN-968654552-10.xml.gz')

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
