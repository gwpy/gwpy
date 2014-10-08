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

"""Unit test for segments module
"""

import unittest
import os.path
import tempfile
import StringIO

from .. import version
from ..segments import (Segment, SegmentList, DataQualityFlag, DataQualityDict)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

KNOWN = SegmentList([
    Segment(0, 3),
    Segment(6, 7),
])
ACTIVE = SegmentList([
    Segment(1, 2),
    Segment(3, 4),
    Segment(5, 7),
])
KNOWNACTIVE = SegmentList([
    Segment(1, 2),
    Segment(6, 7),
])

KNOWN2 = SegmentList([Segment(100, 150)])
ACTIVE2 = SegmentList([
    Segment(100, 101),
    Segment(110, 120),
])

SEGXML = os.path.join(os.path.split(__file__)[0], 'data',
                      'X1-GWPY_TEST_SEGMENTS-0-10.xml.gz')
SEGWIZ = os.path.join(os.path.split(__file__)[0], 'data',
                      'X1-GWPY_TEST_SEGMENTS-0-10.txt')
FLAG1 = 'X1:GWPY-TEST_SEGMENTS:1'
FLAG2 = 'X1:GWPY-TEST_SEGMENTS:2'


class SegmentListTests(unittest.TestCase):
    """Unit tests for the `SegmentList` class
    """
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_segmentlist')

    def test_read_segwizard(self):
        active = SegmentList.read(SEGWIZ, coalesce=False)
        self.assertTrue(active == ACTIVE,
                        'SegmentList.read(segwizard) mismatch:\n\n%s\n\n%s'
                        % (ACTIVE, active))

    def test_write_segwizard(self):
        tmpfile = self.tmpfile % 'txt'
        ACTIVE.write(tmpfile)
        self.assertTrue(open(tmpfile, 'r').read() == open(SEGWIZ, 'r').read(),
                        'SegmentList.write(segwizard) mismatch, %s '
                        'differs from %s' % (tmpfile, SEGWIZ))
        os.remove(tmpfile)


class DataQualityFlagTests(unittest.TestCase):
    """Unit tests for the `DataQualityFlag` class
    """
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_dqflag')

    def test_coalesce(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        self.assertFalse(flag.regular,
                         'flag.regular test failed (should be False)')
        flag.coalesce()
        self.assertTrue(flag.known == KNOWN, 'flag.known changed by coalesce')
        self.assertTrue(flag.active == KNOWNACTIVE,
                        'flag.active misset by coalesce')
        self.assertTrue(flag.regular,
                        'flag.regular test failed (should be True)')

    def test_read_segwizard(self):
        flag = DataQualityFlag.read(SEGWIZ, FLAG1, coalesce=False)
        self.assertTrue(flag.active == ACTIVE,
                        'DataQualityFlag.read(segwizard) mismatch:\n\n%s\n\n%s'
                        % (ACTIVE, flag.active))
        self.assertTrue(flag.known == flag.active)

    def test_write_segwizard(self):
        tmpfile = self.tmpfile % 'txt'
        DataQualityFlag(FLAG1, active=ACTIVE).write(tmpfile)
        self.assertTrue(open(tmpfile, 'r').read() == open(SEGWIZ, 'r').read(),
                        'DataQualityFlag.write(segwizard) mismatch, %s '
                        'differs from %s' % (tmpfile, SEGWIZ))
        os.remove(tmpfile)

    def test_fail_write_segwizard(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        self.assertRaises(ValueError, flag.write, StringIO.StringIO,
                          format='segwizard')

    def test_read_ligolw(self):
        flag = DataQualityFlag.read(SEGXML, FLAG1, coalesce=False)
        self.assertTrue(flag.active == ACTIVE,
                        'DataQualityFlag.read(ligol) mismatch:\n\n%s\n\n%s'
                        % (ACTIVE, flag.active))
        self.assertTrue(flag.known == KNOWN,
                        'DataQualityFlag.read(ligol) mismatch:\n\n%s\n\n%s'
                        % (KNOWN, flag.known))

    def test_write_ligolw(self):
        tmpfile = self.tmpfile % 'xml.gz'
        DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN).write(tmpfile)
        os.remove(tmpfile)

    def test_write_hdf5(self, delete=True):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        hdfout = self.tmpfile % 'hdf'
        try:
            flag.write(hdfout)
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        else:
            if delete:
                os.remove(hdfout)
        return hdfout

    def test_read_hdf5(self):
        try:
            hdfout = self.test_write_hdf5(delete=False)
        except ImportError as e:
            raise unittest.SkipTest(str(e))
        else:
            flag = DataQualityFlag.read(hdfout)
            os.remove(hdfout)
            self.assertTrue(flag.active == ACTIVE,
                            'DataQualityFlag.read(hdf5) mismatch:\n\n%s\n\n%s'
                            % (ACTIVE, flag.active))
            self.assertTrue(flag.known == KNOWN,
                            'DataQualityFlag.read(hdf5) mismatch:\n\n%s\n\n%s'
                            % (KNOWN, flag.known))

if __name__ == '__main__':
    unittest.main()
