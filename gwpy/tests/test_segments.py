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

import os.path
import tempfile
import StringIO
from six import PY3
from urllib2 import (urlopen, URLError)

from compat import unittest

from glue.segments import PosInfinity

from gwpy import version
from gwpy.segments import (Segment, SegmentList,
                           DataQualityFlag, DataQualityDict)

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

# get padding stuff
PADDING = (-0.5, 1)
KNOWNPAD = SegmentList([
    Segment(-.5, 4),
    Segment(5.5, 8),
])
ACTIVEPAD = SegmentList([
    Segment(.5, 3),
    Segment(2.5, 5),
    Segment(4.5, 8),
])
ACTIVEPADC = SegmentList([
    Segment(.5, 4),
    Segment(5.5, 8),
])

SEGXML = os.path.join(os.path.split(__file__)[0], 'data',
                      'X1-GWPY_TEST_SEGMENTS-0-10.xml.gz')
SEGWIZ = os.path.join(os.path.split(__file__)[0], 'data',
                      'X1-GWPY_TEST_SEGMENTS-0-10.txt')
FLAG1 = 'X1:GWPY-TEST_SEGMENTS:1'
FLAG2 = 'X1:GWPY-TEST_SEGMENTS:2'

QUERY_START = 1108598416
QUERY_END = 1108684816
QUERY_KNOWN = SegmentList([(1108598416, 1108632895), (1108632901, 1108684816)])
QUERY_ACTIVE = SegmentList([(1108623497, 1108624217)])
QUERY_FLAG = 'L1:DMT-DC_READOUT:1'
QUERY_URL = 'https://segments.ligo.org'

VETO_DEFINER_FILE = ('https://www.lsc-group.phys.uwm.edu/ligovirgo/cbc/public/'
                     'segments/ER7/H1L1V1-ER7_CBC_OFFLINE.xml')
ER7_START = 'June 3'
ER7_END = 'June 14'


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
        if PY3:
            types = [str]
        else:
            types = [str, unicode]
        for type_ in types:
            tmpfile = type_(self.tmpfile % 'xml.gz')
            try:
                DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN).write(tmpfile)
            finally:
                os.remove(tmpfile)

    def test_write_hdf5(self, delete=True):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        hdfout = self.tmpfile % 'hdf'
        try:
            flag.write(hdfout)
        except ImportError as e:
            self.skipTest(str(e))
        else:
            if delete:
                os.remove(hdfout)
        return hdfout

    def test_read_hdf5(self):
        try:
            hdfout = self.test_write_hdf5(delete=False)
        except ImportError as e:
            self.skipTest(str(e))
        else:
            flag = DataQualityFlag.read(hdfout)
            os.remove(hdfout)
            self.assertTrue(flag.active == ACTIVE,
                            'DataQualityFlag.read(hdf5) mismatch:\n\n%s\n\n%s'
                            % (ACTIVE, flag.active))
            self.assertTrue(flag.known == KNOWN,
                            'DataQualityFlag.read(hdf5) mismatch:\n\n%s\n\n%s'
                            % (KNOWN, flag.known))

    def _query(self, cm, *args, **kwargs):
        try:
            return cm(*args, **kwargs)
        except (ImportError, URLError) as e:
            self.skipTest(str(e))
        except AttributeError as e:
            if 'PKCS5_SALT_LEN' in str(e):
                self.skipTest(str(e))
            else:
                raise

    def test_query(self):
        flag = self._query(DataQualityFlag.query,
                           QUERY_FLAG, QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertEqual(flag.known, QUERY_KNOWN)
        self.assertEqual(flag.active, QUERY_ACTIVE)

    def test_query_dqsegdb(self):
        flag = self._query(DataQualityFlag.query_dqsegdb,
                           QUERY_FLAG, QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertEqual(flag.known, QUERY_KNOWN)
        self.assertEqual(flag.active, QUERY_ACTIVE)

    def test_query_dqsegdb_versionless(self):
        flag = self._query(DataQualityFlag.query_dqsegdb,
                           QUERY_FLAG.rsplit(':', 1)[0], QUERY_START,
                           QUERY_END, url=QUERY_URL)
        self.assertEqual(flag.known, QUERY_KNOWN)
        self.assertEqual(flag.active, QUERY_ACTIVE)

    def test_query_dqsegdb_multi(self):
        querymid = int(QUERY_START + (QUERY_END - QUERY_START) /2.)
        segs = SegmentList([Segment(QUERY_START, querymid),
                            Segment(querymid, QUERY_END)])
        flag = self._query(DataQualityFlag.query_dqsegdb,
                           QUERY_FLAG, segs, url=QUERY_URL)
        self.assertEqual(flag.known, QUERY_KNOWN)
        self.assertEqual(flag.active, QUERY_ACTIVE)

    def test_pad(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        # test without arguments (and no padding)
        padded = flag.pad()
        self.assertListEqual(padded.known, flag.known)
        self.assertListEqual(padded.active, flag.active)
        # test without arguments (and no padding)
        flag.padding = PADDING
        padded = flag.pad()
        self.assertListEqual(padded.known, KNOWNPAD)
        self.assertListEqual(padded.active, ACTIVEPAD)
        # test with arguments
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        padded = flag.pad(*PADDING)
        self.assertListEqual(padded.known, KNOWNPAD)
        self.assertListEqual(padded.active, ACTIVEPAD)
        # test coalesce
        padded.coalesce()
        self.assertListEqual(padded.active, ACTIVEPADC)


class DataQualityDictTestCase(unittest.TestCase):
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_dqdict')
    VETO_DEFINER = tmpfile % 'vdf.xml'

    def setUp(self):
        # download veto definer
        vdffile = urlopen(VETO_DEFINER_FILE)
        with open(self.VETO_DEFINER, 'w') as f:
            f.write(vdffile.read())

    def tearDown(self):
        if os.path.isfile(self.VETO_DEFINER):
            os.remove(self.VETO_DEFINER)

    def test_from_veto_definer_file(self):
        vdf = DataQualityDict.from_veto_definer_file(self.VETO_DEFINER)
        self.assertNotEqual(len(vdf.keys()), 0)
        # test missing h(t) flag
        self.assertIn('H1:DCH-MISSING_H1_HOFT_C00:1', vdf)
        self.assertEquals(vdf['H1:DCH-MISSING_H1_HOFT_C00:1'].known[0][0],
                          1073779216)
        self.assertEquals(vdf['H1:DCH-MISSING_H1_HOFT_C00:1'].known[0][-1],
                          PosInfinity)
        self.assertEquals(vdf['H1:DCH-MISSING_H1_HOFT_C00:1'].category, 1)
        # test injections padding
        self.assertEquals(vdf['H1:ODC-INJECTION_CBC:1'].padding, Segment(-8, 8))

    def test_read_ligolw(self):
        flags = DataQualityDict.read(SEGXML)
        self.assertEquals(len(flags.keys()), 2)
        self.assertIn(FLAG1, flags)
        self.assertIn(FLAG2, flags)
        flags = DataQualityDict.read(SEGXML, [FLAG2])
        self.assertEquals(len(flags.keys()), 1)
        self.assertEquals(flags[FLAG2].known, KNOWN2)
        self.assertEquals(flags[FLAG2].active, ACTIVE2)

    def test_write_ligolw(self):
        tmpfile = self.tmpfile % 'xml.gz'
        try:
            flags = DataQualityDict.read(SEGXML)
        except Exception as e:
            self.skipTest(str(e))
        try:
            flags.write(tmpfile)
        finally:
            os.remove(tmpfile)


if __name__ == '__main__':
    unittest.main()
