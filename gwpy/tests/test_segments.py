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
import warnings
from ssl import SSLError

from six import PY3
from six.moves.urllib.request import urlopen
from six.moves.urllib.error import URLError
from six.moves import StringIO

import pytest

try:
    from unittest import mock
except ImportError:
    import mock

from glue.segments import PosInfinity
from glue.LDBDWClient import LDBDClientException

from gwpy.segments import (Segment, SegmentList,
                           DataQualityFlag, DataQualityDict)
from gwpy.io.registry import identify_format
from gwpy.plotter import (SegmentPlot, SegmentAxes)

from compat import unittest
import common
import mockutils

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

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
ACTIVE_CONTRACTED = SegmentList([
    Segment(1.1, 1.9),
    Segment(3.1, 3.9),
    Segment(5.1, 6.9),
])
ACTIVE_PROTRACTED = SegmentList([
    Segment(.9, 2.1),
    Segment(2.9, 4.1),
    Segment(4.9, 7.1),
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

QUERY_START = 968630415
QUERY_END = 968716815
QUERY_FLAGS = ['H1:DMT-SCIENCE:4', 'L1:DMT-SCIENCE:4']
QUERY_RESULT = DataQualityDict()
QUERY_RESULT['H1:DMT-SCIENCE:4'] = DataQualityFlag(
    'H1:DMT-SCIENCE:4',
    known=[(968630415, 968716815)],
    active=[(968632249, 968641010), (968642871, 968644430),
            (968646220, 968681205), (968686357, 968686575),
            (968688760, 968690950), (968692881, 968714403)])
QUERY_RESULT['L1:DMT-SCIENCE:4'] = DataQualityFlag(
    'L1:DMT-SCIENCE:4',
     known=[(968630415, 968716815)],
     active=[(968630415, 968634911), (968638548, 968644632),
             (968646025, 968675387), (968676835, 968679443),
             (968680215, 968686803), (968688905, 968691838),
             (968693321, 968694106), (968694718, 968699812),
             (968701111, 968713996), (968714886, 968716815)])
QUERY_URL = 'https://segments-s6.ligo.org'
QUERY_URL_SEGDB = 'https://segdb.ligo.caltech.edu'

VETO_DEFINER_FILE = ('https://www.lsc-group.phys.uwm.edu/ligovirgo/cbc/public/'
                     'segments/ER7/H1L1V1-ER7_CBC_OFFLINE.xml')
VETO_DEFINER_TEST_SEGMENTS = SegmentList([Segment(1117411216, 1117497616)])


class TestCaseWithQueryMixin(object):

    def _mock_query(self, cm, result, *args, **kwargs):
        """Query for segments using a mock of the dqsegdb API
        """
        try:
            return cm(*args, **kwargs)
        except (UnboundLocalError, AttributeError, URLError) as e:
            warnings.warn("Test query failed with %s: %s, "
                          "rerunning with mock..."
                          % (type(e).__name__, str(e)))
            with mock.patch('dqsegdb.apicalls.dqsegdbQueryTimes',
                            mockutils.mock_query_times(result)):
                return cm(*args, **kwargs)


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

    def test_io_identify(self):
        common.test_io_identify(SegmentList, ['txt', 'hdf', 'hdf5'])


class DataQualityFlagTests(unittest.TestCase, TestCaseWithQueryMixin):
    """Unit tests for the `DataQualityFlag` class
    """
    tmpfile = '%s.%%s' % tempfile.mktemp(prefix='gwpy_test_dqflag')

    def test_properties(self):
        empty = DataQualityFlag()
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN,
                               padding=(-4, 8))
        # name
        self.assertEqual(empty.name, None)
        self.assertEqual(flag.name, FLAG1)
        self.assertEqual(flag.ifo, FLAG1.split(':')[0])
        self.assertEqual(flag.version, int(FLAG1.split(':')[-1]))
        # known
        self.assertIsInstance(empty.known, SegmentList)
        self.assertListEqual(empty.known, SegmentList())
        self.assertListEqual(flag.known, KNOWN)
        # active
        self.assertIsInstance(empty.active, SegmentList)
        self.assertListEqual(empty.active, SegmentList())
        self.assertListEqual(flag.active, ACTIVE)
        # padding
        self.assertTupleEqual(empty.padding, (0, 0))
        self.assertTupleEqual(flag.padding, (-4, 8))
        # texname
        self.assertEqual(flag.texname, FLAG1.replace('_', r'\_'))
        self.assertEqual(empty.texname, None)
        # livetime
        self.assertEqual(flag.livetime, 4)

    def test_deprecated(self):
        with pytest.warns(DeprecationWarning):
            flag = DataQualityFlag(FLAG1, active=ACTIVE, valid=KNOWN)
        with pytest.warns(DeprecationWarning):
            flag.valid
        with pytest.warns(DeprecationWarning):
            flag.valid = flag.known
        with pytest.warns(DeprecationWarning):
            del flag.valid

    def test_plot(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        plot = flag.plot()
        self.assertIsInstance(plot, SegmentPlot)
        self.assertIsInstance(plot.gca(), SegmentAxes)
        self.assertEqual(plot.gca().get_epoch(), flag.known[0][0])

    def test_parse_name(self):
        flag = DataQualityFlag(None)
        self.assertIsNone(flag.name)
        self.assertIsNone(flag.ifo)
        self.assertIsNone(flag.tag)
        self.assertIsNone(flag.version)
        flag = DataQualityFlag('test')
        self.assertEqual(flag.name, 'test')
        self.assertIsNone(flag.ifo)
        self.assertIsNone(flag.tag)
        self.assertIsNone(flag.version)
        flag = DataQualityFlag('L1:test')
        self.assertEqual(flag.name, 'L1:test')
        self.assertEqual(flag.ifo, 'L1')
        self.assertEqual(flag.tag, 'test')
        self.assertIsNone(flag.version)
        flag = DataQualityFlag('L1:test:1')
        self.assertEqual(flag.name, 'L1:test:1')
        self.assertEqual(flag.ifo, 'L1')
        self.assertEqual(flag.tag, 'test')
        self.assertEqual(flag.version, 1)
        flag = DataQualityFlag('test:1')
        self.assertEqual(flag.name, 'test:1')
        self.assertIsNone(flag.ifo)
        self.assertEqual(flag.tag, 'test')
        self.assertEqual(flag.version, 1)

    def test_math(self):
        flag1 = DataQualityFlag(FLAG1, active=ACTIVE[:2], known=KNOWN)
        flag2 = DataQualityFlag(FLAG1, active=ACTIVE[2:], known=KNOWN)
        # and
        x = flag1 & flag2
        self.assertListEqual(x.active, [])
        self.assertListEqual(x.known, KNOWN)
        # sub
        x = flag1 - flag2
        self.assertListEqual(x.active, flag1.active)
        self.assertListEqual(x.known, flag1.known)
        # or
        x = flag1 | flag2
        self.assertListEqual(x.active, ACTIVE)
        self.assertListEqual(x.known, KNOWN)

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

    def test_contract(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        flag.contract(.1)
        self.assertListEqual(flag.active, ACTIVE_CONTRACTED)

    def test_protract(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        flag.protract(.1)
        self.assertListEqual(flag.active, ACTIVE_PROTRACTED)

    def test_round(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE_CONTRACTED, known=KNOWN)
        flag2 = flag.round()
        self.assertListEqual(flag2.active, ACTIVE & KNOWN)

    def test_repr_str(self):
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        repr(flag)
        str(flag)

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
        self.assertRaises(ValueError, flag.write, StringIO,
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

    def test_query(self):
        flag = QUERY_FLAGS[0]
        result = self._mock_query(
            DataQualityFlag.query, QUERY_RESULT,
            flag, QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertEqual(result.known, QUERY_RESULT[flag].known)
        self.assertEqual(result.active, QUERY_RESULT[flag].active)

    def test_query_dqsegdb(self):
        flag = QUERY_FLAGS[0]
        result = self._mock_query(
            DataQualityFlag.query_dqsegdb, QUERY_RESULT,
            flag, QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertEqual(result.known, QUERY_RESULT[flag].known)
        self.assertEqual(result.active, QUERY_RESULT[flag].active)

    def test_query_segdb(self):
        flag = QUERY_FLAGS[0]
        result = self._mock_query(
            DataQualityFlag.query_segdb, QUERY_RESULT,
            flag, QUERY_START, QUERY_END, url=QUERY_URL_SEGDB)
        self.assertEqual(result.known, QUERY_RESULT[flag].known)
        self.assertEqual(result.active, QUERY_RESULT[flag].active)

    def test_query_dqsegdb_versionless(self):
        flag = QUERY_FLAGS[0]
        result = self._mock_query(
            DataQualityFlag.query, QUERY_RESULT,
            flag.rsplit(':', 1)[0], QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertEqual(result.known, QUERY_RESULT[flag].known)
        self.assertEqual(result.active, QUERY_RESULT[flag].active)


    def test_query_dqsegdb_multi(self):
        querymid = int(QUERY_START + (QUERY_END - QUERY_START) /2.)
        segs = SegmentList([Segment(QUERY_START, querymid),
                            Segment(querymid, QUERY_END)])
        flag = QUERY_FLAGS[0]
        result = self._mock_query(
            DataQualityFlag.query, QUERY_RESULT,
            flag, segs, url=QUERY_URL)
        self.assertEqual(result.known, QUERY_RESULT[flag].known)
        self.assertEqual(result.active, QUERY_RESULT[flag].active)

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
        # test in-place
        flag = DataQualityFlag(FLAG1, active=ACTIVE, known=KNOWN)
        padded = flag.pad(*PADDING)
        self.assertIsNot(flag, padded)
        padded = flag.pad(*PADDING, inplace=True)
        self.assertIs(flag, padded)
        # test other kwargs fail
        self.assertRaises(TypeError, flag.pad, *PADDING, kwarg='test')

    def test_io_identify(self):
        common.test_io_identify(DataQualityFlag, ['xml', 'xml.gz', 'txt'])


class DataQualityDictTests(unittest.TestCase, TestCaseWithQueryMixin):
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

    def create(self):
        flgd = DataQualityDict()
        flgd['flag1'] = DataQualityFlag(name='flag1', active=ACTIVE,
                                        known=KNOWN)
        flgd['flag2'] = DataQualityFlag(name='flag2', active=ACTIVE2,
                                        known=KNOWN2)
        return flgd

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
        # test download URL
        vdf2 = DataQualityDict.from_veto_definer_file(VETO_DEFINER_FILE)
        self.assertEqual(len(vdf.keys()), len(vdf2.keys()))

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

    def test_io_identify(self):
        common.test_io_identify(DataQualityDict, ['xml', 'xml.gz'])

    def test_populate(self):
        # read veto definer
        start, end = VETO_DEFINER_TEST_SEGMENTS[0]
        vdf = DataQualityDict.from_veto_definer_file(
            VETO_DEFINER_FILE, ifo='H1', start=start, end=end)
        # test query that should fail with 404
        try:
            vdf.populate(url='https://segments.ligo.org')
        except (ImportError, UnboundLocalError) as e:
            self.skipTest(str(e))
        except URLError as e:
            if e.code == 401:  # 401 is uninformative
                self.skipTest(str(e))
            elif e.code == 404:
                pass
            else:
                raise
        else:
            raise AssertionError("URLError not raised")
        # check reduction to warning
        vdf = DataQualityDict.from_veto_definer_file(
            VETO_DEFINER_FILE, ifo='H1', start=start, end=end)
        with pytest.warns(UserWarning):
            vdf.populate(url='https://segments.ligo.org', on_error='warn')
        # check results
        self.assertEqual(
            len(vdf['H1:HVT-ER7_A_RND17:1'].active), 36)
        # check use of specific segments
        vdf = DataQualityDict.from_veto_definer_file(
            VETO_DEFINER_FILE, ifo='H1')
        vdf.populate(segments=VETO_DEFINER_TEST_SEGMENTS, on_error='ignore')

    def test_query(self):
        result = self._mock_query(
            DataQualityDict.query, QUERY_RESULT,
            QUERY_FLAGS, QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertListEqual(result.keys(), QUERY_FLAGS)
        for flag in result:
            self.assertEqual(result[flag].known, QUERY_RESULT[flag].known)
            self.assertEqual(result[flag].active, QUERY_RESULT[flag].active)

    def test_query_dqsegdb(self):
        result = self._mock_query(
            DataQualityDict.query_dqsegdb, QUERY_RESULT,
            QUERY_FLAGS, QUERY_START, QUERY_END, url=QUERY_URL)
        self.assertListEqual(result.keys(), QUERY_FLAGS)
        for flag in result:
            self.assertEqual(result[flag].known, QUERY_RESULT[flag].known)
            self.assertEqual(result[flag].active, QUERY_RESULT[flag].active)

    def test_query_segdb(self):
        result = self._mock_query(
            DataQualityDict.query_segdb, QUERY_RESULT,
            QUERY_FLAGS, QUERY_START, QUERY_END, url=QUERY_URL_SEGDB)
        self.assertListEqual(result.keys(), QUERY_FLAGS)
        for flag in result:
            self.assertEqual(result[flag].known, QUERY_RESULT[flag].known)
            self.assertEqual(result[flag].active, QUERY_RESULT[flag].active)

    def test_union(self):
        flgd = self.create()
        union = flgd.union()
        self.assertIsInstance(union, DataQualityFlag)
        self.assertListEqual(union.known, KNOWN + KNOWN2)
        self.assertListEqual(union.active, ACTIVE + ACTIVE2)

    def test_intersection(self):
        flgd = self.create()
        inter = flgd.intersection()
        self.assertIsInstance(inter, DataQualityFlag)
        self.assertListEqual(inter.known, KNOWN & KNOWN2)
        self.assertListEqual(inter.active, ACTIVE & ACTIVE2)

    def test_plot(self):
        flgd = self.create()
        plot = flgd.plot()
        self.assertIsInstance(plot, SegmentPlot)
        self.assertIsInstance(plot.gca(), SegmentAxes)
        self.assertEqual(len(plot.gca().collections), len(flgd) * 2)

if __name__ == '__main__':
    unittest.main()
