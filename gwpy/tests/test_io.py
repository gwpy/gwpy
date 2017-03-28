# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""Unit test for `io` module
"""

import os

from common import skip_missing_import
from compat import (unittest, mock)
import mockutils

from gwpy.io import (datafind, gwf)
from gwpy.io.cache import (Cache, CacheEntry, cache_segments,
                           flatten, find_contiguous)
from gwpy.segments import (Segment, SegmentList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_GWF_FILE = os.path.join(os.path.split(__file__)[0], 'data',
                          'HLV-GW100916-968654552-1.gwf')
TEST_CHANNELS = [
    'H1:LDAS-STRAIN', 'L1:LDAS-STRAIN', 'V1:h_16384Hz',
]


# -- gwpy.io.nds --------------------------------------------------------------

class NdsIoTestCase(unittest.TestCase):
    def test_nds2_host_order_none(self):
        """Test `host_resolution_order` with `None` IFO
        """
        try:
            from gwpy.io import nds
        except ImportError as e:
            self.skipTest(str(e))
        hro = nds.host_resolution_order(None, env=None)
        self.assertListEqual(hro, [('nds.ligo.caltech.edu', 31200)])

    def test_nds2_host_order_ifo(self):
        """Test `host_resolution_order` with `ifo` argument
        """
        try:
            from gwpy.io import nds
        except ImportError as e:
            self.skipTest(str(e))
        hro = nds.host_resolution_order('L1', env=None)
        self.assertListEqual(
            hro, [('nds.ligo-la.caltech.edu', 31200),
                  ('nds.ligo.caltech.edu', 31200)])

    def test_nds2_host_order_ndsserver(self):
        """Test `host_resolution_order` with default env set
        """
        try:
            from gwpy.io import nds
        except ImportError as e:
            self.skipTest(str(e))
        os.environ['NDSSERVER'] = 'test1.ligo.org:80,test2.ligo.org:43'
        hro = nds.host_resolution_order(None)
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                  ('nds.ligo.caltech.edu', 31200)])
        hro = nds.host_resolution_order('L1')
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                  ('nds.ligo-la.caltech.edu', 31200),
                  ('nds.ligo.caltech.edu', 31200)])

    def test_nds2_host_order_env(self):
        """Test `host_resolution_order` with non-default env set
        """
        try:
            from gwpy.io import nds
        except ImportError as e:
            self.skipTest(str(e))
        os.environ['TESTENV'] = 'test1.ligo.org:80,test2.ligo.org:43'
        hro = nds.host_resolution_order(None, env='TESTENV')
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                  ('nds.ligo.caltech.edu', 31200)])

    def test_nds2_host_order_epoch(self):
        """Test `host_resolution_order` with old GPS epoch
        """
        try:
            from gwpy.io import nds
        except ImportError as e:
            self.skipTest(str(e))
        # test kwarg doesn't change anything
        hro = nds.host_resolution_order('L1', epoch='now', env=None)
        self.assertListEqual(
            hro, [('nds.ligo-la.caltech.edu', 31200),
                  ('nds.ligo.caltech.edu', 31200)])
        # test old epoch puts CIT ahead of LLO
        hro = nds.host_resolution_order('L1', epoch='Jan 1 2015', env=None)
        self.assertListEqual(
            hro, [('nds.ligo.caltech.edu', 31200),
                  ('nds.ligo-la.caltech.edu', 31200)])
        # test epoch doesn't operate with env
        os.environ['TESTENV'] = 'test1.ligo.org:80,test2.ligo.org:43'
        hro = nds.host_resolution_order('L1', epoch='now', env='TESTENV')
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                  ('nds.ligo-la.caltech.edu', 31200),
                  ('nds.ligo.caltech.edu', 31200)])


# -- gwpy.io.cache ------------------------------------------------------------

class CacheIoTestCase(unittest.TestCase):
    @staticmethod
    def make_cache():
        segs = SegmentList()
        cache = Cache()
        for seg in [(0, 1), (1, 2), (4, 5)]:
            d = seg[1] - seg[0]
            f = 'A-B-%d-%d.tmp' % (seg[0], d)
            cache.append(CacheEntry.from_T050017(f, coltype=int))
            segs.append(Segment(*seg))
        return cache, segs

    def test_cache_segments(self):
        # check empty input
        sl = cache_segments()
        self.assertIsInstance(sl, SegmentList)
        self.assertEquals(len(sl), 0)
        cache, segs = self.make_cache()
        segs.coalesce()
        sl = cache_segments(cache)
        self.assertEquals(sl, segs)
        sl = cache_segments(cache[:2], cache[2:])
        self.assertEquals(sl, segs)

    def test_flatten(self):
        # check flattened version of single cache is unchanged
        a, _ = self.make_cache()
        self.assertListEqual(flatten(a), a)
        self.assertListEqual(flatten(a, a), a)
        # check two caches get concatenated properly
        b, _ = self.make_cache()
        for e in b:
            e.segment = e.segment.shift(10)
        c = a + b
        self.assertListEqual(flatten(a, b), c)

    def test_find_contiguous(self):
        a, segs = self.make_cache()
        segs.coalesce()
        for i, cache in enumerate(find_contiguous(a)):
            self.assertEqual(cache.to_segmentlistdict()['A'].extent(), segs[i])


# -- gwpy.io.gwf --------------------------------------------------------------

def mock_call(*args, **kwargs):
    raise OSError("")

class GwfIoTestCase(unittest.TestCase):

    @skip_missing_import('lalframe')
    def test_iter_channel_names(self):
        # maybe need something better?
        from types import GeneratorType
        names = gwf.iter_channel_names(TEST_GWF_FILE)
        self.assertIsInstance(names, GeneratorType)
        self.assertSequenceEqual(list(names), TEST_CHANNELS)
        with mock.patch('gwpy.utils.shell.call', mock_call):
            names = gwf.iter_channel_names(TEST_GWF_FILE)
            self.assertIsInstance(names, GeneratorType)
            self.assertSequenceEqual(list(names), TEST_CHANNELS)

    @skip_missing_import('lalframe')
    def test_get_channel_names(self):
        self.assertListEqual(gwf.get_channel_names(TEST_GWF_FILE),
                             TEST_CHANNELS)

    @skip_missing_import('lalframe')
    def test_num_channels(self):
        self.assertEqual(gwf.num_channels(TEST_GWF_FILE), 3)

    @skip_missing_import('lalframe')
    def test_get_channel_type(self):
        self.assertEqual(gwf.get_channel_type(
            'L1:LDAS-STRAIN', TEST_GWF_FILE), 'proc')
        self.assertRaises(ValueError, gwf.get_channel_type,
                          'X1:NOT-IN_FRAME', TEST_GWF_FILE)

    @skip_missing_import('lalframe')
    def test_channel_in_frame(self):
        self.assertTrue(
            gwf.channel_in_frame('L1:LDAS-STRAIN', TEST_GWF_FILE))
        self.assertFalse(
            gwf.channel_in_frame('X1:NOT-IN_FRAME', TEST_GWF_FILE))


# -- gwpy.io.datafind ---------------------------------------------------------

class DataFindIoTestCase(unittest.TestCase):
    MOCK_CONNECTION = mockutils.mock_datafind_connection(TEST_GWF_FILE)

    def test_on_tape(self):
        self.assertFalse(datafind.on_tape(TEST_GWF_FILE))
        self.assertFalse(datafind.on_tape(
            CacheEntry.from_T050017(TEST_GWF_FILE, coltype=int)))

    def test_connect(self):
        with mock.patch('glue.datafind.GWDataFindHTTPConnection',
                        self.MOCK_CONNECTION), \
             mock.patch('glue.datafind.GWDataFindHTTPSConnection',
                        self.MOCK_CONNECTION), \
             mock.patch('glue.datafind.find_credential',
                        mockutils.mock_find_credential):
            datafind.connect()
            datafind.connect('host', 443)

    def test_find_frametype(self):
        with mock.patch('glue.datafind.GWDataFindHTTPConnection') as \
                 mock_connection, \
                 mock.patch('gwpy.io.datafind.num_channels', lambda x: 1), \
                 mock.patch('gwpy.io.gwf.iter_channel_names',
                            lambda x: ['L1:LDAS-STRAIN']):
            mock_connection.return_value = self.MOCK_CONNECTION
            ft = datafind.find_frametype('L1:LDAS-STRAIN', allow_tape=True)
            self.assertEqual(ft, 'GW100916')
            ft = datafind.find_frametype('L1:LDAS-STRAIN', return_all=True)
            self.assertListEqual(ft, ['GW100916'])
            self.assertRaises(ValueError, datafind.find_frametype, 'X1:TEST')
            self.assertRaises(ValueError, datafind.find_frametype,
                              'bad channel name')
            # test trend sorting ends up with an error
            self.assertRaises(ValueError, datafind.find_frametype,
                              'X1:TEST.rms,s-trend')
            self.assertRaises(ValueError, datafind.find_frametype,
                              'X1:TEST.rms,m-trend')

    def test_find_best_frametype(self):
        with mock.patch('glue.datafind.GWDataFindHTTPConnection') as \
                 mock_connection, \
                 mock.patch('gwpy.io.datafind.num_channels', lambda x: 1), \
                 mock.patch('gwpy.io.gwf.iter_channel_names',
                            lambda x: ['L1:LDAS-STRAIN']):
            mock_connection.return_value = self.MOCK_CONNECTION
            ft = datafind.find_best_frametype('L1:LDAS-STRAIN', 968654552,
                                              968654553)
            self.assertEqual(ft, 'GW100916')


if __name__ == '__main__':
    unittest.main()
