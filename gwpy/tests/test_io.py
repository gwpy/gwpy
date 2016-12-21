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
import tempfile

from compat import unittest

from gwpy.io import datafind
from gwpy.io.cache import (Cache, CacheEntry, cache_segments)
from gwpy.segments import (Segment, SegmentList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_GWF_FILE = os.path.join(os.path.split(__file__)[0], 'data',
                          'HLV-GW100916-968654552-1.gwf')


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


class CacheIoTestCase(unittest.TestCase):
    @staticmethod
    def make_cache():
        segs = SegmentList()
        cache = Cache()
        for seg in [(0, 1), (1, 2), (4, 5)]:
            d = seg[1] - seg[0]
            f = 'A-B-%d-%d.tmp' % (seg[0], d)
            cache.append(CacheEntry.from_T050017(f))
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


class DataFindIoTestCase(unittest.TestCase):
    def test_num_channels(self):
        try:
            self.assertEqual(datafind.num_channels(TEST_GWF_FILE), 3)
        except ImportError as e:
            self.skipTest(str(e))

    def test_get_channel_type(self):
        try:
            self.assertEqual(datafind.get_channel_type(
                'L1:LDAS-STRAIN', TEST_GWF_FILE), 'proc')
        except ImportError as e:
            self.skipTest(str(e))

    def test_channel_in_frame(self):
        try:
            self.assertTrue(
                datafind.channel_in_frame('L1:LDAS-STRAIN', TEST_GWF_FILE))
        except ImportError as e:
            self.skipTest(str(e))
        else:
            self.assertFalse(
                datafind.channel_in_frame('X1:NOT-IN_FRAME', TEST_GWF_FILE))

    def test_on_tape(self):
        self.assertFalse(datafind.on_tape(TEST_GWF_FILE))
        self.assertFalse(datafind.on_tape(
            CacheEntry.from_T050017(TEST_GWF_FILE)))

if __name__ == '__main__':
    unittest.main()
