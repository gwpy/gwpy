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

from gwpy import version
from gwpy.io.cache import (Cache, CacheEntry, cache_segments)
from gwpy.segments import (Segment, SegmentList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class IoTests(unittest.TestCase):

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

    @staticmethod
    def make_cache():
        segs = SegmentList()
        cache = Cache()
        for seg in [(0, 1), (1, 2), (4, 5)]:
            d = seg[1] - seg[0]
            _, f = tempfile.mkstemp(prefix='A-',
                                    suffix='-%d-%d.tmp' % (seg[0], d))
            cache.append(CacheEntry.from_T050017(f))
            segs.append(Segment(*seg))
        return cache, segs

    @staticmethod
    def destroy_cache(cache):
        for f in cache.pfnlist():
            if os.path.isfile(f):
                os.remove(f)

    def test_cache_segments(self):
        # check empty input
        sl = cache_segments()
        self.assertIsInstance(sl, SegmentList)
        self.assertEquals(len(sl), 0)
        cache, segs = self.make_cache()
        try:
            # check good cache
            sl = cache_segments(cache)
            self.assertNotEquals(sl, segs)
            self.assertEquals(sl, type(segs)(segs).coalesce())
            # check bad cache
            os.remove(cache[0].path)
            sl = cache_segments(cache)
            self.assertEquals(sl, segs[1:])
            # check cache with no existing files
            sl = cache_segments(cache[:1])
            self.assertEquals(sl, SegmentList())
            # check errors
            self.assertRaises(TypeError, cache_segments, blah='blah')
            self.assertRaises(ValueError, cache_segments, cache,
                              on_missing='error')
            self.assertRaises(ValueError, cache_segments, cache,
                             on_missing='blah')
        # clean up
        finally:
            self.destroy_cache(cache)


if __name__ == '__main__':
    unittest.main()
