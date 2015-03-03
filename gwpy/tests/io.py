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
import sys
if sys.version_info < (2, 7):
    import unittest2 as unittest
else:
    import unittest

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class IoTests(unittest.TestCase):

    def test_nds2_host_order_none(self):
        """Test `host_resolution_order` with `None` IFO
        """
        try:
            from ..io import nds
        except ImportError as e:
            self.skipTest(str(e))
        hro = nds.host_resolution_order(None, env=None)
        self.assertListEqual(hro, [('nds.ligo.caltech.edu', 31200)])

    def test_nds2_host_order_ifo(self):
        """Test `host_resolution_order` with `ifo` argument
        """
        try:
            from ..io import nds
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
            from ..io import nds
        except ImportError as e:
            self.skipTest(str(e))
        os.environ['NDSSERVER'] = 'test1.ligo.org:80,test2.ligo.org:43'
        hro = nds.host_resolution_order(None)
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43)])
        hro = nds.host_resolution_order('L1')
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43)])

    def test_nds2_host_order_env(self):
        """Test `host_resolution_order` with non-default env set
        """
        try:
            from ..io import nds
        except ImportError as e:
            self.skipTest(str(e))
        os.environ['TESTENV'] = 'test1.ligo.org:80,test2.ligo.org:43'
        hro = nds.host_resolution_order(None, env='TESTENV')
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43)])

    def test_nds2_host_order_epoch(self):
        """Test `host_resolution_order` with old GPS epoch
        """
        try:
            from ..io import nds
        except ImportError as e:
            self.skipTest(str(e))
        # test kwarg doesn't change anything
        hro = nds.host_resolution_order('L1', epoch='now')
        self.assertListEqual(
            hro, [('nds.ligo-la.caltech.edu', 31200),
                  ('nds.ligo.caltech.edu', 31200)])
        # test old epoch puts CIT ahead of LLO
        hro = nds.host_resolution_order('L1', epoch='Jan 1 2015')
        self.assertListEqual(
            hro, [('nds.ligo.caltech.edu', 31200),
                  ('nds.ligo-la.caltech.edu', 31200)])
        # test epoch doesn't operate with env
        os.environ['TESTENV'] = 'test1.ligo.org:80,test2.ligo.org:43'
        hro = nds.host_resolution_order('L1', epoch='now', env='TESTENV')
        self.assertListEqual(
            hro, [('test1.ligo.org', 80), ('test2.ligo.org', 43)])



if __name__ == '__main__':
    unittest.main()
