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

"""Unit test for detector module
"""

from urllib2 import URLError

from compat import unittest

import numpy

from astropy import units

from gwpy import version
from gwpy.detector import Channel
from gwpy.utils import with_import

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

NDSHOST = 'nds.ligo-la.caltech.edu'


class ChannelTests(unittest.TestCase):
    """`TestCase` for the timeseries module
    """
    channel = 'L1:PSL-ISS_PDB_OUT_DQ'

    def test_empty_create(self):
        new = Channel('')
        self.assertTrue(str(new) == '')
        self.assertTrue(new.sample_rate is None)
        self.assertTrue(new.dtype is None)

    def test_create(self):
        new = Channel('L1:LSC-DARM_ERR', sample_rate=16384, unit='m')
        self.assertTrue(str(new) == 'L1:LSC-DARM_ERR')
        self.assertTrue(new.ifo == 'L1')
        self.assertTrue(new.system == 'LSC')
        self.assertTrue(new.subsystem == 'DARM')
        self.assertTrue(new.signal == 'ERR')
        self.assertTrue(new.sample_rate == units.Quantity(16384, 'Hz'))
        self.assertTrue(new.unit == units.meter)
        self.assertTrue(new.texname == r'L1:LSC-DARM\_ERR')

    def test_query(self):
        try:
            new = Channel.query(self.channel)
        except URLError as e:
            msg = str(e)
            if ('timed out' in msg.lower() or
                'connection reset' in msg.lower()):
                self.skipTest(msg)
            raise
        except Exception as e:
            try:
                import kerberos
            except ImportError:
                self.skipTest('Channel.query() requires kerberos '
                                        'to be installed')
            else:
                if isinstance(e, kerberos.GSSError):
                    self.skipTest(str(e))
                else:
                    raise
        self.assertTrue(str(new) == self.channel)
        self.assertTrue(new.ifo == self.channel.split(':', 1)[0])
        self.assertTrue(new.sample_rate == units.Quantity(32768, 'Hz'))
        self.assertTrue(new.texname == self.channel.replace('_', r'\_'))

    def test_query_nds2(self):
        try:
            import nds2
        except ImportError as e:
            self.skipTest(str(e))
        try:
            from gwpy.io import kerberos
            kerberos.which('kinit')
        except ValueError as e:
            self.skipTest(str(e))
        try:
            new = Channel.query_nds2(self.channel, host=NDSHOST,
                                     type=nds2.channel.CHANNEL_TYPE_RAW)
        except IOError as e:
            self.skipTest(str(e))
        self.assertTrue(str(new) == self.channel)
        self.assertTrue(new.ifo == self.channel.split(':', 1)[0])
        self.assertTrue(new.sample_rate == units.Quantity(32768, 'Hz'))
        self.assertTrue(new.type == 'raw')
        self.assertTrue(new.texname == self.channel.replace('_', r'\_'))

    def test_nds2_conversion(self):
        try:
            import nds2
        except ImportError as e:
            self.skipTest(str(e))
        else:
            try:
                conn = nds2.connection(NDSHOST)
            except Exception as f:
                self.skipTest(str(f))
            else:
                nds2channel = conn.find_channels(self.channel)[0]
                new = Channel.from_nds2(nds2channel)
                self.assertTrue(str(new) == self.channel)
                self.assertTrue(new.ifo == self.channel.split(':', 1)[0])
                self.assertTrue(new.sample_rate == units.Quantity(32768, 'Hz'))

    def test_fmcs_parse(self):
        new = Channel('LVE-EX:X3_810BTORR.mean,m-trend')
        self.assertEqual(new.ifo, None)
        self.assertEqual(new.name, 'LVE-EX:X3_810BTORR.mean')
        self.assertEqual(new.trend, 'mean')
        self.assertEqual(new.type, 'm-trend')


if __name__ == '__main__':
    unittest.main()
