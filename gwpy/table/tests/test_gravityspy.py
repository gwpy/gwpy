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

"""Unit tests for `gwpy.table`
"""

import json
import os.path
from ssl import SSLError

from six.moves.urllib.error import URLError

import pytest

from ...tests import utils
from .. import GravitySpyTable
from .test_table import TestEventTable as _TestEventTable

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_JSON_RESPONSE_FILE = os.path.join(utils.TEST_DATA_DIR,
                                       'test_json_query.json')


@utils.skip_minimum_version('astropy', '2.0.4')
class TestGravitySpyTable(_TestEventTable):
    TABLE = GravitySpyTable

    def test_search(self):
        try:
            t2 = self.TABLE.search(uniqueID="8FHTgA8MEu", howmany=1)
        except (URLError, SSLError) as e:
            pytest.skip(str(e))

        with open(TEST_JSON_RESPONSE_FILE) as f:
            table = GravitySpyTable(json.load(f))

        utils.assert_table_equal(table, t2)
