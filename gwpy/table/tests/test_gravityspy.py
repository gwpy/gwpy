# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

from ssl import SSLError

from six.moves.urllib.error import URLError

import pytest

from ...testing import utils
from .. import GravitySpyTable
from .test_table import TestEventTable as _TestEventTable

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

JSON_RESPONSE = {
    "url4": [u"https://panoptes-uploads.zooniverse.org/production/"
             "subject_location/08895951-ea30-4cf7-9374-135a335afe0e.png"],
    "peak_frequency": [84.4759674072266],
    "links_subjects": [5740011.0],
    "ml_label": [u"Scratchy"],
    "searchedID": [u"8FHTgA8MEu"],
    "snr": [8.96664047241211],
    "gravityspy_id": [u"8FHTgA8MEu"],
    "searchedzooID": [5740011.0],
    "ifo": [u"H1"],
    "url3": [u"https://panoptes-uploads.zooniverse.org/production/"
             "subject_location/415dde44-3109-434c-b3ad-b722a879c159.png"],
    "url2": [u"https://panoptes-uploads.zooniverse.org/production/"
             "subject_location/09ebb6f4-e839-466f-80a1-64d79ac4d934.png"],
    "url1": [u"https://panoptes-uploads.zooniverse.org/production/"
             "subject_location/5e89d817-583c-4646-8e6c-9391bb99ad41.png"],
}


@utils.skip_minimum_version('astropy', '2.0.4')
class TestGravitySpyTable(_TestEventTable):
    TABLE = GravitySpyTable

    def test_search(self):
        try:
            table = self.TABLE.search(gravityspy_id="8FHTgA8MEu", howmany=1)
        except (URLError, SSLError) as e:
            pytest.skip(str(e))

        utils.assert_table_equal(table, self.TABLE(JSON_RESPONSE))
