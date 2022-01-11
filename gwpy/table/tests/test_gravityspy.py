# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-2021)
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

"""Unit tests for `gwpy.table.gravityspy`
"""

import pytest

import requests

from ...testing.errors import pytest_skip_network_error
from .. import (
    gravityspy as table_gravityspy,
    GravitySpyTable,
)
from .test_table import TestEventTable as _TestEventTable

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_ID = "8FHTgA8MEu"
TEST_IFO = "H1"


class TestGravitySpyTable(_TestEventTable):
    TABLE = GravitySpyTable

    @classmethod
    def search(cls):
        return cls.TABLE.search(
            gravityspy_id=TEST_ID,
            howmany=1,
            ifos="H1L1",
            timeout=60,
        )

    @pytest_skip_network_error
    def test_search(self):
        """Test `GravitySpyTable.search`
        """
        # run the search
        table = self.search()

        # validate the result
        assert len(table) == 1
        t = table[0]
        assert t["gravityspy_id"] == TEST_ID
        assert t["ifo"] == TEST_IFO
        assert t["ml_label"] == "Scratchy"

    def test_search_error(self):
        """Test `GravitySpyTable.search` error handling

        Mainly to make sure that an HTTP Error is raised instead
        of a JSONDecodeError.
        """
        requests_mock = pytest.importorskip("requests_mock")
        with requests_mock.Mocker() as rmock, \
             pytest.raises(requests.HTTPError) as exc:
            rmock.get(
                "{}{}{}/?{}".format(
                    table_gravityspy.DEFAULT_HOST,
                    table_gravityspy.SEARCH_PATH,
                    table_gravityspy.SIMILARITY_SEARCH_PATH,
                    "&".join((
                        "howmany=10",
                        "imageid=abcde",
                        "era=event_time+BETWEEN+1126400000+AND+1584057618",
                        "ifo=%27H1%27%2C+%27L1%27",
                        "database=similarity_index_o3",
                    )),
                ),
                text="<!DOCTYPE html><html></html>",
                status_code=200,
                headers={
                    "Content-Type": "text/html; charset=utf-8",
                },
            )
            self.TABLE.search(gravityspy_id="abcde")
        assert str(exc.value).endswith("please check the request parameters")

    @pytest_skip_network_error
    def test_download(self, tmp_path):
        """Test `GravitySpyTable.download`
        """
        table = self.search()
        table.download(download_path=tmp_path)
        assert (
            tmp_path
            / "{}_{}_spectrogram_0.5.png".format(TEST_IFO, TEST_ID)
        ).is_file()
