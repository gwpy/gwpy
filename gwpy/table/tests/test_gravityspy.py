# Copyright (c) 2014-2017 Louisiana State University
#               2017-2021 Cardiff University
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

"""Unit tests for `gwpy.table.gravityspy`."""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlencode

import pytest
import requests

from ...testing.errors import pytest_skip_flaky_network
from .. import (
    GravitySpyTable,
    gravityspy as table_gravityspy,
)
from .test_table import TestEventTable as _TestEventTable

if TYPE_CHECKING:
    from pathlib import Path

    from requests_mock import Mocker as RequestMocker

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

TEST_ID = "8FHTgA8MEu"
TEST_IFO = "H1"


class TestGravitySpyTable(_TestEventTable[GravitySpyTable]):
    """Tests for `GravitySpyTable`."""

    TABLE: type[GravitySpyTable] = GravitySpyTable

    @classmethod
    def search(cls) -> GravitySpyTable:
        """Search for a known GravitySpy event."""
        return cls.TABLE.search(
            gravityspy_id=TEST_ID,
            howmany=1,
            ifos="H1L1",
            timeout=60,
        )

    @pytest_skip_flaky_network
    def test_search(self):
        """Test `GravitySpyTable.search`."""
        # run the search
        table = self.search()

        # validate the result
        assert len(table) == 1
        t = table[0]
        assert t["gravityspy_id"] == TEST_ID
        assert t["ifo"] == TEST_IFO
        assert t["ml_label"] == "Scratchy"

    def test_search_error(self, requests_mock: RequestMocker):
        """Test `GravitySpyTable.search` error handling.

        Mainly to make sure that an HTTP Error is raised instead
        of a JSONDecodeError.
        """
        url = (
            table_gravityspy.DEFAULT_HOST
            + table_gravityspy.SEARCH_PATH
            + table_gravityspy.SIMILARITY_SEARCH_PATH
            + "/?"
        )
        requests_mock.get(
            url + urlencode({
                "howmany": 10,
                "imageid": "abcde",
                "era": "event_time BETWEEN 1126400000 AND 1584057618",
                "ifo": "'H1', 'L1'",
                "database": "similarity_index_o3",
            }),
            text="<!DOCTYPE html><html></html>",
            status_code=200,
            headers={
                "Content-Type": "text/html; charset=utf-8",
            },
            complete_qs=True,
        )
        with pytest.raises(
            requests.HTTPError,
            match=r"please check the request parameters$",
        ):

            self.TABLE.search(gravityspy_id="abcde")

    @pytest_skip_flaky_network
    def test_download(self, tmp_path: Path):
        """Test `GravitySpyTable.download`."""
        table = self.search()
        table.download(download_path=tmp_path)
        assert (
            tmp_path
            / f"{TEST_IFO}_{TEST_ID}_spectrogram_0.5.png"
        ).is_file()
