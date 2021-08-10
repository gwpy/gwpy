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

from unittest import mock
from urllib.error import HTTPError

import pytest

from ...testing import utils
from ...testing.errors import pytest_skip_network_error
from .. import GravitySpyTable
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
            remote_timeout=60,
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

    @mock.patch(
        "gwpy.table.gravityspy.get_readable_fileobj",
        side_effect=HTTPError(None, 500, "test", None, None),
    )
    def test_search_error(self, _):
        """Test `GravitySpyTable.search` error handling
        """
        with pytest.raises(HTTPError) as exc:
            self.TABLE.search(gravityspy_id="abcde")
        assert str(exc.value) == (
            "HTTP Error 500: test, confirm the gravityspy_id is valid"
        )

    @utils.skip_missing_dependency("pandas")
    @pytest_skip_network_error
    def test_download(self, tmp_path):
        """Test `GravitySpyTable.download`
        """
        table = self.search()
        table.download(download_path=tmp_path, ifos=TEST_IFO)
        assert (
            tmp_path
            / "{}_{}_spectrogram_0.5.png".format(TEST_IFO, TEST_ID)
        ).is_file()
