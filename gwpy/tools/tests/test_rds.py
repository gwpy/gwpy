# Copyright (c) 2024-2025 Cardiff University
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

"""Tests for :mod:`gwpy.tools.rds`."""

import os
from unittest import mock

import pytest

from gwpy.testing.utils import skip_kerberos_credential
from gwpy.timeseries import TimeSeries
from gwpy.tools.rds import main as gwpy_rds


@pytest.mark.requires("nds2")
@skip_kerberos_credential
@mock.patch.dict(os.environ)
def test_rds(tmp_path):
    """Test the ``gwpy-rds`` tool."""
    # get using NDS2 (if datafind could have been used to start with)
    os.environ.pop("GWDATAFIND_SERVER", None)
    outfile = tmp_path / "test.h5"
    gwpy_rds([
        "-s", "1126259460",
        "-e", "1126259464",
        "-c", "H1:GDS-CALIB_STRAIN",
        "-o", str(outfile),
    ])
    data = TimeSeries.read(outfile, "H1:GDS-CALIB_STRAIN")
    hdata = data["H1:GDS-CALIB_STRAIN"]
    assert len(hdata) == 4 * 16384
    assert hdata.t0.value == 1126259460
