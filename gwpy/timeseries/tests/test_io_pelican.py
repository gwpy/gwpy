# Copyright (c) 2025 Cardiff University
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

"""Tests of pelican interaction with `TimeSeries`."""

from unittest import mock

import pytest

from ...testing.errors import pytest_skip_network_error
from .. import TimeSeries

# O3b data for LIGO-Hanford
PELICAN_URL = (
    "osdf:///gwdata/O3b/strain.4k/hdf.v1/H1/1268776960/"
    "H-H1_GWOSC_O3b_4KHZ_R1-1269358592-4096.hdf5",
)
READ_KWARGS = {
    "cache": False,
    "format": "hdf5.gwosc",
}


@pytest_skip_network_error
@pytest.mark.requires("requests_pelican")
def test_timeseries_read_pelican(tmp_path):
    """Check that `TimeSeries.read` can handle Pelican URLs."""
    # force astropy to use tmp_path as the temporary download directory
    with mock.patch("tempfile.gettempdir", return_value=str(tmp_path)):
        data = TimeSeries.read(
            PELICAN_URL,
            **READ_KWARGS,
        )
    assert data.mean().value == pytest.approx(2.768754379315686e-26, abs=1e-28)
