# Copyright (C) Cardiff University (2025-)
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

ET_PELICAN_URL = (
    "osdf:///et-gw/PUBLIC/MDC1/data/E0/E-E0_STRAIN_DATA-1000436224-2048.gwf"
)
ET_CHANNEL = "E0:STRAIN"


@pytest_skip_network_error
@pytest.mark.requires("lalframe", "requests_pelican")
def test_timeseries_read_pelican(tmp_path):
    """Check that `TimeSeries.read` can handle Pelican URLs."""
    # force astropy to use tmp_path as the temporary download directory
    with mock.patch("tempfile.gettempdir", return_value=str(tmp_path)):
        data = TimeSeries.read(
            ET_PELICAN_URL,
            ET_CHANNEL,
            cache=False,
            verbose=True,
        )
    assert data.mean().value == pytest.approx(5.778819180203806e-29)
