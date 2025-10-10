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

"""Tests for :mod:`gwpy.table.io.sql`."""

from ...testing.errors import pytest_skip_flaky_network
from .. import EventTable


@pytest_skip_flaky_network
def test_fetch_gwosc():
    """Test fetching a GWOSC event catalog."""
    table = EventTable.fetch(
        "GWTC-1-confident",
        source="gwosc",
    )
    assert len(table)
    assert {
        "mass_1_source",
        "luminosity_distance",
        "chi_eff",
    }.intersection(table.colnames)

    # check unit parsing worked
    assert table["luminosity_distance"].unit == "Mpc"


@pytest_skip_flaky_network
def test_fetch_open_data_kwargs():
    """Test fetching a GWOSC event catalog with ``where`` and ``columns``."""
    table = EventTable.fetch_open_data(
        "GWTC-1-confident",
        where="mass_1_source < 5",
        columns=[
            "name",
            "mass_1_source",
            "mass_2_source",
            "luminosity_distance",
        ],
    )
    assert len(table) == 1
    assert table[0]["name"] == "GW170817-v3"
    assert set(table.colnames) == {
        "name",
        "mass_1_source",
        "mass_2_source",
        "luminosity_distance",
    }
