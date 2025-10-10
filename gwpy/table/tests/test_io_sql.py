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

import pytest

from ..io import sql as io_sql


def _sql_str(expr) -> str:
    """Format a SQL expression or string for easy comparison."""
    out = str(expr).lower()
    return out.replace("\n", "")


@pytest.mark.parametrize(("kwargs", "result"), [
    # just table
    pytest.param(
        None,
        'select * from "table"',
        id="table",
    ),
    # columns
    pytest.param(
        {"columns": ["col1", "col2"]},
        'select col1, col2 from "table"',
        id="columns",
    ),
    # 'where' conditions
    pytest.param(
        {
            "columns": ["col1", "col2"],
            "where": "col3 > 4 && col4 = value",
        },
        'select col1, col2 from "table" where col3 > :col3_1 and col4 = :col4_1',
        id="where",
    ),
    # order_by
    pytest.param(
        {
            "order_by": "col5",
        },
        'select * from "table" order by col5 asc',
        id="order_by",
    ),
    # order_by_desc
    pytest.param(
        {
            "columns": ["col1", "col2"],
            "where": ["col3 < 10", "col7 = something"],
            "order_by": "col6",
            "order_by_desc": True,
        },
        'select col1, col2 from "table" where col3 < :col3_1 and col7 = :col7_1 order by col6 desc',  # noqa: E501
        id="order_by_desc",
    ),


])
@pytest.mark.requires("sqlalchemy")
def test_format_query(kwargs, result):
    """Test that `gwpy.table.io.sql.format_query` works."""
    assert _sql_str(io_sql.format_query(
        "table",
        **(kwargs or {}),
    )) == _sql_str(result)
