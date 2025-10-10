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

"""Tests for `Table.read` and `Table.write` with ``format='gwf'``."""

import pytest

from ...testing import utils
from ..filter import filter_table
from .utils import (
    TABLE_CLASSES,
    random_table,
)

# -- test config ---------------------

def pytest_generate_tests(metafunc):
    """Parametrize the ``table`` fixture for each valid table type."""
    if "table" in metafunc.fixturenames:
        metafunc.parametrize("table", TABLE_CLASSES, indirect=True)


@pytest.fixture
def table(request):
    """Create a random table with the right columns."""
    return random_table(
        length=100,
        names=["time", "frequency", "snr"],
        table_class=request.param,
    )


@pytest.fixture
def gwf_file(table, tmp_path):
    """Write the ``table`` to a (temporary) file and return the path.

    Enables read tests.
    """
    tmp = tmp_path / "table.gwf"
    table.write(tmp, "test")
    return tmp


# -- tests ---------------------------

@pytest.mark.requires("LDAStools.frameCPP")
def test_read_write_gwf(table):
    """Test `Table.read` and `Table.write` with ``format='gwf'``."""
    utils.test_read_write(
        table,
        "gwf",
        write_args=("test",),
        read_args=("test",),
        read_kw={
            "columns": table.colnames,
        },
        autoidentify=True,
        assert_equal=utils.assert_table_equal,
        assert_kw={
            "almost_equal": True,
        },
    )


@pytest.mark.requires("LDAStools.frameCPP")
def test_read_gwf_columns_where(table, gwf_file):
    """Test `Table.read(format='gwf')` with ``columns`` and ``where``."""
    # check where works
    t2 = type(table).read(
        gwf_file,
        "test",
        columns=["frequency", "snr"],
        where="frequency>500",
    )
    utils.assert_table_equal(
        t2,
        filter_table(table, "frequency>500")["frequency", "snr"],
    )
