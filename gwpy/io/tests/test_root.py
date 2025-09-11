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

"""Tests for `gwpy.io.root`."""

import pytest

from .. import root as io_root

uproot = pytest.importorskip("uproot")


@pytest.fixture
def rootfile(tmp_path):
    """Create an empty ROOT file using `uproot.create`."""
    path = tmp_path / "test.root"
    with uproot.create(path):
        pass
    return path


def test_identify_root_fileobj(rootfile):
    """Test that `gwpy.io.root.identify_root` works with file objects."""
    with open(rootfile, "rb") as file:
        assert io_root.identify_root(
            "read",
            None,
            file,
        )


def test_identify_root_filepath(rootfile):
    """Test that `gwpy.io.root.identify_root` works with file paths."""
    assert io_root.identify_root(
        "read",
        str(rootfile),
        None,
    )


def test_identify_root_uproot(rootfile):
    """Test that `gwpy.io.root.identify_root` works with `uproot` objects."""
    with uproot.open(rootfile) as rootf:
        assert io_root.identify_root(
            "read",
            None,
            None,
            rootf,
        )
