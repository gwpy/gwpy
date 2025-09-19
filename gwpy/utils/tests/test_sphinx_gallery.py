# Copyright (c) 2023-2025 Cardiff University
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

"""Tests for :mod:`gwpy.utils.sphinx.gallery`."""

from __future__ import annotations

import importlib.metadata
from configparser import ConfigParser
from typing import TYPE_CHECKING

import pytest

from ..sphinx import gallery

if TYPE_CHECKING:
    from pathlib import Path

# load the default entry point
GWPY_PLOT, = importlib.metadata.entry_points(name="gwpy-plot")

# example configuration
TEST_CONFIG = {
    "test-1": {
        "title": "Title 1",
        "description": "Description 1",
        "argv": "one two three --four five",
    },
}


@pytest.fixture
def ini() -> ConfigParser:
    """Format ``TEST_CONFIG`` as a `configparser.ConfigParser`."""
    ini = ConfigParser()
    ini.update(TEST_CONFIG)
    return ini


@pytest.fixture
def inifile(tmp_path, ini) -> Path:
    """Write ``ini`` to a temporary file and return the path."""
    inifile = tmp_path / "examples.ini"
    with inifile.open("w") as conf:
        ini.write(conf)
    return inifile


def test_render_entry_point_examples(tmp_path, inifile):
    """Test `render_entry_point_examples`."""
    gallery.render_entry_point_examples(
        inifile,
        tmp_path,
        entry_point=GWPY_PLOT.name,
        filename_prefix="test_",
    )

    # check gallery header
    assert (tmp_path / "GALLERY_HEADER.rst").read_text().startswith(
        "Click on a thumbnail",
    )

    test1py = (tmp_path / "test_test-1.py").read_text()

    # check title and description
    assert """
Title 1
#######

Description 1""" in test1py

    # check python function execution
    assert f"""
from {GWPY_PLOT.value.replace(':', ' import ')}
main([
    'one', 'two', 'three',
    '--four', 'five',
])""" in test1py
