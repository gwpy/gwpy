# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Unit tests for :mod:`gwpy.io.registry`."""

from .. import registry as io_registry

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def test_identify_factory():
    """Test `identify_factory`."""
    id_func = io_registry.identify_factory(".blah", ".blah2")
    assert id_func("read", None, None) is False
    assert id_func("read", "test.txt", None) is False
    assert id_func("read", "test.blah", None) is True
    assert id_func("read", "test.blah2", None) is True
    assert id_func("read", "test.blah2x", None) is False
