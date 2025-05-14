# Copyright (c) 2019-2025 Cardiff University
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

"""Tests for :mod:`gwpy.utils.enum`."""

import numpy
import pytest

from .. import enum as gwpy_enum


class _MyEnum(gwpy_enum.NumpyTypeEnum):
    INT16 = 100
    FLOAT32 = 200


class TestNumpyTypeEnum:
    """Test `gwpy.utils.enum.NumpyTypeEnum`."""

    TEST_CLASS = _MyEnum

    def test_dtype(self):
        """Test `NumpyTypeEnum.dtype` property."""
        assert self.TEST_CLASS.INT16.dtype is numpy.dtype("int16")

    def test_type(self):
        """Test `NumpyTypeEnum.type` property."""
        assert self.TEST_CLASS.INT16.type is numpy.int16

    @pytest.mark.parametrize("arg", [
        "INT16",
        "int16",
        numpy.int16,
        numpy.dtype("int16"),
    ])
    def test_find(self, arg):
        """Test :meth:`NumpyTypeEnum.find` method."""
        assert self.TEST_CLASS.find(arg) is self.TEST_CLASS.INT16

    def test_find_value(self):
        """Test :meth:`NumpyTypeEnum.find` method with value.

        This is a round-about way of testing that .find() can take in an
        the enum value (integer) for TEST_CLASS, and return just the enum
        """
        assert self.TEST_CLASS.find(
            self.TEST_CLASS.INT16.value,
        ) is self.TEST_CLASS.INT16

    def test_find_errors(self):
        """Test :meth:`NumpyTypeEnum.find` method error handling."""
        with pytest.raises(
            ValueError,
            match=f"^'blah' is not a valid {self.TEST_CLASS.__name__}$",
        ):
            self.TEST_CLASS.find("blah")
