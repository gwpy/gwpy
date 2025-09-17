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

"""Tests for :mod:`gwpy.types.index`."""

from astropy import units

from .. import Index


class TestIndex:
    """Test `gwpy.types.Index`."""

    TEST_CLASS = Index

    def test_define_regular(self):
        """Check for regression against gwpy/gwpy#1596."""
        ind_len = 1000
        a = self.TEST_CLASS.define(
            1262936373.4853957,
            0.051,
            ind_len,
        )
        assert len(a) == ind_len

    def test_define_index_length(self):
        """Check for regression against gwpy/gwpy#1506."""
        a = self.TEST_CLASS.define(
            units.Quantity(1000000000),
            units.Quantity(0.01),
            500,
        )
        assert a.is_regular()

    def test_is_regular(self):
        """Test `Index.is_regular()`."""
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], "s")
        assert a.is_regular()
        assert a[::-1].is_regular()

    def test_is_regular_scalar(self):
        """Test `Index.is_regular()` returns `False` for scalars."""
        a = self.TEST_CLASS([1])
        assert not a.is_regular()

    def test_not_is_regular(self):
        """Test `Index.is_regular()` (return `False`)."""
        b = self.TEST_CLASS([1, 2, 4, 5, 7, 8, 9])
        assert not b.is_regular()

    def test_regular(self):
        """Test `Index.regular`."""
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], "s")
        assert a.regular
        assert a.regular is a.info.meta["regular"]

    def test_getitem(self):
        """Test `Index.__getitem__()`."""
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], "Hz")
        assert type(a[0]) is units.Quantity
        assert a[0] == 1 * units.Hz
        assert isinstance(a[:2], type(a))

    def test_setitem(self):
        """Test `Index.__setitem__()`."""
        a = self.TEST_CLASS([1, 2, 3, 4, 5, 6], "Hz")
        a[0:2] = [7, 8] * a.unit
        assert list(a.value) == [7, 8, 3, 4, 5, 6]
        assert a[0] == 7 * units.Hz
