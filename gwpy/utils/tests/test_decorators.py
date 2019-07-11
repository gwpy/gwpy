# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018-2019)
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

"""Tests for :mod:`gwpy.utils.decorators`
"""

import pytest

from .. import decorators


def test_return_as():
    """Test `gwpy.utils.decorators.return_as` works
    """
    @decorators.return_as(float)
    def myfunc(value):
        return int(value)

    result = myfunc(1.5)

    # check type was cast
    assert isinstance(result, float)
    # check result is still the same
    assert result == 1.0


def test_return_as_error():
    """Test that `gwpy.utils.decorators.return_as` error handling works
    """
    @decorators.return_as(int)
    def myfunc(value):
        return str(value)

    with pytest.raises(ValueError) as exc:
        myfunc('test')
    assert 'failed to cast return from myfunc as int: ' in str(exc.value)
