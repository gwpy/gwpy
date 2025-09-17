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

"""Test :mod:`gwpy.timeseries.io.gwf.utils`."""

import pytest

from ..io.gwf import utils as io_gwf_utils

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


@pytest.mark.parametrize(("value", "channels", "expected_type", "result"), [
    pytest.param(
        10,
        ["A", "B"],
        None,
        {"A": 10, "B": 10},
        id="single",
    ),
    pytest.param(
        (10, 20),
        ["A", "B"],
        None,
        {"A": 10, "B": 20},
        id="zip",
    ),
    pytest.param(
        {"A": 10, "B": 20, "C": 30},
        ["A", "B"],
        None,
        {"A": 10, "B": 20},
        id="dict",
    ),
    pytest.param(
        [10, 20],
        ["A", "B"],
        list,
        {"A": [10, 20], "B": [10, 20]},
        id="iterable",
    ),
    pytest.param(
        [[10, 20], [30, 40]],
        ["A", "B"],
        list,
        {"A": [10, 20], "B": [30, 40]},
        id="nested",
    ),
])
def test_channel_dict_kwarg(value, channels, expected_type, result):
    """Test `_channel_dict_kwarg()`."""
    assert io_gwf_utils._channel_dict_kwarg(
        value,
        channels,
        expected_type=expected_type,
    ) == result


def test_channel_dict_kwarg_error():
    """Test `_channel_dict_kwarg()` zip error handling."""
    with pytest.raises(
        ValueError,
        match="TEST is shorter than channels list",
    ):
        io_gwf_utils._channel_dict_kwarg(
            [1, 2, 3],
            ["A", "B", "C", "D"],
            varname="TEST",
        )
