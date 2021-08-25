# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2019-2020)
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

"""Unit tests for :mod:`gwpy.io.mp`
"""

import pytest

from astropy.table import (Table, vstack)

from ...testing.utils import assert_table_equal
from .. import (
    mp as io_mp,
)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


@pytest.fixture
def tmp1(tmp_path):
    tmp = tmp_path / "tmp1.csv"
    tmp.write_text("a,b,c\n1,2,3\n4,5,6")
    yield str(tmp)


@pytest.fixture
def tmp2(tmp_path):
    tmp = tmp_path / "tmp2.csv"
    tmp.write_text("a,b,c\n7,8,9\n10,11,12")
    yield tmp


def test_read_multi_single(tmp1):
    """Check that serial processing works with `mp.read_multi`
    """
    tab = io_mp.read_multi(vstack, Table, tmp1, verbose=True)
    assert tab.colnames == ["a", "b", "c"]
    assert list(tab[0]) == [1, 2, 3]


def test_read_multi_list_of_one(tmp1):
    """Check that a list of one is the same as just passing the element
    """
    assert_table_equal(
        io_mp.read_multi(vstack, Table, tmp1),
        io_mp.read_multi(vstack, Table, [tmp1]),
    )


def test_read_multi_nproc(tmp1, tmp2):
    """Check that simple multiprocessing works with `mp.read_multi`
    """
    tab = io_mp.read_multi(vstack, Table, [tmp1, tmp2], nproc=2)
    assert tab.colnames == ["a", "b", "c"]
    assert list(tab["a"]) == [1, 4, 7, 10]


def test_read_multi_order_preservation(tmp1, tmp2):
    """Check that input order is preserved in `mp.read_multi`
    """
    tab = io_mp.read_multi(vstack, Table, [tmp2, tmp1], nproc=2)
    assert tab.colnames == ["a", "b", "c"]
    assert list(tab["a"]) == [7, 10, 1, 4]


def test_read_multi_error_empty():
    """Check that an `IndexError` is raised when the input list is empty
    """
    with pytest.raises(IndexError) as exc:
        io_mp.read_multi(1, int, [])
    assert str(exc.value) == "cannot read int from empty source list"


def test_read_multi_not_a_file():
    """Check that a strange input gets passed along properly
    so that errors can be raise by the reader.
    """
    with pytest.raises(TypeError):
        io_mp.read_multi(vstack, Table, None, format="ascii.csv", nproc=1)


@pytest.mark.parametrize("nproc", (
    pytest.param(1, id="serial"),
    pytest.param(2, id="multi"),
))
def test_read_multi_error_propagation(tmp1, tmp2, nproc):
    """Check that errors get raised in-place during reads
    """
    # write nonsense into the file
    with open(tmp1, "a") as tmp:
        tmp.write("blahblahblah\n1,2,3,4blah")
    # try and read it
    with pytest.raises(ValueError):
        io_mp.read_multi(vstack, Table, [tmp1, tmp2],
                         format="ascii.csv", nproc=nproc)
