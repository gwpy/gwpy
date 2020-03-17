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

"""Tests for :mod:`gwpy.table.io.pycbc`
"""

import pytest

import numpy

import h5py

from ..io import pycbc as io_pycbc


@pytest.fixture
def h5file():
    with h5py.File(
        'test',
        mode='w-',
        driver='core',
        backing_store=False,
    ) as h5f:
        yield h5f


def test_empty_hdf5_file(h5file):
    assert io_pycbc.empty_hdf5_file(h5file)

    # add an IFO group
    h1group = h5file.create_group("H1")
    assert io_pycbc.empty_hdf5_file(h5file)
    assert io_pycbc.empty_hdf5_file(h5file, ifo="H1")

    # add some datasets and check that we get the correct response
    h1group.create_dataset("psd", data=numpy.empty(10))
    assert io_pycbc.empty_hdf5_file(h5file, ifo="H1")
    h1group.create_dataset("gates", data=numpy.empty(10))
    assert io_pycbc.empty_hdf5_file(h5file, ifo="H1")
    h1group.create_dataset("time", data=numpy.empty(10))
    assert not io_pycbc.empty_hdf5_file(h5file, ifo="H1")
