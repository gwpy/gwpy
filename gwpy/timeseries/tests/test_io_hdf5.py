# Copyright (c) 2026 Cardiff University
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

"""Test reading and writing HDF5 files using h5py."""

import h5py

from ...testing.utils import (
    TEST_HDF5_FILE,
)
from .. import (
    TimeSeries,
    TimeSeriesDict,
)

__author__ = "Duncan Macleod <macleoddm@cardiff.ac.uk>"

TEST_DATA_SPAN = (968654552.0, 968654553.0)


def test_hdf5_read():
    """Test reading HDF5 file using TimeSeries.read()."""
    data = TimeSeries.read(TEST_HDF5_FILE, path="H1:LDAS-STRAIN")
    assert data.size == 16384
    assert data.span == TEST_DATA_SPAN


def test_hdf5_read_h5file():
    """Test reading HDF5 file opened with h5py.File()."""
    with h5py.File(TEST_HDF5_FILE, "r") as h5file:
        data = TimeSeries.read(h5file, path="H1:LDAS-STRAIN")
    assert data.size == 16384
    assert data.span == TEST_DATA_SPAN


def test_hdf5_read_h5dataset():
    """Test reading HDF5 dataset using TimeSeries.read()."""
    with h5py.File(TEST_HDF5_FILE, "r") as h5file:
        dset = h5file["H1:LDAS-STRAIN"]
        data = TimeSeries.read(dset)
    assert data.size == 16384
    assert data.span == TEST_DATA_SPAN


def test_hdf5_read_values():
    """Test reading HDF5 dataset values using TimeSeries.read().

    Checks for regression against https://gitlab.com/gwpy/gwpy/-/issues/1843.
    """
    with h5py.File(TEST_HDF5_FILE, "r") as h5file:
        for ds in h5file.values():
            data = TimeSeries.read(ds)
            assert data.size == 16384
            assert data.span == TEST_DATA_SPAN


def test_hdf5_read_parallel_keys():
    """Test reading HDF5 datasets in parallel with a KeysView.

    Checks for regression against https://gitlab.com/gwpy/gwpy/-/issues/1843.
    """
    with h5py.File(TEST_HDF5_FILE, "r") as h5file:
        parallel = len(h5file)
        data = TimeSeriesDict.read(h5file, h5file.keys(), parallel=parallel)
        for key in h5file:
            ts = data[key]
            assert ts.size == 16384
            assert ts.span == TEST_DATA_SPAN
