# -*- coding: utf-8 -*-
# Copyright (C) Cardiff University (2019-2022)
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

"""Tests for :mod:`gwpy.table.io.pycbc` and its integration with `EventTable`.
"""

import pytest

import numpy
from numpy.random import randn

import h5py

from .. import EventTable
from ..filter import filter_table
from ..io import pycbc as io_pycbc
from ...frequencyseries import FrequencySeries
from ...testing.utils import (
    assert_array_equal,
    assert_quantity_sub_equal,
    assert_table_equal,
)


# -- fixtures -----------------------------------

@pytest.fixture
def h5file():
    """Create an empty in-memory HDF5 file.
    """
    with h5py.File(
        'test',
        mode='w-',
        driver='core',
        backing_store=False,
    ) as h5f:
        yield h5f


@pytest.fixture
def pycbclivetable():
    """A populated `EventTable` in PyCBC format.
    """
    names = [
        'a',
        'b',
        'c',
        'chisq',
        'd',
        'e',
        'f',
        'mass1',
        'mass2',
        'snr',
    ]
    rows = []
    for i, name in enumerate(names):
        rows.append(randn(100) * 1000)
    return EventTable(rows, names=names)


@pytest.fixture
def pycbclivepsd():
    """A mock PSD.
    """
    return FrequencySeries(randn(1000), df=1)


@pytest.fixture
def pycbclivefile(tmp_path, pycbclivetable, pycbclivepsd):
    """A fully-formed PyCBC-format HDF5 file.
    """
    # create table
    loudest = (pycbclivetable['snr'] > 500).nonzero()[0]

    # manually create pycbc_live-format HDF5 file
    tmp = tmp_path / "X1-Live-0-0.hdf"
    with h5py.File(tmp, "w") as h5f:
        group = h5f.create_group('X1')
        for col in pycbclivetable.columns:
            group.create_dataset(data=pycbclivetable[col], name=col)
        group.create_dataset('loudest', data=loudest)
        group.create_dataset('psd', data=pycbclivepsd.value)
        group['psd'].attrs['delta_f'] = pycbclivepsd.df.to('Hz').value

    return tmp


# -- internal tests -----------------------------


@pytest.mark.parametrize(("filename", "result"), [
    ("X1-Live-0-0.h5", True),
    ("X1-Live-0-0.hdf5", True),
    ("X1-Live-0-0.h5", True),
    ("X1-MY_DATA-0-0.h5", False),
])
def test_idenfity_pycbc_live(tmp_path, filename, result):
    path = str(tmp_path / filename)
    h5py.File(path, "w").close()
    assert io_pycbc.identify_pycbc_live("read", path, None) is result
    with open(path, "rb") as h5f:  # check with open file as well
        assert io_pycbc.identify_pycbc_live("read", path, h5f) is result


def test_empty_hdf5_file(h5file):
    """Check that :func:`empty_hdf5_file` works.
    """
    assert io_pycbc.empty_hdf5_file(h5file)


def test_empty_hdf5_file_group(h5file):
    """Check that :func:`empty_hdf5_file` works with (empty) groups.
    """
    h5file.create_group("H1")
    assert io_pycbc.empty_hdf5_file(h5file)
    assert io_pycbc.empty_hdf5_file(h5file, ifo="H1")


def test_empty_hdf5_file_datasets(h5file):
    """Check that :func:`empty_hdf5_file` works with (empty) datasets.
    """
    h1group = h5file.create_group("H1")
    h1group.create_dataset("psd", data=numpy.empty(10))
    assert io_pycbc.empty_hdf5_file(h5file, ifo="H1")

    h1group.create_dataset("gates", data=numpy.empty(10))
    assert io_pycbc.empty_hdf5_file(h5file, ifo="H1")

    h1group.create_dataset("time", data=numpy.empty(10))
    assert not io_pycbc.empty_hdf5_file(h5file, ifo="H1")


# -- EventTable integration tests ---------------

@pytest.mark.parametrize("fmt", [
    None,  # should default to hdf5.pycbc_live as the highest-priority format
    "hdf5.pycbc_live",
])
def test_read_pycbc_live(pycbclivetable, pycbclivefile, fmt):
    """Check that `EventTable` can read a PyCBC-Live file.
    """
    table = EventTable.read(pycbclivefile, format=fmt)
    assert_table_equal(pycbclivetable, table)
    assert table.meta['ifo'] == 'X1'


def test_read_pycbc_live_kwargs(pycbclivetable, pycbclivefile):
    """Check that `EventTable` can read a PyCBC-Live file using keywords.
    """
    table = EventTable.read(
        pycbclivefile,
        format='hdf5.pycbc_live',
        ifo='X1',
    )
    assert_table_equal(pycbclivetable, table)


def test_read_pycbc_live_loudest(pycbclivetable, pycbclivefile):
    """Check that `EventTable` can read the `loudest` table from
    a PyCBC-Live file.
    """
    table = EventTable.read(
        pycbclivefile,
        format="hdf5.pycbc_live",
        loudest=True,
    )
    assert_table_equal(pycbclivetable.filter('snr > 500'), table)


def test_read_pycbc_live_extended_metadata(
    pycbclivetable,
    pycbclivepsd,
    pycbclivefile,
):
    """Check that `EventTable` can read extended metadata from a PyCBC file.
    """
    table = EventTable.read(
        pycbclivefile,
        format="hdf5.pycbc_live",
        extended_metadata=True,  # default
    )
    assert_table_equal(pycbclivetable, table)
    assert_array_equal(
        table.meta['loudest'],
        (pycbclivetable['snr'] > 500).nonzero()[0],
    )
    assert_quantity_sub_equal(
        table.meta['psd'],
        pycbclivepsd,
        exclude=['name', 'channel', 'unit', 'epoch'])


def test_read_pycbc_live_extended_metadata_false(
    pycbclivetable,
    pycbclivefile,
):
    """Check that `EventTable` can read a PyCBC file without extended metadata.
    """
    # check extended_metadata=False works
    table = EventTable.read(
        pycbclivefile,
        format="hdf5.pycbc_live",
        extended_metadata=False,
    )
    assert table.meta == {'ifo': 'X1'}


def test_read_pycbc_live_multiple_ifos(
    pycbclivetable,
    pycbclivefile,
):
    """Check that `EventTable` can handle multiple IFOs in a PyCBC-Live file
    """
    with h5py.File(pycbclivefile, "r+") as h5f:
        h5f.create_group('Z1')
    with pytest.raises(ValueError) as exc:
        EventTable.read(pycbclivefile, format="hdf5.pycbc_live")
    assert str(exc.value).startswith(
        'PyCBC live HDF5 file contains dataset groups')

    # but check that we can still read the original
    table = EventTable.read(
        pycbclivefile,
        format='hdf5.pycbc_live',
        ifo='X1',
    )
    assert_table_equal(pycbclivetable, table)


def test_read_pycbc_live_processed_columns(
    pycbclivetable,
    pycbclivefile,
):
    """Check that `EventTable` can read processed columns from a PyCBC file.
    """
    # assert processed colums works
    table = EventTable.read(
        pycbclivefile,
        format="hdf5.pycbc_live",
        ifo="X1",
        columns=["mchirp", "new_snr"],
    )
    mchirp = (
        (pycbclivetable['mass1'] * pycbclivetable['mass2']) ** (3/5.)
        / (pycbclivetable['mass1'] + pycbclivetable['mass2']) ** (1/5.)
    )
    assert_array_equal(table['mchirp'], mchirp)


def test_read_pycbc_live_selection_columns(
    pycbclivetable,
    pycbclivefile,
):
    """Check that the selection and columns kwargs work when
    reading from a PyCBC-Live file.
    """
    # test with selection and columns
    table = EventTable.read(
        pycbclivefile,
        format='hdf5.pycbc_live',
        ifo='X1',
        selection='snr>.5',
        columns=("a", "b", "mass1"),
    )
    assert_table_equal(
        table,
        filter_table(pycbclivetable, 'snr>.5')[("a", "b", "mass1")],
    )


def test_read_pycbc_live_regression_1081(
    pycbclivetable,
    pycbclivefile,
):
    """Check against regression of gwpy/gwpy#1081.
    """
    table = EventTable.read(
        pycbclivefile,
        format='hdf5.pycbc_live',
        ifo='X1',
        selection='snr>.5',
        columns=("a", "b", "snr"),
    )
    assert_table_equal(
        table,
        filter_table(pycbclivetable, 'snr>.5')[("a", "b", "snr")],
    )
