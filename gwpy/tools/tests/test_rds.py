# Copyright (c) 2024-2026 Cardiff University
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

"""Tests for :mod:`gwpy.tools.rds`."""

import pytest

from ...testing import utils as test_utils
from ...testing.errors import pytest_skip_network_error
from ...timeseries import TimeSeries, TimeSeriesDict
from ..rds import (
    create_parser,
    main as gwpy_rds,
)


@pytest_skip_network_error
def test_rds(tmp_path):
    """Test the ``gwpy-rds`` tool."""
    ifo = "H1"
    outfile = tmp_path / "test.h5"
    gwpy_rds([
        "1126259460",
        "1126259464",
        ifo,
        "-o", str(outfile),
        "-O", "format=hdf5",
        "-O", "sample_rate=4096",
        "-O", "version=4",
    ])
    data = TimeSeries.read(outfile, ifo)
    assert data.name == "H1:Strain"
    assert data.span == (1126259460, 1126259464)


def test_rds_read_from_file(tmp_path):
    """Test the ``gwpy-rds`` tool with -i/--input-file option."""
    # Read from test HDF5 file
    # The test file has data from 968654552 to 968654553 (1 second)
    outfile = tmp_path / "test_read.h5"
    gwpy_rds([
        "968654552",  # start time in file
        "968654553",  # end time in file (file only has 1 second of data)
        "H1:LDAS-STRAIN",  # channel in test file
        "-i", test_utils.TEST_HDF5_FILE,
        "-o", str(outfile),
        "-O", "format=hdf5",
    ])
    # Verify output file was created and contains data
    assert outfile.exists()
    data = TimeSeriesDict.read(str(outfile))
    assert "H1:LDAS-STRAIN" in data


def test_rds_source_and_input_mutually_exclusive():
    """Test that --source and -i/--input-file are mutually exclusive."""
    parser = create_parser()
    with pytest.raises(SystemExit):
        # Try to use both --source and -i/--input-file
        parser.parse_args([
            "968654552",
            "968654562",
            "H1:LDAS-STRAIN",
            "-g", "gwosc",
            "-i", test_utils.TEST_HDF5_FILE,
        ])


def test_rds_input_file_not_exist():
    """Test that gwpy-rds propagates the right error when input file doesn't exist."""
    with pytest.raises(FileNotFoundError):
        gwpy_rds([
            "968654552",
            "968654562",
            "H1:LDAS-STRAIN",
            "-i", "/nonexistent/file.hdf5",
        ])


def test_rds_input_file_is_directory(tmp_path):
    """Test that gwpy-rds propagates the right error when input file is a directory."""
    subdir = tmp_path / "testdir"
    subdir.mkdir()
    with pytest.raises((IsADirectoryError, PermissionError)):
        gwpy_rds([
            "968654552",
            "968654562",
            "H1:LDAS-STRAIN",
            "-i", str(subdir),
        ])
