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

"""Unit tests for :mod:`gwpy.io.gwf`."""

import pytest

from ...testing.utils import TEST_GWF_FILE
from .. import gwf as io_gwf

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def test_open_gwf_r():
    """Test `open_gwf()` for reading."""
    mod_framecpp = pytest.importorskip("LDAStools.frameCPP")
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    assert isinstance(open_gwf(TEST_GWF_FILE), mod_framecpp.IFrameFStream)


def test_open_gwf_w(tmp_path):
    """Test `open_gwf()` for writing."""
    mod_framecpp = pytest.importorskip("LDAStools.frameCPP")
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    tmp = tmp_path / "test.gwf"
    assert isinstance(open_gwf(tmp, mode="w"), mod_framecpp.OFrameFStream)


def test_open_gwf_w_file_url(tmp_path):
    """Test `open_gwf()` for writing with a file:// URL."""
    mod_framecpp = pytest.importorskip("LDAStools.frameCPP")
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    # check that we can use a file:// URL as well
    tmp = tmp_path / "test.gwf"
    assert isinstance(
        open_gwf(tmp.as_uri(), mode="w"),
        mod_framecpp.OFrameFStream,
    )


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_a_error():
    """Test that `open_gwf()` raises an error for append mode."""
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    with pytest.raises(
        ValueError,
        match="mode must be either 'r' or 'w'",
    ):
        open_gwf("test", mode="a")


@pytest.mark.requires("LDAStools.frameCPP")
def test_create_frvect(noisy_sinusoid):
    """Test `create_frvect()`."""
    create_frvect = io_gwf.get_backend_function(
        "create_frvect",
        backend="frameCPP",
    )
    vect = create_frvect(noisy_sinusoid)
    assert vect.nData == noisy_sinusoid.size
    assert vect.nBytes == noisy_sinusoid.nbytes
    assert vect.name == noisy_sinusoid.name
    assert vect.unitY == noisy_sinusoid.unit
    xdim = vect.GetDim(0)
    assert xdim.unitX == noisy_sinusoid.xunit
    assert xdim.dx == noisy_sinusoid.dx.value
    assert xdim.startX == noisy_sinusoid.x0.value
