# Copyright (C) Duncan Macleod (2014-2020)
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

"""Unit tests for :mod:`gwpy.io.gwf`
"""

import pytest

from ...testing.utils import TEST_GWF_FILE
from .. import gwf as io_gwf

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_r(tmp_path):
    from LDAStools import frameCPP
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    assert isinstance(open_gwf(TEST_GWF_FILE), frameCPP.IFrameFStream)


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_w(tmp_path):
    from LDAStools import frameCPP
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    tmp = tmp_path / "test.gwf"
    assert isinstance(open_gwf(tmp, mode="w"), frameCPP.OFrameFStream)


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_w_file_url(tmp_path):
    from LDAStools import frameCPP
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    # check that we can use a file:// URL as well
    tmp = tmp_path / "test.gwf"
    assert isinstance(
        open_gwf(tmp.as_uri(), mode="w"),
        frameCPP.OFrameFStream,
    )


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_a_error():
    open_gwf = io_gwf.get_backend_function("open_gwf", backend="frameCPP")
    with pytest.raises(ValueError):
        open_gwf("test", mode="a")


@pytest.mark.requires("LDAStools.frameCPP")
def test_create_frvect(noisy_sinusoid):
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
