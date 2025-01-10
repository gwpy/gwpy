# Copyright (C) Cardiff University (2024-)
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

"""Tests for :mod:`gwpy.io.gwf.backend`.
"""

import os
from unittest import mock

import pytest

from .. import gwf as io_gwf

BACKEND_LIBRARY = {
    "frameCPP": "LDAStools.frameCPP",
    "FrameL": "framel",
    "LALFrame": "lalframe",
}


@mock.patch.dict("os.environ")
@pytest.mark.requires(BACKEND_LIBRARY["LALFrame"])  # need at least one backend
def test_get_backend():
    """Test that :func:`get_backend` returns a sensible value.
    """
    os.environ.pop("GWPY_FRAME_LIBRARY", None)
    assert io_gwf.get_backend() in io_gwf.BACKENDS


@mock.patch.dict("os.environ")
@pytest.mark.parametrize("backend", [
    pytest.param(backend, marks=pytest.mark.requires(BACKEND_LIBRARY[backend]))
    for backend in io_gwf.BACKENDS
])
def test_get_backend_environ(backend):
    """Test that :func:`get_backend` reads the environment poroperly.
    """
    os.environ["GWPY_FRAME_LIBRARY"] = backend
    assert io_gwf.get_backend() == backend


@mock.patch.dict("os.environ")
@pytest.mark.requires(BACKEND_LIBRARY["frameCPP"])  # need at least one backend
def test_get_backend_environ_bad():
    """Test that :func:`get_backend_environ` returns a valid value
    even when the environment variable has a bad one.
    """
    os.environ["GWPY_FRAME_LIBRARY"] = "blahblahblah"
    assert io_gwf.get_backend() in io_gwf.BACKENDS


@pytest.mark.requires(BACKEND_LIBRARY["frameCPP"])
def test_get_backend_function():
    """Test that `get_backend_function` works.
    """
    from gwpy.io.gwf import framecpp as io_framecpp
    func = io_gwf.get_backend_function("create_frame", backend="frameCPP")
    assert func is io_framecpp.create_frame


def test_get_backend_function_badbackend():
    """Test that `get_backend_function` formats errors correctly.
    """
    with pytest.raises(
        ImportError,
        match="No module named 'gwpy.io.gwf.notimplemented'",
    ):
        io_gwf.get_backend_function("notimplemented", backend="notimplemented")


def test_get_backend_function_notimplemented():
    """Test that `get_backend_function` formats errors correctly.
    """
    with pytest.raises(
        NotImplementedError,
        match="'notimplemented'",
    ):
        io_gwf.get_backend_function("notimplemented")
