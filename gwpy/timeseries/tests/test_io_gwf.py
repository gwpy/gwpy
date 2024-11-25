# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2023)
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

"""Tests for :mod:`gwpy.timeseries.io.gwf`.
"""

import os
from unittest import mock

import pytest

from ..io import gwf as ts_io_gwf


@mock.patch.dict("os.environ")
@pytest.mark.requires("framel")  # need at least one library
def test_get_default_gwf_api():
    """Test that :func:`get_default_gwf_api` returns a sensible value.
    """
    os.environ.pop("GWPY_FRAME_LIBRARY", None)
    assert ts_io_gwf.get_default_gwf_api() in ts_io_gwf.APIS


@mock.patch.dict("os.environ")
@pytest.mark.parametrize("library", [
    pytest.param('LALFrame', marks=pytest.mark.requires("lalframe")),
    pytest.param('FrameCPP', marks=pytest.mark.requires("LDAStools.frameCPP")),
    pytest.param('FrameL', marks=pytest.mark.requires("framel")),
])
def test_get_default_gwf_api_environ(library):
    os.environ["GWPY_FRAME_LIBRARY"] = library
    assert ts_io_gwf.get_default_gwf_api() == library.lower()


@mock.patch.dict("os.environ")
@pytest.mark.requires("framel")  # need at least one library
def test_get_default_gwf_api_environ_bad():
    """Test that :func:`get_default_gwf_api_environ` returns a valid value
    even when the environment variable has a bad one.
    """
    os.environ["GWPY_FRAME_LIBRARY"] = "blahblahblah"
    assert ts_io_gwf.get_default_gwf_api() in ts_io_gwf.APIS
