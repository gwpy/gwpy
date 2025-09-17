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

"""Tests for :mod:`gwpy.io.gwf`."""

from pathlib import Path
from types import GeneratorType

import pytest

from ...testing.utils import (
    TEST_GWF_FILE,
    assert_segmentlist_equal,
)
from .. import gwf as io_gwf
from ..cache import file_segment

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

TEST_CHANNELS = [
    "H1:LDAS-STRAIN",
    "L1:LDAS-STRAIN",
    "V1:h_16384Hz",
]


def _backend_param(backend, *args, **kwargs):
    """Build a `pytest.param` for this GWF backend.

    This is just a convenience to apply the xfail mark for FrameL.
    """
    marks = [
        pytest.mark.requires(f"gwpy.io.gwf.{backend.lower()}"),
    ]
    if backend == "FrameL":
        marks.append(
            pytest.mark.xfail(reason="python-framel only supports getting data"),
        )
    return pytest.param(backend, *args, marks=marks, **kwargs)


parametrize_gwf_backends = pytest.mark.parametrize(
    "backend",
    [
        # backend-agnostic selection (require lalframe just to
        # skip when there are NO GWF backends installed)
        pytest.param(
            None,  # choose any backend
            marks=pytest.mark.requires("lalframe"),
            id="any",
        ),
        *map(_backend_param, io_gwf.BACKENDS),
    ],
)


def test_identify_gwf():
    """Test :func:`gwpy.io.gwf.identify_gwf`."""
    assert io_gwf.identify_gwf("read", TEST_GWF_FILE, None) is True
    with Path(TEST_GWF_FILE).open("rb") as gwff:
        assert io_gwf.identify_gwf("read", None, gwff) is True
    assert not io_gwf.identify_gwf("read", None, None)


@parametrize_gwf_backends
def test_iter_channel_names(backend):
    """Test :func:`gwpy.io.gwf.iter_channel_names`."""
    names = io_gwf.iter_channel_names(
        TEST_GWF_FILE,
        backend=backend,
    )
    assert isinstance(names, GeneratorType)
    assert list(names) == TEST_CHANNELS


@parametrize_gwf_backends
def test_get_channel_names(backend):
    """Test :func:`gwpy.io.gwf.get_channel_names`."""
    assert io_gwf.get_channel_names(
        TEST_GWF_FILE,
        backend=backend,
    ) == TEST_CHANNELS


@parametrize_gwf_backends
def test_num_channels(backend):
    """Test :func:`gwpy.io.gwf.num_channels`."""
    assert io_gwf.num_channels(
        TEST_GWF_FILE,
        backend=backend,
    ) == 3


@parametrize_gwf_backends
def test_get_channel_type(backend):
    """Test :func:`gwpy.io.gwf.get_channel_type`."""
    assert io_gwf.get_channel_type(
        TEST_CHANNELS[0],
        TEST_GWF_FILE,
        backend=backend,
    ) == "proc"
    with pytest.raises(
        ValueError,
        match=(
            "^'X1:NOT-IN_FRAME' not found in table-of-contents "
            f"for {TEST_GWF_FILE}$"
        ),
    ):
        io_gwf.get_channel_type(
            "X1:NOT-IN_FRAME",
            TEST_GWF_FILE,
            backend=backend,
        )


@parametrize_gwf_backends
@pytest.mark.parametrize(("channel", "result"), [
    (TEST_CHANNELS[0], True),
    ("X1:NOT-IN_FRAME", False),
])
def test_channel_exists(channel, result, backend):
    """Test :func:`gwpy.io.gwf.channel_exists`."""
    assert io_gwf.channel_exists(
        TEST_GWF_FILE,
        channel,
        backend=backend,
    ) is result


@parametrize_gwf_backends
def test_data_segments(backend):
    """Test :func:`gwpy.io.gwf.data_segments`."""
    assert_segmentlist_equal(
        io_gwf.data_segments(
            [TEST_GWF_FILE],
            TEST_CHANNELS[0],
            backend=backend,
        ),
        [file_segment(TEST_GWF_FILE)],
    )


@parametrize_gwf_backends
def test_data_segments_missing(backend):
    """Test :func:`gwpy.io.gwf.data_segments` with a bad channel name."""
    with pytest.warns(
        UserWarning,
        match="'X1:BAD-NAME' not found in frame",
    ):
        assert_segmentlist_equal(
            io_gwf.data_segments(
                [TEST_GWF_FILE],
                "X1:BAD-NAME",
                warn=True,
                backend=backend,
            ),
            [],
        )
