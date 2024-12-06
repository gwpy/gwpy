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

from ...testing.utils import (
    TEST_GWF_FILE,
    assert_segmentlist_equal,
)
from .. import gwf as io_gwf
from ..cache import file_segment

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

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
        # backend-agnostic selection (require LDASTools.frameCPP just to
        # skip when there are NO GWF backends installed)
        pytest.param(
            None,  # choose any backend
            marks=pytest.mark.requires("LDAStools.frameCPP"),
            id="any",
        ),
    ] + list(map(_backend_param, io_gwf.BACKENDS)),
)


def test_identify_gwf():
    assert io_gwf.identify_gwf('read', TEST_GWF_FILE, None) is True
    with open(TEST_GWF_FILE, 'rb') as gwff:
        assert io_gwf.identify_gwf('read', None, gwff) is True
    assert not io_gwf.identify_gwf('read', None, None)


@parametrize_gwf_backends
def test_iter_channel_names(backend):
    # maybe need something better?
    from types import GeneratorType
    names = io_gwf.iter_channel_names(
        TEST_GWF_FILE,
        backend=backend,
    )
    assert isinstance(names, GeneratorType)
    assert list(names) == TEST_CHANNELS


@parametrize_gwf_backends
def test_get_channel_names(backend):
    assert io_gwf.get_channel_names(
        TEST_GWF_FILE,
        backend=backend,
    ) == TEST_CHANNELS


@parametrize_gwf_backends
def test_num_channels(backend):
    assert io_gwf.num_channels(
        TEST_GWF_FILE,
        backend=backend,
    ) == 3


@parametrize_gwf_backends
def test_get_channel_type(backend):
    assert io_gwf.get_channel_type(
        'L1:LDAS-STRAIN',
        TEST_GWF_FILE,
        backend=backend,
    ) == 'proc'
    with pytest.raises(
        ValueError,
        match=(
            "^'X1:NOT-IN_FRAME' not found in table-of-contents "
            f"for {TEST_GWF_FILE}$"
        ),
    ):
        io_gwf.get_channel_type(
            'X1:NOT-IN_FRAME',
            TEST_GWF_FILE,
            backend=backend,
        )


@parametrize_gwf_backends
def test_channel_in_frame(backend):
    assert io_gwf.channel_in_frame(
        'L1:LDAS-STRAIN',
        TEST_GWF_FILE,
        backend=backend,
    ) is True
    assert io_gwf.channel_in_frame(
        'X1:NOT-IN_FRAME',
        TEST_GWF_FILE,
        backend=backend,
    ) is False


@parametrize_gwf_backends
def test_data_segments(backend):
    assert_segmentlist_equal(
        io_gwf.data_segments(
            [TEST_GWF_FILE],
            "L1:LDAS-STRAIN",
            backend=backend,
        ),
        [file_segment(TEST_GWF_FILE)],
    )
    with pytest.warns(UserWarning):
        assert_segmentlist_equal(
            io_gwf.data_segments(
                [TEST_GWF_FILE],
                "X1:BAD-NAME",
                warn=True,
                backend=backend,
            ),
            [],
        )
