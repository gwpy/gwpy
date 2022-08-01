# -*- coding: utf-8 -*-
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
    'H1:LDAS-STRAIN', 'L1:LDAS-STRAIN', 'V1:h_16384Hz',
]


def test_identify_gwf():
    assert io_gwf.identify_gwf('read', TEST_GWF_FILE, None) is True
    with open(TEST_GWF_FILE, 'rb') as gwff:
        assert io_gwf.identify_gwf('read', None, gwff) is True
    assert not io_gwf.identify_gwf('read', None, None)


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_r(tmp_path):
    from LDAStools import frameCPP
    assert isinstance(io_gwf.open_gwf(TEST_GWF_FILE), frameCPP.IFrameFStream)


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_w(tmp_path):
    from LDAStools import frameCPP
    tmp = tmp_path / "test.gwf"
    assert isinstance(io_gwf.open_gwf(tmp, mode='w'), frameCPP.OFrameFStream)


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_w_file_url(tmp_path):
    from LDAStools import frameCPP
    # check that we can use a file:// URL as well
    tmp = tmp_path / "test.gwf"
    assert isinstance(
        io_gwf.open_gwf(tmp.as_uri(), mode='w'),
        frameCPP.OFrameFStream,
    )


@pytest.mark.requires("LDAStools.frameCPP")
def test_open_gwf_a_error():
    with pytest.raises(ValueError):
        io_gwf.open_gwf('test', mode='a')


@pytest.mark.requires("LDAStools.frameCPP")
def test_create_frvect(noisy_sinusoid):
    vect = io_gwf.create_frvect(noisy_sinusoid)
    assert vect.nData == noisy_sinusoid.size
    assert vect.nBytes == noisy_sinusoid.nbytes
    assert vect.name == noisy_sinusoid.name
    assert vect.unitY == noisy_sinusoid.unit
    xdim = vect.GetDim(0)
    assert xdim.unitX == noisy_sinusoid.xunit
    assert xdim.dx == noisy_sinusoid.dx.value
    assert xdim.startX == noisy_sinusoid.x0.value


@pytest.mark.requires("LDAStools.frameCPP")
def test_iter_channel_names():
    # maybe need something better?
    from types import GeneratorType
    names = io_gwf.iter_channel_names(TEST_GWF_FILE)
    assert isinstance(names, GeneratorType)
    assert list(names) == TEST_CHANNELS


@pytest.mark.requires("LDAStools.frameCPP")
def test_get_channel_names():
    assert io_gwf.get_channel_names(TEST_GWF_FILE) == TEST_CHANNELS


@pytest.mark.requires("LDAStools.frameCPP")
def test_num_channels():
    assert io_gwf.num_channels(TEST_GWF_FILE) == 3


@pytest.mark.requires("LDAStools.frameCPP")
def test_get_channel_type():
    assert io_gwf.get_channel_type('L1:LDAS-STRAIN', TEST_GWF_FILE) == 'proc'
    with pytest.raises(ValueError) as exc:
        io_gwf.get_channel_type('X1:NOT-IN_FRAME', TEST_GWF_FILE)
    assert str(exc.value) == (
        f"'X1:NOT-IN_FRAME' not found in table-of-contents for {TEST_GWF_FILE}"
    )


@pytest.mark.requires("LDAStools.frameCPP")
def test_channel_in_frame():
    assert io_gwf.channel_in_frame('L1:LDAS-STRAIN', TEST_GWF_FILE) is True
    assert io_gwf.channel_in_frame('X1:NOT-IN_FRAME', TEST_GWF_FILE) is False


@pytest.mark.requires("LDAStools.frameCPP")
def test_data_segments():
    assert_segmentlist_equal(
        io_gwf.data_segments([TEST_GWF_FILE], "L1:LDAS-STRAIN"),
        [file_segment(TEST_GWF_FILE)],
    )
    with pytest.warns(UserWarning):
        assert_segmentlist_equal(
            io_gwf.data_segments([TEST_GWF_FILE], "X1:BAD-NAME", warn=True),
            [],
        )
