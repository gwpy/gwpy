# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

from six.moves.urllib.parse import urljoin

import pytest

from ...testing.utils import (TEST_GWF_FILE, skip_missing_dependency,
                              TemporaryFilename)
from .. import gwf as io_gwf

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_CHANNELS = [
    'H1:LDAS-STRAIN', 'L1:LDAS-STRAIN', 'V1:h_16384Hz',
]


def test_identify_gwf():
    assert io_gwf.identify_gwf('read', TEST_GWF_FILE, None) is True
    with open(TEST_GWF_FILE, 'rb') as gwff:
        assert io_gwf.identify_gwf('read', None, gwff) is True
    assert not io_gwf.identify_gwf('read', None, None)


@skip_missing_dependency('LDAStools.frameCPP')
def test_open_gwf():
    from LDAStools import frameCPP
    assert isinstance(io_gwf.open_gwf(TEST_GWF_FILE), frameCPP.IFrameFStream)
    with TemporaryFilename() as tmp:
        assert isinstance(io_gwf.open_gwf(tmp, mode='w'),
                          frameCPP.OFrameFStream)
        # check that we can use a file:// URL as well
        url = urljoin('file:', tmp)
        assert isinstance(io_gwf.open_gwf(url, mode='w'),
                          frameCPP.OFrameFStream)
    with pytest.raises(ValueError):
        io_gwf.open_gwf('test', mode='a')


@skip_missing_dependency('LDAStools.frameCPP')
def test_iter_channel_names():
    # maybe need something better?
    from types import GeneratorType
    names = io_gwf.iter_channel_names(TEST_GWF_FILE)
    assert isinstance(names, GeneratorType)
    assert list(names) == TEST_CHANNELS


@skip_missing_dependency('LDAStools.frameCPP')
def test_get_channel_names():
    assert io_gwf.get_channel_names(TEST_GWF_FILE) == TEST_CHANNELS


@skip_missing_dependency('LDAStools.frameCPP')
def test_num_channels():
    assert io_gwf.num_channels(TEST_GWF_FILE) == 3


@skip_missing_dependency('LDAStools.frameCPP')
def test_get_channel_type():
    assert io_gwf.get_channel_type('L1:LDAS-STRAIN', TEST_GWF_FILE) == 'proc'
    with pytest.raises(ValueError) as exc:
        io_gwf.get_channel_type('X1:NOT-IN_FRAME', TEST_GWF_FILE)
    assert str(exc.value) == (
        'X1:NOT-IN_FRAME not found in table-of-contents for {gwf}'.format(
            gwf=TEST_GWF_FILE))


@skip_missing_dependency('LDAStools.frameCPP')
def test_channel_in_frame():
    assert io_gwf.channel_in_frame('L1:LDAS-STRAIN', TEST_GWF_FILE) is True
    assert io_gwf.channel_in_frame('X1:NOT-IN_FRAME', TEST_GWF_FILE) is False
