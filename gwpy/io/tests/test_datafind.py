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

"""Unit tests for :mod:`gwpy.io.datafind`
"""

from __future__ import print_function

import os

from six.moves.http_client import HTTPConnection

import pytest

from ...tests.utils import (skip_missing_dependency, TEST_DATA_DIR)
from ...tests import mocks
from ...tests.mocks import mock
from .. import datafind as io_datafind

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_GWF_FILE = os.path.join(TEST_DATA_DIR, 'HLV-HW100916-968654552-1.gwf')


# -- utilities ----------------------------------------------------------------

def mock_connection(framefile):
    import gwdatafind
    # create mock up of connection object
    conn = mock.create_autospec(gwdatafind.http.HTTPConnection)
    conn.find_types.return_value = [os.path.basename(framefile).split('-')[1]]
    conn.find_latest.return_value = [framefile]
    conn.find_urls.return_value = [framefile]
    conn.host = 'mockhost'
    conn.port = 80
    return conn


@pytest.fixture(scope='class')
def connection():
    with mock.patch('gwdatafind.ui.HTTPConnection',
                    return_value=mock_connection(TEST_GWF_FILE)) as mconn:
        yield mconn


# -- tests --------------------------------------------------------------------

def test_reconnect():
    a = HTTPConnection('127.0.0.1')
    b = io_datafind.reconnect(a)
    assert b is not a
    assert b.host == a.host
    assert b.port == a.port


@skip_missing_dependency('gwdatafind')
@mock.patch('gwpy.io.datafind.iter_channel_names',
            return_value=['L1:LDAS-STRAIN', 'H1:LDAS-STRAIN'])
@mock.patch('gwpy.io.datafind.num_channels', return_value=1)
@mock.patch('gwpy.io.datafind.reconnect')
def test_find_frametype(reconnect, num_channels, iter_channels, connection):
    reconnect.return_value = connection.return_value

    # simple test
    assert io_datafind.find_frametype('L1:LDAS-STRAIN',
                                      allow_tape=True) == 'HW100916'
    assert io_datafind.find_frametype('L1:LDAS-STRAIN',
                                      return_all=True) == ['HW100916']

    # test multiple channels
    assert io_datafind.find_frametype(['H1:LDAS-STRAIN'], allow_tape=True) == (
        {'H1:LDAS-STRAIN': 'HW100916'})

    # test missing channel raises sensible error
    with pytest.raises(ValueError) as exc:
        io_datafind.find_frametype('X1:TEST', allow_tape=True)
    assert str(exc.value) == (
        'Cannot locate the following channel(s) '
        'in any known frametype:\n    X1:TEST')

    # test malformed channel name raises sensible error
    with pytest.raises(ValueError) as exc:
        io_datafind.find_frametype('bad channel name')
    assert str(exc.value) == ('Cannot parse interferometer prefix '
                              'from channel name \'bad channel name\','
                              ' cannot proceed with find()')

    # test trend sorting ends up with an error
    with pytest.raises(ValueError) as exc:
        io_datafind.find_frametype('X1:TEST.rms,s-trend',
                                   allow_tape=True)
    with pytest.raises(ValueError):
        io_datafind.find_frametype('X1:TEST.rms,m-trend',
                                   allow_tape=True)

    # check that allow_tape errors get handled properly
    with mock.patch('gwpy.io.datafind.on_tape', return_value=True):
        with pytest.raises(ValueError) as exc:
            io_datafind.find_frametype('X1:TEST', allow_tape=False)
        assert '[files on tape have not been checked' in str(exc.value)


@skip_missing_dependency('gwdatafind')
@mock.patch('gwpy.io.datafind.iter_channel_names',
            return_value=['L1:LDAS-STRAIN'])
@mock.patch('gwpy.io.datafind.num_channels', return_value=1)
@mock.patch('gwpy.io.datafind.reconnect', side_effect=lambda x: x)
def test_find_best_frametype(reconnect, num_channels, iter_channels,
                             connection):
    assert io_datafind.find_best_frametype(
        'L1:LDAS-STRAIN', 968654552, 968654553) == 'HW100916'


def test_on_tape():
    assert io_datafind.on_tape(TEST_GWF_FILE) is False
