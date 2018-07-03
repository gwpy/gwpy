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

import pytest

from ...tests.utils import (skip_missing_dependency, TEST_DATA_DIR)
from ...tests import mocks
from ...tests.mocks import mock
from .. import datafind as io_datafind

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_GWF_FILE = os.path.join(TEST_DATA_DIR, 'HLV-HW100916-968654552-1.gwf')


@pytest.fixture(scope='class')
@skip_missing_dependency('lal')
def connection():
    return mocks.mock_datafind_connection(TEST_GWF_FILE)


def test_on_tape():
    assert io_datafind.on_tape(TEST_GWF_FILE) is False


def test_connect(connection):
    with mock.patch('glue.datafind.GWDataFindHTTPConnection',
                    connection), \
            mock.patch('glue.datafind.GWDataFindHTTPSConnection',
                       connection), \
            mock.patch('glue.datafind.find_credential',
                       mocks.mock_find_credential):
        io_datafind.connect()  # HTTP
        io_datafind.connect('host', 443)  # HTTPS


def test_find_frametype(connection):
    with mock.patch('glue.datafind.GWDataFindHTTPConnection') as \
            mock_connection, \
            mock.patch('gwpy.io.datafind.num_channels', lambda x: 1), \
            mock.patch('gwpy.io.datafind.iter_channel_names',
                       lambda x: ['L1:LDAS-STRAIN']):
        mock_connection.return_value = connection
        assert io_datafind.find_frametype('L1:LDAS-STRAIN',
                                          allow_tape=True) == 'HW100916'
        assert io_datafind.find_frametype('L1:LDAS-STRAIN',
                                          return_all=True) == ['HW100916']

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


def test_find_best_frametype(connection):
    with mock.patch('glue.datafind.GWDataFindHTTPConnection') as \
            mock_connection, \
            mock.patch('gwpy.io.datafind.num_channels', lambda x: 1), \
            mock.patch('gwpy.io.datafind.iter_channel_names',
                       lambda x: ['L1:LDAS-STRAIN']):
        mock_connection.return_value = connection
        assert io_datafind.find_best_frametype(
            'L1:LDAS-STRAIN', 968654552, 968654553) == 'HW100916'
