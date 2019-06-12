# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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
from io import BytesIO
from itertools import cycle

import six
from six.moves.http_client import (HTTPConnection, HTTPException)

import pytest

import gwdatafind

from ...testing.compat import mock
from ...testing.utils import (TEST_GWF_FILE, TemporaryFilename)
from .. import datafind as io_datafind

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

if six.PY2:
    OPEN = '__builtin__.open'
else:
    OPEN = 'builtins.open'

# -- mock the environment -----------------------------------------------------

MOCK_ENV = None


def setup_module():
    global MOCK_ENV
    MOCK_ENV = mock.patch.dict('os.environ', {
        'VIRGODATA': 'tmp',
        'LIGO_DATAFIND_SERVER': 'test:80',
    })
    MOCK_ENV.start()


def teardown_module():
    global MOCK_ENV
    if MOCK_ENV is not None:
        MOCK_ENV.stop()


# -- utilities ----------------------------------------------------------------

def mock_connection(framefile):
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


# -- FFL tests ----------------------------------------------------------------

FFL_WALK = [
    (os.curdir, [], ['test.ffl', 'test2.ffl']),
]


@mock.patch('os.walk', return_value=FFL_WALK)
class TestFflConnection(object):
    TEST_CLASS = io_datafind.FflConnection

    @mock.patch('gwpy.io.datafind.FflConnection._read_last_line',
                return_value='X-TEST-0-1.gwf 0 1 0 0')
    def test_init(self, mwalk, mreadlast):
        conn = self.TEST_CLASS()
        assert conn.paths == {
            ('X', 'test'): os.path.join(os.curdir, 'test.ffl'),
            ('X', 'test2'): os.path.join(os.curdir, 'test2.ffl'),
        }

    def test_get_ffl_dir(self, _):
        with mock.patch.dict(os.environ, {'FFLPATH': 'somepath'}):
            assert self.TEST_CLASS._get_ffl_dir() == 'somepath'
        with mock.patch.dict(os.environ, {'VIRGODATA': 'somepath'}):
            assert self.TEST_CLASS._get_ffl_dir() == (
                os.path.join('somepath', 'ffl'))
        with mock.patch.dict(os.environ), pytest.raises(KeyError):
            os.environ.pop('FFLPATH')
            os.environ.pop('VIRGODATA')
            self.TEST_CLASS._get_ffl_dir()

    def test_is_ffl_file(self, _):
        assert self.TEST_CLASS._is_ffl_file('test.ffl')
        assert not self.TEST_CLASS._is_ffl_file('test.ffl2')

    @mock.patch('gwpy.io.datafind.FflConnection._read_last_line',
                side_effect=[OSError(), 'X-TEST-0-1.gwf 0 1 0 0'])
    def test_find_paths(self, mwalk, mreadlast):
        conn = self.TEST_CLASS()  # find_paths() called by __init__()
        assert conn.paths == {
            ('X', 'test2'): os.path.join(os.curdir, 'test2.ffl'),
        }

    @mock.patch(OPEN, return_value=BytesIO(b"""
/path/to/X-TEST-0-1.gwf 0 1 0 0
/path/to/X-TEST-1-1.gwf 1 1 0 0
""".lstrip()))
    @mock.patch('os.path.getmtime', return_value=1)
    @mock.patch('gwpy.io.datafind.FflConnection._read_last_line',
                return_value='X-TEST-0-1.gwf 0 1 0 0')
    def test_read_ffl_cache(self, mwalk, mreadlast, mgetmtime, mopen):
        conn = self.TEST_CLASS()
        cache = list(conn._read_ffl_cache('X', 'test'))
        assert [c.path for c in cache] == [
            '/path/to/X-TEST-0-1.gwf',
            '/path/to/X-TEST-1-1.gwf'
        ]
        assert mopen.call_count == 1

        # check that calling the same again is a no-op
        conn._read_ffl_cache('X', 'test')
        assert mopen.call_count == 1

    def test_read_last_line(self, _):
        with TemporaryFilename() as tmp:
            with open(tmp, 'w') as fobj:
                print('line1', file=fobj)
                print('line2', file=fobj)
            assert self.TEST_CLASS._read_last_line(tmp) == 'line2'

    @mock.patch('gwpy.io.datafind.FflConnection._read_last_line',
                return_value='X-TEST-0-1.gwf 0 1 0 0')
    def test_ffl_path(self, mwalk, mreadlast):
        conn = self.TEST_CLASS()
        assert conn.ffl_path('X', 'test') == os.path.join(
            os.curdir, 'test.ffl')
        conn.paths = {}
        assert conn.ffl_path('X', 'test') == os.path.join(
            os.curdir, 'test.ffl')

    @mock.patch('gwpy.io.datafind.FflConnection._get_site_tag',
                side_effect=cycle([('X', 'test'), ('Y', 'test2')]))
    def test_find_types(self, mwalk, msitetag):

        conn = self.TEST_CLASS()
        assert sorted(conn.find_types(match=None)) == ['test', 'test2']
        assert conn.find_types('X') == ['test']
        assert conn.find_types(match='test2') == ['test2']

    @mock.patch(OPEN, return_value=BytesIO(b"""
/path/to/X-TEST-0-1.gwf 0 1 0 0
/path/to/X-TEST-1-1.gwf 1 1 0 0
/path/to/X-TEST-2-1.gwf 2 1 0 0
""".lstrip()))
    @mock.patch('os.path.getmtime', return_value=1)
    @mock.patch('gwpy.io.datafind.FflConnection._get_site_tag',
                side_effect=cycle([('X', 'test'), ('Y', 'test2')]))
    def test_find_urls(self, mwalk, msitetag, getmtime, mopen):
        conn = self.TEST_CLASS()
        assert conn.find_urls('X', 'test', 0, 2) == [
            '/path/to/X-TEST-0-1.gwf',
            '/path/to/X-TEST-1-1.gwf',
        ]
        assert conn.find_urls('X', 'test', 0, 2, match='TEST-0') == [
            '/path/to/X-TEST-0-1.gwf',
        ]

        # check exceptions or warnings get raised as designed
        with pytest.raises(RuntimeError):
            conn.find_urls('X', 'test', 10, 20, on_gaps='raise')
        with pytest.warns(UserWarning) as rec:
            conn.find_urls('X', 'test', 10, 20, on_gaps='warn')
            conn.find_urls('X', 'test', 10, 20, on_gaps='ignore')
        assert len(rec) == 1

    @mock.patch('gwpy.io.datafind.FflConnection._read_last_line',
                return_value='/path/to/file.gwf 0 1 0 0')
    @mock.patch('gwpy.io.datafind.FflConnection._get_site_tag',
                side_effect=cycle([('X', 'test'), ('Y', 'test2')]))
    def test_find_latest(self, mwalk, msitetag, mreadlast):
        conn = self.TEST_CLASS()
        assert conn.find_latest('X', 'test') == ['/path/to/file.gwf']
        assert mreadlast.call_count == 1
        assert conn.find_latest('X', 'test') == ['/path/to/file.gwf']
        assert mreadlast.call_count == 1  # doesn't call again

        # check exceptions or warnings get raised as designed
        with pytest.raises(RuntimeError):
            conn.find_latest('Z', 'test3', on_missing='raise')
        with pytest.warns(UserWarning) as rec:
            conn.find_latest('Z', 'test3', on_missing='warn')
            conn.find_latest('Z', 'test3', on_missing='ignore')
        assert len(rec) == 1


# -- tests --------------------------------------------------------------------

@mock.patch("gwdatafind.ui.connect", return_value="connection")
def test_with_connection(connect):
    func = mock.MagicMock()
    # https://stackoverflow.com/questions/22204660/python-mock-wrapsf-problems
    func.__name__ = "func"
    wrapped_func = io_datafind.with_connection(func)

    wrapped_func(1, host="host")
    assert func.called_with(1, connection="connection")
    assert connect.called_with(host="host")


@mock.patch("gwdatafind.ui.connect", return_value="connection")
@mock.patch("gwpy.io.datafind.reconnect", lambda x: x)
def test_with_connection_reconnect(connect):
    func = mock.MagicMock()
    # https://stackoverflow.com/questions/22204660/python-mock-wrapsf-problems
    func.__name__ = "func"
    func.side_effect = [HTTPException, "return"]
    wrapped_func = io_datafind.with_connection(func)

    assert wrapped_func(1, host="host") == "return"
    assert func.call_count == 2


def test_reconnect():
    a = HTTPConnection('127.0.0.1')
    b = io_datafind.reconnect(a)
    assert b is not a
    assert b.host == a.host
    assert b.port == a.port

    with mock.patch('os.walk', return_value=[]):
        a = io_datafind.FflConnection()
        b = io_datafind.reconnect(a)
        assert b is not a
        assert b.ffldir == a.ffldir


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


@mock.patch('gwpy.io.datafind.iter_channel_names',
            return_value=['L1:LDAS-STRAIN'])
@mock.patch('gwpy.io.datafind.num_channels', return_value=1)
@mock.patch('gwpy.io.datafind.reconnect', side_effect=lambda x: x)
def test_find_best_frametype(reconnect, num_channels, iter_channels,
                             connection):
    assert io_datafind.find_best_frametype(
        'L1:LDAS-STRAIN', 968654552, 968654553) == 'HW100916'


def test_find_types(connection):
    types = ["a", "b", "c"]
    connection.find_types.return_value = types
    assert io_datafind.find_types(
        "X",
        connection=connection,
    ) == types


def test_find_types_priority(connection):
    types = ["L1_R", "L1_T", "L1_M"]
    connection.find_types.return_value = types
    assert io_datafind.find_types(
        "X",
        trend="m-trend",
        connection=connection,
    ) == ["L1_M", "L1_R", "L1_T"]


def test_on_tape():
    assert io_datafind.on_tape(TEST_GWF_FILE) is False


@pytest.mark.parametrize('ifo, ftype, trend, priority', [
    ('L1', 'L1_HOFT_C00', None, 1),  # hoft
    ('H1', 'H1_HOFT_C02_T1700406_v3', None, 1),  # cleaned hoft
    ('H1', 'H1_M', 'm-trend', 0),  # minute trends
    ('K1', 'K1_T', 's-trend', 0),  # second trends
    ('K1', 'K1_R', 's-trend', 5),  # raw type when looking for second trend
    ('K1', 'K1_M', None, 10),  # trend type, but not looking for trend channel
    ('K1', 'K1_C', None, 6),  # commissioning type
    ('X1', 'SOMETHING_GRB051103', None, 10),  # low priority type
    ('X1', 'something else', None, 5),  # other
])
def test_type_priority(ifo, ftype, trend, priority):
    assert io_datafind._type_priority(ifo, ftype, trend=trend)[0] == priority
