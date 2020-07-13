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

"""Unit tests for :mod:`gwpy.io.datafind`
"""

import os
from http.client import (HTTPConnection, HTTPException)
from io import BytesIO
from itertools import cycle
from unittest import mock

import pytest

import gwdatafind

from ...testing.utils import (
    TEST_GWF_FILE,
    TemporaryFilename,
    skip_missing_dependency,
)
from .. import datafind as io_datafind

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# -- mock the environment -----------------------------------------------------

MOCK_ENV = {
    'VIRGODATA': 'tmp',
    'LIGO_DATAFIND_SERVER': 'test:80',
}

_mock_env = mock.patch.dict("os.environ", MOCK_ENV)


# -- utilities ----------------------------------------------------------------

def mock_connection(framefile=TEST_GWF_FILE):
    # create mock up of connection object
    conn = mock.create_autospec(gwdatafind.http.HTTPConnection)
    conn.find_types.return_value = [os.path.basename(framefile).split('-')[1]]
    conn.find_latest.return_value = [framefile]
    conn.find_urls.return_value = [framefile]
    conn.host = 'mockhost'
    conn.port = 80
    return conn


@pytest.fixture()
def connection():
    return mock_connection(TEST_GWF_FILE)


_mock_connection = mock.patch(
    "gwdatafind.connect",
    mock.MagicMock(return_value=mock_connection(TEST_GWF_FILE)),
)

_mock_iter_channel_names = mock.patch(
    'gwpy.io.datafind.iter_channel_names',
    mock.MagicMock(return_value=['L1:LDAS-STRAIN', 'H1:LDAS-STRAIN']),
)

_mock_num_channels = mock.patch(
    'gwpy.io.datafind.num_channels',
    mock.MagicMock(return_value=1),
)

# -- FFL tests ----------------------------------------------------------------

FFL_WALK = [
    (os.curdir, [], ['test.ffl', 'test2.ffl']),
]


@mock.patch.dict("os.environ", MOCK_ENV)
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

    @mock.patch("builtins.open", return_value=BytesIO(b"""
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
        mopen.assert_called_once()

        # check that calling the same again is a no-op
        conn._read_ffl_cache('X', 'test')
        mopen.assert_called_once()

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

    @mock.patch("builtins.open", return_value=BytesIO(b"""
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
        mreadlast.assert_called_once()
        assert conn.find_latest('X', 'test') == ['/path/to/file.gwf']
        mreadlast.assert_called_once()  # doesn't call again

        # check exceptions or warnings get raised as designed
        with pytest.raises(RuntimeError):
            conn.find_latest('Z', 'test3', on_missing='raise')
        with pytest.warns(UserWarning) as rec:
            conn.find_latest('Z', 'test3', on_missing='warn')
            conn.find_latest('Z', 'test3', on_missing='ignore')
        assert len(rec) == 1


# -- tests --------------------------------------------------------------------

@mock.patch.dict("os.environ", MOCK_ENV)
@mock.patch("gwdatafind.connect", return_value="connection")
def test_with_connection(connect):
    # mock out the function, and wrap it
    func = mock.MagicMock()
    wrapped_func = io_datafind.with_connection(func)

    # call it
    wrapped_func(1, host="host")

    # check that connect() was called once
    connect.assert_called_once_with(host="host", port=None)
    # and that the function was called once with that connection
    func.assert_called_once_with(1, connection="connection", host="host")


@mock.patch.dict("os.environ", MOCK_ENV)
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


@mock.patch.dict("os.environ", MOCK_ENV)
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


@_mock_connection
@_mock_env
@_mock_iter_channel_names
@_mock_num_channels
def test_find_frametype():
    # simple test
    assert io_datafind.find_frametype(
        'L1:LDAS-STRAIN',
        allow_tape=True,
    ) == 'HW100916'


@_mock_connection
@_mock_env
@_mock_iter_channel_names
@_mock_num_channels
def test_find_frametype_return_all():
    assert io_datafind.find_frametype(
        'L1:LDAS-STRAIN',
        return_all=True,
    ) == ['HW100916']


@_mock_connection
@_mock_env
@_mock_iter_channel_names
@_mock_num_channels
def test_find_frametype_multiple():
    # test multiple channels
    assert io_datafind.find_frametype(
        ['H1:LDAS-STRAIN'],
        allow_tape=True,
    ) == {'H1:LDAS-STRAIN': 'HW100916'}


@_mock_connection
@_mock_env
@_mock_iter_channel_names
@_mock_num_channels
def test_find_frametype_errors():
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


@_mock_connection
@_mock_env
@_mock_iter_channel_names
@_mock_num_channels
def test_find_best_frametype(connection):
    assert io_datafind.find_best_frametype(
        'L1:LDAS-STRAIN',
        968654552,
        968654553,
    ) == 'HW100916'


@skip_missing_dependency('LDAStools.frameCPP')
@pytest.mark.skipif(
    'LIGO_DATAFIND_SERVER' not in os.environ,
    reason='No LIGO datafind server configured on this host',
)
@pytest.mark.parametrize('channel, expected', [
    ('H1:GDS-CALIB_STRAIN', ['H1_HOFT_C00', 'H1_ER_C00_L1']),
    ('L1:IMC-ODC_CHANNEL_OUT_DQ', ['L1_R']),
    ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend', ['H1_T']),
    ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend', ['H1_M'])
])
def test_find_best_frametype_ligo(channel, expected):
    try:
        ft = io_datafind.find_best_frametype(
            channel, 1143504017, 1143504017+100)
    except ValueError as exc:  # pragma: no-cover
        if str(exc).lower().startswith('cannot locate'):
            pytest.skip(str(exc))
        raise
    except RuntimeError as exc:  # pragma: no-cover
        if "credential" in str(exc):
            pytest.skip(str(exc))
        raise
    assert ft in expected


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
