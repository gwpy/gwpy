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

"""Unit tests for :mod:`gwpy.io.nds2`
"""

import os
from unittest import mock

import pytest

from ...detector import Channel
from ...segments import (Segment, SegmentList)
from ...testing import mocks
from ...utils.tests.test_enum import TestNumpyTypeEnum as _TestNumpyTypeEnum
from .. import nds2 as io_nds2

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


class _TestNds2Enum(object):
    TEST_CLASS = None

    def test_any(self):
        assert self.TEST_CLASS.any() == 2 * max(self.TEST_CLASS).value - 1

    def test_find_errors(self):
        """Test error raising for :meth:`gwpy.io.nds2.Nds2ChannelType.find`
        """
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS.find('blah')
        assert str(exc.value) == (
            f"'blah' is not a valid {self.TEST_CLASS.__name__}"
        )


class TestNds2ChannelType(_TestNds2Enum):
    """Tests of :class:`gwpy.io.nds2.Nds2DataType`
    """
    TEST_CLASS = io_nds2.Nds2ChannelType

    def test_nds2name(self):
        assert self.TEST_CLASS.MTREND.nds2name == 'm-trend'

    def test_nds2names(self):
        expected = sorted(
            io_nds2._NDS2_CHANNEL_TYPE[x.name][1]
            for x in self.TEST_CLASS
        )
        assert sorted(self.TEST_CLASS.nds2names()) == expected

    @pytest.mark.parametrize(('input_', 'expected'), (
        (TEST_CLASS.MTREND.value, TEST_CLASS.MTREND),
        (TEST_CLASS.MTREND.name, TEST_CLASS.MTREND),
        (TEST_CLASS.MTREND.nds2name, TEST_CLASS.MTREND),
        ('mtrend', TEST_CLASS.MTREND),
        ('rds', TEST_CLASS.RDS),
        ('RDS', TEST_CLASS.RDS),
        ('reduced', TEST_CLASS.RDS),
        ('REDUCED', TEST_CLASS.RDS),
    ))
    def test_find(self, input_, expected):
        """Test :meth:`gwpy.io.nds2.Nds2ChannelType.find`
        """
        assert self.TEST_CLASS.find(input_) == expected


class TestNds2DataType(_TestNds2Enum, _TestNumpyTypeEnum):
    """Tests of :class:`gwpy.io.nds2.Nds2DataType`
    """
    TEST_CLASS = io_nds2.Nds2DataType


@pytest.mark.parametrize('key, value, hosts', [
    ('NDSSERVER',
     'test1.ligo.org:80,test2.ligo.org:43',
     [('test1.ligo.org', 80), ('test2.ligo.org', 43)]),
    ('NDSSERVER',
     'test1.ligo.org:80,test2.ligo.org:43,test.ligo.org,test2.ligo.org:43',
     [('test1.ligo.org', 80), ('test2.ligo.org', 43),
      ('test.ligo.org', None)]),
    ('TESTENV',
     'test1.ligo.org:80,test2.ligo.org:43',
     [('test1.ligo.org', 80), ('test2.ligo.org', 43)]),
])
def test_parse_nds_env(key, value, hosts):
    """Test `gwpy.io.nds2.parse_nds_env`
    """
    with mock.patch.dict(os.environ, {key: value}):
        assert io_nds2.parse_nds_env(env=key) == hosts


@pytest.mark.parametrize('ifo, hosts', [
    (None, [('nds.ligo.caltech.edu', 31200)]),
    ('L1', [('nds.ligo-la.caltech.edu', 31200),
            ('nds.ligo.caltech.edu', 31200)]),
])
def test_host_resolution_order(ifo, hosts):
    """Test `gwpy.io.nds2.host_resolution_order` basic usage
    """
    assert io_nds2.host_resolution_order(ifo, env=None) == hosts


@mock.patch.dict(os.environ,
                 {'NDSSERVER': 'test1.ligo.org:80,test2.ligo.org:43'})
@pytest.mark.parametrize('ifo, hosts', [
    (None, [('test1.ligo.org', 80), ('test2.ligo.org', 43),
            ('nds.ligo.caltech.edu', 31200)]),
    ('L1', [('test1.ligo.org', 80), ('test2.ligo.org', 43),
            ('nds.ligo-la.caltech.edu', 31200),
            ('nds.ligo.caltech.edu', 31200)]),
])
def test_host_resolution_order_env(ifo, hosts):
    """Test `gwpy.io.nds2.host_resolution_order` environment parsing
    """
    assert io_nds2.host_resolution_order(ifo) == hosts


@mock.patch.dict(os.environ,
                 {'TESTENV': 'test1.ligo.org:80,test2.ligo.org:43'})
def test_host_resolution_order_named_env():
    """Test `gwpy.io.nds2.host_resolution_order` environment parsing
    with a named environment variable
    """
    hro = io_nds2.host_resolution_order(None, env='TESTENV')
    assert hro == [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                   ('nds.ligo.caltech.edu', 31200)]


@mock.patch.dict(os.environ,
                 {'TESTENV': 'test1.ligo.org:80,test2.ligo.org:43'})
@pytest.mark.parametrize('ifo, epoch, env, hosts', [
    ('L1', 'Jan 1 2015', None,
     [('nds.ligo.caltech.edu', 31200), ('nds.ligo-la.caltech.edu', 31200)]),
    ('L1', 'now', 'TESTENV',
     [('test1.ligo.org', 80), ('test2.ligo.org', 43),
      ('nds.ligo-la.caltech.edu', 31200), ('nds.ligo.caltech.edu', 31200)]),
])
def test_host_resolution_order_epoch(ifo, epoch, env, hosts):
    """Test `gwpy.io.nds2.host_resolution_order` epoch parsing
    """
    assert io_nds2.host_resolution_order(ifo, epoch=epoch, env=env) == hosts


@mock.patch.dict(os.environ,
                 {'TESTENV': 'test1.ligo.org:80,test2.ligo.org:43'})
def test_host_resolution_order_warning():
    """Test `gwpy.io.nds2.host_resolution_order` warnings
    """
    # test warnings for unknown IFO
    with pytest.warns(UserWarning) as record:
        # should produce warning
        hro = io_nds2.host_resolution_order('X1', env=None)
        assert hro == [('nds.ligo.caltech.edu', 31200)]
        # should _not_ produce warning
        hro = io_nds2.host_resolution_order('X1', env='TESTENV')
    assert len(record) == 1  # make sure only one warning was emitted


@pytest.mark.requires("nds2")
@pytest.mark.parametrize('host, port, callport', [
    ('nds.test.gwpy', None, None),
    ('nds.test.gwpy', 31200, 31200),
    ('x1nds9', None, 8088),
])
@mock.patch('nds2.connection')
def test_connect(connector, host, port, callport):
    """Test `gwpy.io.nds2.connect`
    """
    io_nds2.connect(host, port=port)
    if callport is None:
        connector.assert_called_once_with(host)
    else:
        connector.assert_called_once_with(host, callport)


@pytest.mark.requires("nds2")
@mock.patch('gwpy.io.nds2.connect')
def test_auth_connect(connect):
    """Test `gwpy.io.nds2.auth_connect`
    """
    io_nds2.auth_connect('host', 'port')
    connect.assert_called_once_with('host', 'port')


@pytest.mark.requires("nds2")
@mock.patch('gwpy.io.nds2.kinit')
@mock.patch(
    'gwpy.io.nds2.connect',
    side_effect=(
        RuntimeError('Request SASL authentication something something'),
        True,
    ),
)
def test_auth_connect_kinit(connect, kinit):
    """Test `gwpy.io.nds2.auth_connect` with a callout to
    `gwpy.io.kerberos.kinit`
    """
    with pytest.warns(io_nds2.NDSWarning):
        assert io_nds2.auth_connect('host', 'port')
    kinit.assert_called_with()
    assert connect.call_count == 2
    connect.assert_called_with('host', 'port')


@pytest.mark.requires("nds2")
@mock.patch('gwpy.io.nds2.connect', side_effect=RuntimeError('Anything else'))
def test_auth_connect_error(connect):
    """Test errors from `gwpy.io.nds2.auth_connect`
    """
    with pytest.raises(RuntimeError) as exc:
        io_nds2.auth_connect('host', 'port')
    assert str(exc.value) == 'Anything else'
    connect.assert_called_once_with("host", "port")


@mock.patch('gwpy.io.nds2.auth_connect', return_value=1)
def test_open_connection(auth_connect):
    """Test the `gwpy.io.nds2.open_connection` decorator
    """
    @io_nds2.open_connection
    def new_func(arg1, connection=None):
        return arg1, connection

    with pytest.raises(TypeError):
        new_func(0)

    # call function and check that `connection` was injected
    assert new_func(0, host='test') == (0, 1)
    # check that auth_connect was called with the right arguments
    auth_connect.assert_called_once_with('test', None)


@pytest.mark.requires("nds2")
@mock.patch('nds2.connection')
def test_find_channels(connection):
    """Test `gwpy.io.nds2.find_channels`
    """
    # set up connection.find_channels
    chan = mocks.nds2_channel('X1:test', 16, 'm')
    conn = mock.MagicMock()
    connection.return_value = conn
    conn.find_channels.return_value = [chan]

    # call function and check result
    assert io_nds2.find_channels(['X1:test'], host='test.nds2') == [chan]

    # check callouts in find_channels
    connection.assert_called_once_with('test.nds2')
    conn.set_epoch.assert_any_call('ALL')

    # check callouts in _find_channels
    conn.find_channels.assert_called_once_with(
        'X1:test',
        io_nds2.Nds2ChannelType.any(),
        io_nds2.Nds2DataType.any(),
    )

    # test keyword handling
    conn.get_protocol.return_value = 1
    io_nds2.find_channels(['X1:test,m-trend'], host='test.nds2',
                          sample_rate=16, dtype=float)
    conn.find_channels.assert_called_with(
        'X1:test,m-trend',
        io_nds2.Nds2ChannelType.MTREND.value,  # m-trend
        io_nds2.Nds2DataType.FLOAT64.value,  # float
        16, 16,  # sample_rate
    )

    # test unique check
    onlinechan = mocks.nds2_channel('X1:test', 16, 'm')
    onlinechan.channel_type = 1
    conn.find_channels.return_value = [chan, onlinechan]
    assert io_nds2.find_channels(['X1:test'], host='test',
                                 unique=True) == [chan]

    # test unique errors (with nds1 protocol)
    conn.find_channels.return_value = [chan, chan]  # any two channels
    with pytest.raises(ValueError) as exc:
        assert io_nds2.find_channels(['X1:test'], host='test',
                                     unique=True) == [chan]
    assert str(exc.value) == (
        "unique NDS2 channel match not found for 'X1:test'")


@pytest.mark.requires("nds2")
@mock.patch('gwpy.io.nds2.find_channels', return_value=['X1:test'])
@mock.patch('gwpy.io.nds2.auth_connect')
def test_get_availability(auth_connect, _):
    """Test `gwpy.io.nds2.get_availability`
    """
    # setup mocks
    conn = mock.MagicMock()
    auth_connect.return_value = conn
    conn.get_availability.return_value = [
        mocks.nds2_availability('X1:test', [(0, 1), (2, 3)]),
    ]

    # validate call and parsing of results
    assert io_nds2.get_availability(['X1:test'], 0, 1, host='test') == {
        'X1:test': SegmentList([Segment(0, 1), Segment(2, 3)]),
    }

    # check callouts
    conn.set_epoch.assert_has_calls([
        mock.call(0, 1),
        mock.call(
            conn.current_epoch().gps_start,
            conn.current_epoch().gps_stop,
        ),
    ])
    conn.get_availability.assert_called_once_with(['X1:test'])


@pytest.mark.parametrize('start, end, out', [
    (0, 60, (0, 60)),  # no change
    (1, 60, (0, 60)),  # expand at start
    (0, 61, (0, 120)),  # expand at end
    (59, 61, (0, 120)),  # expand both
    (1167264018, 1198800018, (1167264000, 1198800060)),  # expand both
])
def test_minute_trend_times(start, end, out):
    """Test `gwpy.io.nds2.minute_trend_times`
    """
    assert io_nds2.minute_trend_times(start, end) == out


@pytest.mark.requires("nds2")
def test_get_nds2_name():
    """Test `gwpy.io.nds2._get_nds2_name`
    """
    # we can't use parametrize because mocks.nds2_channel requires
    # the nds2-client and is executed _before_ the skip decorator is
    # applied
    for channel, name in [
        ('test', 'test'),
        (Channel('X1:TEST', type='m-trend'), 'X1:TEST,m-trend'),
        (mocks.nds2_channel('X1:TEST', 16, 'NONE'), 'X1:TEST,raw'),
    ]:
        assert io_nds2._get_nds2_name(channel) == name


@pytest.mark.requires("nds2")
def test_get_nds2_names():
    """Test `gwpy.io.nds2._get_nds2_names`
    """
    channels = ['test', Channel('X1:TEST', type='m-trend'),
                mocks.nds2_channel('X1:TEST', 16, 'NONE')]
    names = ['test', 'X1:TEST,m-trend', 'X1:TEST,raw']
    assert list(io_nds2._get_nds2_names(channels)) == names
