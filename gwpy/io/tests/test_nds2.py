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

"""Unit tests for :mod:`gwpy.io.nds2`
"""

import os

import pytest

from ...tests import mocks
from ...tests.mocks import mock
from ...tests.utils import skip_missing_dependency
from .. import nds2 as io_nds2

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_channel_type_find():
    """Test `gwpy.io.nds2.Nds2ChannelType` enum
    """
    # check that m-trend gets recognised properly
    a = io_nds2.Nds2ChannelType.find('m-trend')
    b = io_nds2.Nds2ChannelType.find('MTREND')
    assert a == b == io_nds2.Nds2ChannelType.MTREND
    # test unknown
    with pytest.raises(ValueError):
        io_nds2.Nds2ChannelType.find('blah')


def test_data_type_find():
    """Test `gwpy.io.nds2.Nds2DataType` enum
    """
    # test float
    assert io_nds2.Nds2DataType.find(float) == io_nds2.Nds2DataType.FLOAT64
    # test uint
    assert io_nds2.Nds2DataType.find('uint32') == \
        io_nds2.Nds2DataType.UINT32
    # test unknown
    with pytest.raises(TypeError):
        io_nds2.Nds2DataType.find('blah')


def test_parse_nds_env():
    """Test :func:`gwpy.io.nds2.parse_nds_env`
    """
    # test basic NDSSERVER usage
    os.environ['NDSSERVER'] = 'test1.ligo.org:80,test2.ligo.org:43'
    hosts = io_nds2.parse_nds_env()
    assert hosts == [('test1.ligo.org', 80), ('test2.ligo.org', 43)]

    # check that duplicates get removed
    os.environ['NDSSERVER'] = ('test1.ligo.org:80,test2.ligo.org:43,'
                               'test.ligo.org,test2.ligo.org:43')
    hosts = io_nds2.parse_nds_env()
    assert hosts == [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                     ('test.ligo.org', None)]

    # check that a named environment variable works
    os.environ['TESTENV'] = 'test1.ligo.org:80,test2.ligo.org:43'
    hosts = io_nds2.parse_nds_env('TESTENV')
    assert hosts == [('test1.ligo.org', 80), ('test2.ligo.org', 43)]


def test_nds2_host_order():
    """Test :func:`gwpy.io.nds2.host_resolution_order`
    """
    # check None returns CIT
    hro = io_nds2.host_resolution_order(None, env=None)
    assert hro == [('nds.ligo.caltech.edu', 31200)]

    # check L1 returns (LLO, CIT)
    hro = io_nds2.host_resolution_order('L1', env=None)
    assert hro == [('nds.ligo-la.caltech.edu', 31200),
                   ('nds.ligo.caltech.edu', 31200)]

    # check NDSSERVER works
    os.environ['NDSSERVER'] = 'test1.ligo.org:80,test2.ligo.org:43'
    hro = io_nds2.host_resolution_order(None)
    assert hro == [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                   ('nds.ligo.caltech.edu', 31200)]

    # check that NDSSERVER and an IFO spec works at the same time
    hro = io_nds2.host_resolution_order('L1')
    assert hro == [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                   ('nds.ligo-la.caltech.edu', 31200),
                   ('nds.ligo.caltech.edu', 31200)]

    # test named environment variable
    os.environ['TESTENV'] = 'test1.ligo.org:80,test2.ligo.org:43'
    hro = io_nds2.host_resolution_order(None, env='TESTENV')
    assert hro == [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                   ('nds.ligo.caltech.edu', 31200)]

    # test epoch='now' doesn't change anything
    os.environ.pop('NDSSERVER')
    hro = io_nds2.host_resolution_order('L1', epoch='now', env=None)
    assert hro == [('nds.ligo-la.caltech.edu', 31200),
                   ('nds.ligo.caltech.edu', 31200)]

    # test old epoch puts CIT ahead of LLO
    hro = io_nds2.host_resolution_order('L1', epoch='Jan 1 2015', env=None)
    assert hro == [('nds.ligo.caltech.edu', 31200),
                   ('nds.ligo-la.caltech.edu', 31200)]

    # test epoch doesn't operate with env
    hro = io_nds2.host_resolution_order('L1', epoch='now', env='TESTENV')
    assert hro == [('test1.ligo.org', 80), ('test2.ligo.org', 43),
                   ('nds.ligo-la.caltech.edu', 31200),
                   ('nds.ligo.caltech.edu', 31200)]

    # test warnings for unknown IFO
    with pytest.warns(UserWarning):
        hro = io_nds2.host_resolution_order('X1')
        assert hro == [('nds.ligo.caltech.edu', 31200)]


@skip_missing_dependency('nds2')
def test_connect():
    """Test :func:`gwpy.io.connect`
    """
    nds_connection = mocks.nds2_connection(host='nds.test.gwpy')
    with mock.patch('nds2.connection') as mock_connection:
        mock_connection.return_value = nds_connection
        conn = io_nds2.connect('nds.test.gwpy')
        assert conn.get_host() == 'nds.test.gwpy'
        assert conn.get_port() == 31200

    nds_connection = mocks.nds2_connection(host='nds2.test.gwpy',
                                           port=8088)
    with mock.patch('nds2.connection') as mock_connection:
        mock_connection.return_value = nds_connection
        conn = io_nds2.connect('nds2.test.gwpy')
        assert conn.get_host() == 'nds2.test.gwpy'
        assert conn.get_port() == 8088


def test_minute_trend_times():
    """Test :func:`gwpy.io.nds2.minute_trend_times`
    """
    assert io_nds2.minute_trend_times(0, 60) == (0, 60)
    assert io_nds2.minute_trend_times(1, 60) == (0, 60)
    assert io_nds2.minute_trend_times(0, 61) == (0, 120)
    assert io_nds2.minute_trend_times(59, 61) == (0, 120)
    assert (io_nds2.minute_trend_times(1167264018, 1198800018) ==
            (1167264000, 1198800060))
