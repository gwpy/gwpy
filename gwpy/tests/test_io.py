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

"""Unit test for `io` module
"""

from __future__ import print_function

import os
import tempfile

import pytest

from gwpy.io import (cache as io_cache,
                     datafind as io_datafind,
                     gwf as io_gwf,
                     nds2 as io_nds2)
from gwpy.segments import (Segment, SegmentList)

import utils
import mocks
from mocks import mock

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TEST_GWF_FILE = os.path.join(os.path.dirname(__file__), 'data',
                             'HLV-GW100916-968654552-1.gwf')
TEST_CHANNELS = [
    'H1:LDAS-STRAIN', 'L1:LDAS-STRAIN', 'V1:h_16384Hz',
]


# -- gwpy.io.nds --------------------------------------------------------------

class TestIoNds2(object):
    """Tests of :mod:`gwpy.io.nds2`
    """
    def test_channel_type_find(self):
        """Test `gwpy.io.nds2.Nds2ChannelType` enum
        """
        # check that m-trend gets recognised properly
        a = io_nds2.Nds2ChannelType.find('m-trend')
        b = io_nds2.Nds2ChannelType.find('MTREND')
        assert a == b == io_nds2.Nds2ChannelType.MTREND
        # test unknown
        with pytest.raises(ValueError):
            io_nds2.Nds2ChannelType.find('blah')

    def test_data_type_find(self):
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

    def test_parse_nds_env(self):
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

    def test_nds2_host_order(self):
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

    @utils.skip_missing_dependency('nds2')
    def test_connect(self):
        """Test :func:`gwpy.io.connect`
        """
        import nds2
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

    def test_minute_trend_times(self):
        """Test :func:`gwpy.io.nds2.minute_trend_times`
        """
        assert io_nds2.minute_trend_times(0, 60) == (0, 60)
        assert io_nds2.minute_trend_times(1, 60) == (0, 60)
        assert io_nds2.minute_trend_times(0, 61) == (0, 120)
        assert io_nds2.minute_trend_times(59, 61) == (0, 120)
        assert (io_nds2.minute_trend_times(1167264018, 1198800018) ==
                (1167264000, 1198800060))


# -- gwpy.io.cache ------------------------------------------------------------

class TestIoCache(object):
    """Tests of :mod:`gwpy.io.cache`
    """
    @staticmethod
    def make_cache():
        try:
            from lal.utils import CacheEntry
        except ImportError as e:
            pytest.skip(str(e))
        from glue.lal import Cache

        segs = SegmentList()
        cache = Cache()
        for seg in [(0, 1), (1, 2), (4, 5)]:
            d = seg[1] - seg[0]
            f = 'A-B-%d-%d.tmp' % (seg[0], d)
            cache.append(CacheEntry.from_T050017(f, coltype=int))
            segs.append(Segment(*seg))
        return cache, segs

    @staticmethod
    def write_cache(cache, f):
        for entry in cache:
            print(str(entry), file=f)

    def test_open_cache(self):
        cache = self.make_cache()[0]
        with tempfile.NamedTemporaryFile() as f:
            self.write_cache(cache, f)
            f.seek(0)

            # read from fileobj
            c2 = io_cache.open_cache(f)
            assert cache == c2

            # read from file name
            c3 = io_cache.open_cache(f.name)
            assert cache == c3

    def test_file_list(self):
        cache = self.make_cache()[0]

        # test file -> [file.name]
        with tempfile.NamedTemporaryFile() as f:
            assert io_cache.file_list(f) == [f.name]

        # test CacheEntry -> [CacheEntry.path]
        assert io_cache.file_list(cache[0]) == [cache[0].path]

        # test cache file -> pfnlist()
        with tempfile.NamedTemporaryFile(suffix='.lcf') as f:
            self.write_cache(cache, f)
            f.seek(0)
            assert io_cache.file_list(f.name) == cache.pfnlist()

        # test comma-separated list -> list
        assert io_cache.file_list('A,B,C,D') == ['A', 'B', 'C', 'D']

        # test cache object -> pfnlist
        assert io_cache.file_list(cache) == cache.pfnlist()

        # test list -> list
        assert io_cache.file_list(['A', 'B', 'C', 'D']) == ['A', 'B', 'C', 'D']

        # otherwise error
        with pytest.raises(ValueError):
            io_cache.file_list(1)

    def test_cache_segments(self):
        """Test :func:`gwpy.io.cache.cache_segments`
        """
        # check empty input
        sl = io_cache.cache_segments()
        assert isinstance(sl, SegmentList)
        assert len(sl) == 0
        # check simple cache
        cache, segs = self.make_cache()
        segs.coalesce()
        sl = io_cache.cache_segments(cache)
        assert sl == segs
        # check multiple caches produces the same result
        sl = io_cache.cache_segments(cache[:2], cache[2:])
        assert sl == segs

    def test_file_segment(self):
        """Test :func:`gwpy.io.cache.file_segment`
        """
        # check basic
        fs = io_cache.file_segment('A-B-1-2.ext')
        assert isinstance(fs, Segment)
        assert fs == Segment(1, 3)
        # check mutliple file extensions
        assert io_cache.file_segment('A-B-1-2.ext.gz') == (1, 3)
        # check floats (and multiple file extensions)
        assert io_cache.file_segment('A-B-1.23-4.ext.gz') == (1.23, 5.23)
        # test errors
        with pytest.raises(ValueError) as exc:
            io_cache.file_segment('blah')
        assert str(exc.value) == ('Failed to parse \'blah\' as '
                                  'LIGO-T050017-compatible filename')

    def test_flatten(self):
        """Test :func:`gwpy.io.cache.flatten`
        """
        # check flattened version of single cache is unchanged
        a, _ = self.make_cache()
        assert io_cache.flatten(a) == a
        assert io_cache.flatten(a, a) == a
        # check two caches get concatenated properly
        b, _ = self.make_cache()
        for e in b:
            e.segment = e.segment.shift(10)
        c = a + b
        assert io_cache.flatten(a, b) == c

    def test_find_contiguous(self):
        """Test :func:`gwpy.io.cache.find_contiguous`
        """
        a, segs = self.make_cache()
        segs.coalesce()
        for i, cache in enumerate(io_cache.find_contiguous(a)):
            assert cache.to_segmentlistdict()['A'].extent() == segs[i]


# -- gwpy.io.gwf --------------------------------------------------------------

def mock_call(*args, **kwargs):
    raise OSError("")


class TestIoGwf(object):

    def test_identify_gwf(self):
        assert io_gwf.identify_gwf('read', TEST_GWF_FILE, None) is True
        with open(TEST_GWF_FILE, 'rb') as gwff:
            assert io_gwf.identify_gwf('read', None, gwff) is True
        assert not io_gwf.identify_gwf('read', None, None)

    @utils.skip_missing_dependency('lalframe')
    def test_iter_channel_names(self):
        # maybe need something better?
        from types import GeneratorType
        names = io_gwf.iter_channel_names(TEST_GWF_FILE)
        assert isinstance(names, GeneratorType)
        assert list(names) == TEST_CHANNELS
        with mock.patch('gwpy.utils.shell.call', mock_call):
            names = io_gwf.iter_channel_names(TEST_GWF_FILE)
            assert isinstance(names, GeneratorType)
            assert list(names) == TEST_CHANNELS

    @utils.skip_missing_dependency('lalframe')
    def test_get_channel_names(self):
        assert io_gwf.get_channel_names(TEST_GWF_FILE) == TEST_CHANNELS

    @utils.skip_missing_dependency('lalframe')
    def test_num_channels(self):
        assert io_gwf.num_channels(TEST_GWF_FILE) == 3

    @utils.skip_missing_dependency('lalframe')
    def test_get_channel_type(self):
        assert io_gwf.get_channel_type('L1:LDAS-STRAIN',
                                       TEST_GWF_FILE) == 'proc'
        with pytest.raises(ValueError) as exc:
            io_gwf.get_channel_type('X1:NOT-IN_FRAME', TEST_GWF_FILE)
        assert str(exc.value) == ('X1:NOT-IN_FRAME not found in '
                                  'table-of-contents for %s' % TEST_GWF_FILE)

    @utils.skip_missing_dependency('lalframe')
    def test_channel_in_frame(self):
        assert io_gwf.channel_in_frame('L1:LDAS-STRAIN', TEST_GWF_FILE) is True
        assert io_gwf.channel_in_frame('X1:NOT-IN_FRAME',
                                       TEST_GWF_FILE) is False


# -- gwpy.io.datafind ---------------------------------------------------------

class TestIoDatafind(object):
    """Tests for :mod:`gwpy.io.datafind`
    """
    @staticmethod
    @pytest.fixture(scope='class')
    @utils.skip_missing_dependency('lal')
    def connection():
        return mocks.mock_datafind_connection(TEST_GWF_FILE)

    def test_on_tape(self):
        """Test :func:`gwpy.io.datafind.on_tape`
        """
        assert io_datafind.on_tape(TEST_GWF_FILE) is False

    def test_connect(self, connection):
        """Test :func:`gwpy.io.datafind.connect`
        """
        with mock.patch('glue.datafind.GWDataFindHTTPConnection',
                        connection), \
                mock.patch('glue.datafind.GWDataFindHTTPSConnection',
                           connection), \
                mock.patch('glue.datafind.find_credential',
                           mocks.mock_find_credential):
            io_datafind.connect()  # HTTP
            io_datafind.connect('host', 443)  # HTTPS

    def test_find_frametype(self, connection):
        """Test :func:`gwpy.io.datafind.find_frametype
        """
        with mock.patch('glue.datafind.GWDataFindHTTPConnection') as \
                mock_connection, \
                mock.patch('gwpy.io.datafind.num_channels', lambda x: 1), \
                mock.patch('gwpy.io.gwf.iter_channel_names',
                           lambda x: ['L1:LDAS-STRAIN']):
            mock_connection.return_value = connection
            assert io_datafind.find_frametype('L1:LDAS-STRAIN',
                                              allow_tape=True) == 'GW100916'
            assert io_datafind.find_frametype('L1:LDAS-STRAIN',
                                              return_all=True) == ['GW100916']
            # test missing channel raises sensible error
            with pytest.raises(ValueError) as exc:
                io_datafind.find_frametype('X1:TEST', allow_tape=True)
            assert str(exc.value) == ('Cannot locate \'X1:TEST\' in any known '
                                      'frametype')
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
            assert str(exc.value) == ('Cannot locate \'X1:TEST.rms\' '
                                      'in any known frametype')
            with pytest.raises(ValueError):
                io_datafind.find_frametype('X1:TEST.rms,m-trend',
                                           allow_tape=True)
            assert str(exc.value) == ('Cannot locate \'X1:TEST.rms\' '
                                      'in any known frametype')

    def test_find_best_frametype(self, connection):
        """Test :func:`gwpy.io.datafind.find_best_frametype
        """
        with mock.patch('glue.datafind.GWDataFindHTTPConnection') as \
                mock_connection, \
                mock.patch('gwpy.io.datafind.num_channels', lambda x: 1), \
                mock.patch('gwpy.io.gwf.iter_channel_names',
                           lambda x: ['L1:LDAS-STRAIN']):
            mock_connection.return_value = connection
            assert io_datafind.find_best_frametype(
                'L1:LDAS-STRAIN', 968654552, 968654553) == 'GW100916'
