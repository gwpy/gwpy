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

"""Unit test for detector module
"""

from unittest import mock

import pytest

import numpy

from astropy import units

from ...segments import SegmentListDict
from ...testing import (utils, mocks)
from .. import (Channel, ChannelList)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

OMEGA_CONFIG = """
[L1:CAL,L1 calibrated]

{
  channelName:                 'L1:GDS-CALIB_STRAIN'
  frameType:                   'L1_HOFT_C00'
  sampleFrequency:             4096
  searchTimeRange:             64
  searchFrequencyRange:        [0 Inf]
  searchQRange:                [4 96]
  searchMaximumEnergyLoss:     0.2
  whiteNoiseFalseRate:         1e-3
  searchWindowDuration:        0.5
  plotTimeRanges:              [1 4 16]
  plotFrequencyRange:          []
  plotNormalizedEnergyRange:   [0 25.5]
  alwaysPlotFlag:              1
}

[L1:PEM,L1 environment]

{
  channelName:                 'L1:PEM-CS_SEIS_LVEA_VERTEX_Z_DQ'
  frameType:                   'L1_R'
  sampleFrequency:             128
  searchTimeRange:             1024
  searchFrequencyRange:        [0 Inf]
  searchQRange:                [4 64]
  searchMaximumEnergyLoss:     0.2
  whiteNoiseFalseRate:         1e-3
  searchWindowDuration:        1.0
  plotTimeRanges:              [8 64 512]
  plotFrequencyRange:          []
  plotNormalizedEnergyRange:   [0 25.5]
  alwaysPlotFlag:              0
}
"""

CLF = """
[group-1]
flow = 4
fhigh = Nyquist
qhigh = 150
frametype = H1_HOFT_C00
channels = H1:GDS-CALIB_STRAIN 16384 unsafe clean

[group-2]
flow = .1
fhigh = 60
qhigh = 60
frametype = H1_R
channels =
    H1:ISI-GND_STS_HAM2_X_DQ 512 safe flat
    H1:ISI-GND_STS_HAM2_Y_DQ 256 unsafe flat
    H1:ISI-GND_STS_HAM2_Z_DQ 512 glitchy
"""


# -- Channel ------------------------------------------------------------------

class TestChannel(object):
    TEST_CLASS = Channel

    # -- test creation --------------------------

    def test_empty(self):
        new = self.TEST_CLASS('')
        assert str(new) == ''
        assert new.sample_rate is None
        assert new.dtype is None

    def test_new(self):
        new = self.TEST_CLASS('X1:GWPY-TEST_CHANNEL_NAME',
                              sample_rate=64, unit='m')
        assert str(new) == 'X1:GWPY-TEST_CHANNEL_NAME'
        assert new.ifo == 'X1'
        assert new.system == 'GWPY'
        assert new.subsystem == 'TEST'
        assert new.signal == 'CHANNEL_NAME'
        assert new.sample_rate == 64 * units.Hz
        assert new.unit is units.meter

        new2 = self.TEST_CLASS(new)
        assert new2.sample_rate == new.sample_rate
        assert new2.unit == new.unit
        assert new2.texname == new.texname

    # -- test properties ------------------------

    @pytest.mark.parametrize('arg, fs', [
        (None, None),
        (1, 1 * units.Hz),
        (1 * units.Hz, 1 * units.Hz),
        (1000 * units.mHz, 1 * units.Hz),
        ('1', 1 * units.Hz),
    ])
    def test_sample_rate(self, arg, fs):
        new = self.TEST_CLASS('test', sample_rate=arg)
        if arg is not None:
            assert isinstance(new.sample_rate, units.Quantity)
        assert new.sample_rate == fs

    @pytest.mark.parametrize('arg, unit', [
        (None, None),
        ('m', units.m),
    ])
    def test_unit(self, arg, unit):
        new = self.TEST_CLASS('test', unit=arg)
        if arg is not None:
            assert isinstance(new.unit, units.UnitBase)
        assert new.unit == unit

    def test_frequency_range(self):
        new = self.TEST_CLASS('test', frequency_range=(1, 40))
        assert isinstance(new.frequency_range, units.Quantity)
        utils.assert_quantity_equal(new.frequency_range, (1, 40) * units.Hz)

        with pytest.raises(TypeError):
            Channel('', frequency_range=1)

    def test_safe(self):
        new = self.TEST_CLASS('')
        assert new.safe is None
        new.safe = 1
        assert new.safe is True

    @pytest.mark.parametrize('arg, model', [
        (None, None),
        ('H1ASCIMC', 'h1ascimc'),
    ])
    def test_model(self, arg, model):
        new = self.TEST_CLASS('test', model=arg)
        assert new.model == model

    @pytest.mark.parametrize('arg, type_, ndstype', [
        (None, None, None),
        ('m-trend', 'm-trend', 16),
        (8, 's-trend', 8),
        ('blah', 'RAISE', '')
    ])
    def test_type_ndstype(self, arg, type_, ndstype):
        if type_ == 'RAISE':  # check invalid raises correct exception
            with pytest.raises(ValueError) as exc:
                c = self.TEST_CLASS('', type=arg)
            assert str(exc.value) == f'{arg!r} is not a valid Nds2ChannelType'
        else:
            c = self.TEST_CLASS('', type=arg)
            assert getattr(c, 'type') == type_
            assert getattr(c, 'ndstype') == ndstype

    @pytest.mark.parametrize('arg, dtype', [
        (None, None),
        (16, numpy.dtype('float64')),
        (float, numpy.dtype('float64')),
        ('float', numpy.dtype('float64')),
        ('float64', numpy.dtype('float64')),
        ('u4', numpy.dtype('uint32')),
    ])
    def test_dtype(self, arg, dtype):
        new = self.TEST_CLASS('test', dtype=arg)
        assert new.dtype is dtype

    @pytest.mark.parametrize('url', [
        None,
        'https://blah',
        'file://local/path',
        'BAD',
        1,
    ])
    def test_url(self, url):
        if url is not None and not str(url).startswith(('http', 'file')):
            with pytest.raises(ValueError) as exc:
                new = self.TEST_CLASS('test', url=url)
            assert str(exc.value) == f"Invalid URL {url!r}"
        else:
            new = self.TEST_CLASS('test', url=url)
            assert new.url == url

    def test_frametype(self):
        new = self.TEST_CLASS('test', frametype='BLAH')
        assert new.frametype == 'BLAH'

    @pytest.mark.parametrize('name, texname', [
        ('X1:TEST', 'X1:TEST'),
        ('X1:TEST-CHANNEL_NAME', r'X1:TEST-CHANNEL\_NAME'),
    ])
    def test_texname(self, name, texname):
        new = self.TEST_CLASS(name)
        assert new.texname == texname

    @pytest.mark.parametrize('ndstype, ndsname', [
        (None, 'X1:TEST'),
        ('m-trend', 'X1:TEST,m-trend'),
        ('raw', 'X1:TEST'),
    ])
    def test_ndsname(self, ndstype, ndsname):
        new = self.TEST_CLASS('X1:TEST', type=ndstype)
        assert new.ndsname == ndsname

    # -- test methods ---------------------------

    def test_copy(self):
        new = self.TEST_CLASS('X1:TEST', sample_rate=128, unit='m',
                              frequency_range=(1, 40), safe=False,
                              dtype='float64')
        copy = new.copy()
        for attr in ('name', 'ifo', 'system', 'subsystem', 'signal',
                     'trend', 'type', 'sample_rate', 'unit', 'dtype',
                     'frametype', 'model', 'url', 'frequency_range', 'safe'):
            a = getattr(new, attr)
            b = getattr(copy, attr)
            if isinstance(a, units.Quantity):
                utils.assert_quantity_equal(a, b)
            else:
                assert a == b

    @pytest.mark.parametrize('name, pdict', [
        ('X1:TEST-CHANNEL_NAME_PARSING.rms,m-trend', {
            'ifo': 'X1',
            'system': 'TEST',
            'subsystem': 'CHANNEL',
            'signal': 'NAME_PARSING',
            'trend': 'rms',
            'type': 'm-trend',
        }),
        ('G1:PSL_SL_PWR-AMPL-OUTLP-av', {
            'ifo': 'G1',
            'system': 'PSL',
            'subsystem': 'SL',
            'signal': 'PWR-AMPL-OUTLP',
            'trend': 'av',
            'type': None,
        }),
        ('V1:h_16384Hz', {
            'ifo': 'V1',
            'system': 'h',
            'subsystem': '16384Hz',
            'signal': None,
            'trend': None,
            'type': None,
        }),
        ('V1:Sa_PR_f0_zL_500Hz', {
            'ifo': 'V1',
            'system': 'Sa',
            'subsystem': 'PR',
            'signal': 'f0_zL_500Hz',
            'trend': None,
            'type': None,
        }),
        ('LVE-EX:X3_810BTORR.mean,m-trend', {
            'ifo': None,
            'system': 'X3',
            'subsystem': '810BTORR',
            'signal': None,
            'trend': 'mean',
            'type': 'm-trend',
        })
    ])
    def test_parse_channel_name(self, name, pdict):
        # check empty parse via __init__
        c = self.TEST_CLASS('')
        for key in pdict:
            assert getattr(c, key) is None

        # check errors
        with pytest.raises(ValueError) as exc:
            self.TEST_CLASS.parse_channel_name('blah')
        assert str(exc.value) == ('Cannot parse channel name according to '
                                  'LIGO channel-naming convention T990033')

        # check parsing returns expected result
        assert self.TEST_CLASS.parse_channel_name(name) == pdict

        # check parsing translates to attributes
        c = self.TEST_CLASS(name)
        for key in pdict:
            assert getattr(c, key) == pdict[key]
        assert c.ndsname == name

    @pytest.mark.requires("ciecplib")
    @pytest.mark.parametrize('name', ('X1:TEST-CHANNEL', 'Y1:TEST_CHANNEL'))
    def test_query(self, name):
        requests_mock = pytest.importorskip("requests_mock")
        # build fake CIS response
        channelinfo = {'X1:TEST-CHANNEL': {
            'name': 'X1:TEST-CHANNEL',
            'units': 'm',
            'datarate': 16384,
            'datatype': 4,
            'source': 'X1MODEL',
            'displayurl': 'https://cis.ligo.org/channel/123456',
        }}
        if name in channelinfo:
            results = [channelinfo[name]]
        else:
            results = []

        # mock response and test parsing
        with requests_mock.Mocker() as rmock:
            rmock.get(
                f"https://cis.ligo.org/api/channel/?q={name}",
                json=results,
            )
            if name == 'X1:TEST-CHANNEL':
                c = self.TEST_CLASS.query(
                    name,
                    kerberos=False,
                    idp="https://idp.example.com/profile/SAML2/SOAP/ECP",
                )
                assert c.name == 'X1:TEST-CHANNEL'
                assert c.unit == units.m
                assert c.sample_rate == 16384 * units.Hz
                assert c.dtype == numpy.dtype('float32')
                assert c.model == 'x1model'
                assert c.url == 'https://cis.ligo.org/channel/123456'
            else:
                with pytest.raises(ValueError) as exc:
                    self.TEST_CLASS.query(
                        name,
                        kerberos=False,
                        idp="https://idp.example.com/profile/SAML2/SOAP/ECP",
                    )
                assert str(exc.value) == f'No channels found matching {name!r}'

    @pytest.mark.parametrize('name', ('X1:TEST-CHANNEL', 'Y1:TEST_CHANNEL'))
    @pytest.mark.requires("nds2")
    def test_query_nds2(self, name):
        # mock NDS2 query
        ndsb = mocks.nds2_buffer('X1:TEST-CHANNEL', [], 0, 64, 'm')
        if ndsb.name == name:
            buffers = [ndsb]
        else:
            buffers = []
        conn = mocks.nds2_connection(buffers=buffers)
        with mock.patch('nds2.connection') as ndsc, \
                mock.patch('nds2.buffer', ndsb):
            ndsc.return_value = conn

            # test query_nds2
            if buffers:
                c = self.TEST_CLASS.query_nds2(name, host='test')
                assert c.name == name
                assert c.sample_rate == 64 * units.Hz
                assert c.unit == units.m
                assert c.dtype == numpy.dtype('float32')
                assert c.type == 'raw'
            else:
                with pytest.raises(ValueError):
                    c = self.TEST_CLASS.query_nds2(name, host='test')

    @pytest.mark.requires("nds2")
    def test_from_nds2(self):
        nds2c = mocks.nds2_channel('X1:TEST-CHANNEL', 64, 'm')
        c = self.TEST_CLASS.from_nds2(nds2c)
        assert c.name == 'X1:TEST-CHANNEL'
        assert c.sample_rate == 64 * units.Hz
        assert c.unit == units.m
        assert c.dtype == numpy.dtype('float32')
        assert c.type == 'raw'


# -- ChannelList --------------------------------------------------------------

class TestChannelList(object):
    TEST_CLASS = ChannelList
    ENTRY_CLASS = Channel

    NAMES = ['X1:GWPY-CHANNEL_1', 'X1:GWPY-CHANNEL_2', 'X1:GWPY-CHANNEL_3']
    SAMPLE_RATES = [1, 4, 8]

    @classmethod
    @pytest.fixture()
    def instance(cls):
        return cls.TEST_CLASS([cls.ENTRY_CLASS(n, sample_rate=s) for
                               n, s in zip(cls.NAMES, cls.SAMPLE_RATES)])

    def test_from_names(self):
        cl = self.TEST_CLASS.from_names(*self.NAMES)
        assert cl == list(map(self.ENTRY_CLASS, self.NAMES))

        cl2 = self.TEST_CLASS.from_names(','.join(self.NAMES))
        assert cl == cl2

    def test_find(self, instance):
        assert instance.find(self.NAMES[2]) == 2
        with pytest.raises(ValueError):
            instance.find('blah')

    def test_sieve(self, instance):
        cl = instance.sieve(name='GWPY-CHANNEL')
        assert cl == instance

        cl = instance.sieve(name='X1:GWPY-CHANNEL_2', exact_match=True)
        assert cl[0] is instance[1]

        cl = instance.sieve(name='GWPY-CHANNEL', sample_range=[2, 16])
        assert cl == instance[1:]

    @pytest.mark.requires("nds2")
    def test_query_nds2(self):
        # mock NDS2 query
        buffers = []
        for name, fs in zip(self.NAMES[:-1], self.SAMPLE_RATES[:-1]):
            buffers.append(mocks.nds2_buffer(name, [], 0, fs, 'm'))
        conn = mocks.nds2_connection(buffers=buffers)
        with mock.patch('nds2.connection') as ndsc:
            ndsc.return_value = conn

            # test query_nds2
            c = self.TEST_CLASS.query_nds2(self.NAMES[:-1], host='test')
            assert len(c) == len(self.NAMES) - 1
            assert c[0].name == self.NAMES[0]
            assert c[0].sample_rate == self.SAMPLE_RATES[0] * units.Hz

            # check errors
            assert len(
                self.TEST_CLASS.query_nds2([self.NAMES[-1]], host='test')) == 0

    @pytest.mark.requires("nds2")
    def test_query_nds2_availability(self):
        # mock NDS2 connection
        ndsb = mocks.nds2_buffer(self.NAMES[0], [], 0, 64, 'm')
        conn = mocks.nds2_connection(buffers=[ndsb])
        # mock availability
        availability = [mocks.nds2_availability(self.NAMES[0],
                                                [(0, 10), (20, 30)])]
        conn.get_availability = lambda x: availability

        with mock.patch('nds2.connection') as ndsc:
            ndsc.return_value = conn

            avail = self.TEST_CLASS.query_nds2_availability(
                [self.NAMES[0]], 0, 30, host='test')

            assert isinstance(avail, SegmentListDict)
            utils.assert_segmentlist_equal(avail[self.NAMES[0]],
                                           [(0, 10), (20, 30)])

    def test_read_write_omega_config(self, tmp_path):
        tmp = tmp_path / "config.ini"
        # write OMEGA_CONFIG to file and read it back
        with open(tmp, 'w') as f:
            f.write(OMEGA_CONFIG)
        cl = self.TEST_CLASS.read(tmp, format='omega-scan')
        assert len(cl) == 2
        assert cl[0].name == 'L1:GDS-CALIB_STRAIN'
        assert cl[0].sample_rate == 4096 * units.Hertz
        assert cl[0].frametype == 'L1_HOFT_C00'
        assert cl[0].params == {
            'channelName': 'L1:GDS-CALIB_STRAIN',
            'frameType': 'L1_HOFT_C00',
            'sampleFrequency': 4096,
            'searchTimeRange': 64,
            'searchFrequencyRange': (0, float('inf')),
            'searchQRange': (4, 96),
            'searchMaximumEnergyLoss': 0.2,
            'whiteNoiseFalseRate': 1e-3,
            'searchWindowDuration': 0.5,
            'plotTimeRanges': (1, 4, 16),
            'plotFrequencyRange': (),
            'plotNormalizedEnergyRange': (0, 25.5),
            'alwaysPlotFlag': 1,
        }
        assert cl[1].name == 'L1:PEM-CS_SEIS_LVEA_VERTEX_Z_DQ'
        assert cl[1].frametype == 'L1_R'

        # write omega config again using ChannelList.write and read it back
        # and check that the two lists match
        with open(tmp, 'w') as f:
            cl.write(f, format='omega-scan')
        cl2 = type(cl).read(tmp, format='omega-scan')
        assert cl == cl2

    def test_read_write_clf(self, tmp_path):
        tmp = tmp_path / "config.clf"
        # write clf to file and read it back
        with open(tmp, 'w') as f:
            f.write(CLF)
        cl = ChannelList.read(tmp)
        assert len(cl) == 4
        a = cl[0]
        assert a.name == 'H1:GDS-CALIB_STRAIN'
        assert a.sample_rate == 16384 * units.Hz
        assert a.frametype == 'H1_HOFT_C00'
        assert a.frequency_range[0] == 4. * units.Hz
        assert a.frequency_range[1] == float('inf') * units.Hz
        assert a.safe is False
        assert a.params == {'qhigh': '150', 'safe': 'unsafe',
                            'fidelity': 'clean'}
        b = cl[1]
        assert b.name == 'H1:ISI-GND_STS_HAM2_X_DQ'
        assert b.frequency_range[0] == .1 * units.Hz
        assert b.frequency_range[1] == 60. * units.Hz
        c = cl[2]
        assert c.name == 'H1:ISI-GND_STS_HAM2_Y_DQ'
        assert c.sample_rate == 256 * units.Hz
        assert c.safe is False
        d = cl[3]
        assert d.name == 'H1:ISI-GND_STS_HAM2_Z_DQ'
        assert d.safe is True
        assert d.params['fidelity'] == 'glitchy'

        # write clf again using ChannelList.write and read it back
        # and check that the two lists match
        with open(tmp, 'w') as f:
            cl.write(f)
        cl2 = type(cl).read(tmp)
        assert cl == cl2
