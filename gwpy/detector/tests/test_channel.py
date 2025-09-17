# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Test the `Channel` and `ChannelList` objects."""

from unittest import mock

import numpy
import pytest
from astropy import units

from ...segments import (
    Segment,
    SegmentList,
)
from ...testing import (
    mocks,
    utils,
)
from .. import (
    Channel,
    ChannelList,
)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

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


# -- Channel -------------------------

class TestChannel:
    """Test `Channel`."""

    TEST_CLASS = Channel

    # -- test creation ---------------

    def test_empty(self):
        """Test `Channel` with no params."""
        new = self.TEST_CLASS("")
        assert str(new) == ""
        assert new.sample_rate is None
        assert new.dtype is None

    def test_new(self):
        """Test `Channel` creation with params."""
        new = self.TEST_CLASS(
            "X1:GWPY-TEST_CHANNEL_NAME",
            sample_rate=64,
            unit="m",
        )
        assert str(new) == "X1:GWPY-TEST_CHANNEL_NAME"
        assert new.ifo == "X1"
        assert new.system == "GWPY"
        assert new.subsystem == "TEST"
        assert new.signal == "CHANNEL_NAME"
        assert new.sample_rate == 64 * units.Hz
        assert new.unit is units.meter

        new2 = self.TEST_CLASS(new)
        assert new2.sample_rate == new.sample_rate
        assert new2.unit == new.unit
        assert new2.texname == new.texname

    # -- test properties -------------

    @pytest.mark.parametrize(("arg", "fs"), [
        pytest.param(None, None, id="None"),
        pytest.param(1, 1 * units.Hz, id="float"),
        pytest.param(1 * units.Hz, 1 * units.Hz, id="hz"),
        pytest.param(1000 * units.mHz, 1 * units.Hz, id="mHz"),
        pytest.param("1", 1 * units.Hz, id="string"),
    ])
    def test_sample_rate(self, arg, fs):
        """Test `Channel.sample_rate`."""
        new = self.TEST_CLASS("test", sample_rate=arg)
        if arg is not None:
            assert isinstance(new.sample_rate, units.Quantity)
        assert new.sample_rate == fs

    @pytest.mark.parametrize(("arg", "unit"), [
        (None, None),
        ("m", units.m),
    ])
    def test_unit(self, arg, unit):
        """Test `Channel.unit`."""
        new = self.TEST_CLASS("test", unit=arg)
        if arg is not None:
            assert isinstance(new.unit, units.UnitBase)
        assert new.unit == unit

    def test_frequency_range(self):
        """Test `Channel.frequency_range`."""
        new = self.TEST_CLASS("test", frequency_range=(1, 40))
        assert isinstance(new.frequency_range, units.Quantity)
        utils.assert_quantity_equal(new.frequency_range, (1, 40) * units.Hz)

        with pytest.raises(TypeError):
            Channel("", frequency_range=1)

    def test_safe(self):
        """Test `Channel.safe`."""
        new = self.TEST_CLASS("")
        assert new.safe is None

    @pytest.mark.parametrize(("value", "result"), [
        pytest.param(None, None, id="None"),
        pytest.param(True, True, id="True"),
        pytest.param(1, True, id="truthy"),
    ])
    def test_safe_setter(self, value, result):
        """Test `Channel.safe` property setting."""
        new = self.TEST_CLASS("")
        new.safe = value  # type: ignore[assignment]
        assert new.safe == result

    @pytest.mark.parametrize(("arg", "model"), [
        (None, None),
        ("H1ASCIMC", "h1ascimc"),
    ])
    def test_model(self, arg, model):
        """Test `Channel.model`."""
        new = self.TEST_CLASS("test", model=arg)
        assert new.model == model

    @pytest.mark.parametrize(("arg", "type_", "ndstype"), [
        pytest.param(None, None, 0, id="unknown"),
        pytest.param("m-trend", "m-trend", 16, id="m-trend"),
        pytest.param(8, "s-trend", 8, id="s-trend"),
    ])
    def test_type_ndstype(self, arg, type_, ndstype):
        """Test `Channel.ndstype`."""
        c = self.TEST_CLASS("", type=arg)
        assert c.type == type_
        assert c.ndstype == ndstype

    def test_type_ndstype_unknown(self):
        """Test `Channel.ndstype` 'unknown'."""
        assert self.TEST_CLASS("", type="blah").type == "UNKNOWN"
        assert self.TEST_CLASS("", type="blah").ndstype == 0

    @pytest.mark.parametrize(("arg", "dtype"), [
        (None, None),
        (16, numpy.dtype("float64")),
        (float, numpy.dtype("float64")),
        ("float", numpy.dtype("float64")),
        ("float64", numpy.dtype("float64")),
        ("u4", numpy.dtype("uint32")),
        (bool, numpy.dtype("bool")),
    ])
    def test_dtype(self, arg, dtype):
        """Test `Channel.dtype`."""
        new = self.TEST_CLASS("test", dtype=arg)
        assert new.dtype is dtype

    @pytest.mark.parametrize("url", [
        None,
        "https://blah",
        "file://local/path",
    ])
    def test_url(self, url):
        """Test `Channel.url`."""
        new = self.TEST_CLASS("test", url=url)
        assert new.url == url

    @pytest.mark.parametrize("url", [
        "BAD",
        "ftp://example.com/data",
    ])
    def test_url_error(self, url):
        """Test `Channel.url` error handling."""
        with pytest.raises(ValueError, match=fr"^Invalid URL {url!r}$"):
            self.TEST_CLASS("test", url=url)

    def test_frametype(self):
        """Test `Channel.frametype`."""
        new = self.TEST_CLASS("test", frametype="BLAH")
        assert new.frametype == "BLAH"

    @pytest.mark.parametrize(("name", "texname"), [
        ("X1:TEST", "X1:TEST"),
        ("X1:TEST-CHANNEL_NAME", r"X1:TEST-CHANNEL\_NAME"),
    ])
    def test_texname(self, name, texname):
        """Test `Channel.texname`."""
        new = self.TEST_CLASS(name)
        assert new.texname == texname

    @pytest.mark.parametrize(("ndstype", "ndsname"), [
        (None, "X1:TEST"),
        ("m-trend", "X1:TEST,m-trend"),
        ("raw", "X1:TEST"),
    ])
    def test_ndsname(self, ndstype, ndsname):
        """Test `Channel.ndsname`."""
        new = self.TEST_CLASS("X1:TEST", type=ndstype)
        assert new.ndsname == ndsname

    # -- test magic ------------------

    @pytest.mark.parametrize(("name", "result"), [
        ("X1:TEST", "X1:TEST"),
        ("X1:TEST,m-trend", "X1:TEST"),
    ])
    def test_str(self, name, result):
        """Test `str(channel)`."""
        assert str(self.TEST_CLASS(name)) == result

    @mock.patch("builtins.hex", lambda x: 12345)  # noqa: ARG005,PT008
    def test_repr(self):
        """Test `repr(channel)`."""
        new = self.TEST_CLASS(
            "X1:TEST-CHANNEL",
            type=8,
            sample_rate=16384,
        )
        assert repr(new) == (
            f'<{type(new).__name__}("{new}" [s-trend], 16384.0 Hz) at 12345>'
        )

    @pytest.mark.parametrize(("a", "b", "result"), [
        pytest.param(
            TEST_CLASS("test"),
            TEST_CLASS("test"),
            True,
            id="channel match",
        ),
        pytest.param(
            TEST_CLASS("test", sample_rate=2),
            TEST_CLASS("test", sample_rate=4),
            False,
            id="channel mismatch",
        ),
        pytest.param(
            TEST_CLASS("test"),
            None,
            False,
            id="wrong type",
        ),
    ])
    def test_eq(self, a, b, result):
        """Test `Channel.__eq__`."""
        assert a.__eq__(b) is result

    # -- test methods ----------------

    def test_copy(self):
        """Test `Channel.copy()`."""
        new = self.TEST_CLASS(
            "X1:TEST",
            sample_rate=128,
            unit="m",
            frequency_range=(1, 40),
            safe=False,
            dtype="float64",
        )
        copy = new.copy()
        for attr in (
            "name",
            "ifo",
            "system",
            "subsystem",
            "signal",
            "trend",
            "type",
            "sample_rate",
            "unit",
            "dtype",
            "frametype",
            "model",
            "url",
            "frequency_range",
            "safe",
        ):
            a = getattr(new, attr)
            b = getattr(copy, attr)
            if isinstance(a, units.Quantity):
                utils.assert_quantity_equal(a, b)
            else:
                assert a == b

    @pytest.mark.parametrize(("name", "pdict"), [
        pytest.param(
            "X1:TEST-CHANNEL_NAME_PARSING.rms,m-trend",
            {
                "ifo": "X1",
                "system": "TEST",
                "subsystem": "CHANNEL",
                "signal": "NAME_PARSING",
                "trend": "rms",
                "type": "m-trend",
            },
            id="LIGO-m-trend",
        ),
        pytest.param(
            "G1:PSL_SL_PWR-AMPL-OUTLP-av",
            {
                "ifo": "G1",
                "system": "PSL",
                "subsystem": "SL",
                "signal": "PWR-AMPL-OUTLP",
                "trend": "av",
                "type": None,
            },
            id="GEO trend",
        ),
        pytest.param(
            "V1:h_16384Hz",
            {
                "ifo": "V1",
                "system": "h",
                "subsystem": "16384Hz",
                "signal": None,
                "trend": None,
                "type": None,
            },
            id="Virgo",
        ),
        pytest.param(
            "V1:Sa_PR_f0_zL_500Hz",
            {
                "ifo": "V1",
                "system": "Sa",
                "subsystem": "PR",
                "signal": "f0_zL_500Hz",
                "trend": None,
                "type": None,
            },
            id="Virgo2",
        ),
        pytest.param(
            "LVE-EX:X3_810BTORR.mean,m-trend",
            {
                "ifo": None,
                "system": "X3",
                "subsystem": "810BTORR",
                "signal": None,
                "trend": "mean",
                "type": "m-trend",
            },
            id="LIGO-LVE",
        ),
    ])
    def test_parse_channel_name(self, name, pdict):
        """Test `Channel.parse_channel_name()`."""
        # check empty parse via __init__
        c = self.TEST_CLASS("")
        for key in pdict:
            assert getattr(c, key) is None

        # check errors
        with pytest.raises(
            ValueError,
            match=r"^Cannot parse channel name according to LIGO-T990033$",
        ):
            self.TEST_CLASS.parse_channel_name("blah")

        # check parsing returns expected result
        assert self.TEST_CLASS.parse_channel_name(name) == pdict

        # check parsing translates to attributes
        c = self.TEST_CLASS(name)
        for key in pdict:
            assert getattr(c, key) == pdict[key]
        assert c.ndsname == name

    @pytest.mark.requires("ciecplib")
    def test_query(self, requests_mock):
        """Test `Channel.query`."""
        # build fake CIS response
        name = "X1:TEST-CHANNEL"
        results = [{
            "name": name,
            "units": "m",
            "datarate": 16384,
            "datatype": 4,
            "source": "X1MODEL",
            "displayurl": "https://cis.ligo.org/channel/123456",
        }]

        # apply the mock
        requests_mock.get(
            f"https://cis.ligo.org/api/channel/?q={name}",
            json=results,
        )

        # run the query
        chan = self.TEST_CLASS.query(
            name,
            idp="https://idp.example.com/profile/SAML2/SOAP/ECP",
            kerberos=False,
        )

        # check
        assert chan.name == name
        assert chan.unit == units.m
        assert chan.sample_rate == 16384 * units.Hz
        assert chan.dtype == numpy.dtype("float32")
        assert chan.model == "x1model"
        assert chan.url == "https://cis.ligo.org/channel/123456"

    @pytest.mark.requires("ciecplib")
    def test_query_notfound(self, requests_mock):
        """Test `Channel.query` handling of an empty response."""
        name = "X1:TEST-CHANNEL"
        requests_mock.get(
            f"https://cis.ligo.org/api/channel/?q={name}",
            json=[],
        )
        with pytest.raises(
            ValueError,
            match=fr"^No channels found matching '{name}'$",
        ):
            self.TEST_CLASS.query(
                name,
                idp="https://idp.example.com/profile/SAML2/SOAP/ECP",
                kerberos=False,
            )

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_query_nds2(self):
        """Test `Channel.query_nds2()`."""
        c = self.TEST_CLASS.query_nds2("X1:test", host="test")
        assert c.name == "X1:test"
        assert c.sample_rate == 16 * units.Hz
        assert c.unit == units.m
        assert c.dtype == numpy.dtype("float32")
        assert c.type == "raw"

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_query_nds2_error(self):
        """Test `Channel.query_nds2()` error handling."""
        with pytest.raises(
            ValueError,
            match="unique NDS2 channel match not found for 'Z1:test'",
        ):
            self.TEST_CLASS.query_nds2("Z1:test", host="test")

    @pytest.mark.requires("nds2")
    def test_from_nds2(self):
        """Test `Channel.from_nds2()`."""
        nds2c = mocks.nds2_channel("X1:TEST-CHANNEL", 64, "m")
        c = self.TEST_CLASS.from_nds2(nds2c)
        assert c.name == "X1:TEST-CHANNEL"
        assert c.sample_rate == 64 * units.Hz
        assert c.unit == units.m
        assert c.dtype == numpy.dtype("float32")
        assert c.type == "raw"

    @pytest.mark.requires("arrakis")
    def test_from_arrakis(self):
        """Test `Channel.from_arrakis()`."""
        import arrakis
        maker = arrakis.Channel(
            "X1:TEST-CHANNEL",
            numpy.dtype("int32"),
            128,
        )
        chan = self.TEST_CLASS.from_arrakis(maker)
        assert chan.name == maker.name
        assert chan.sample_rate == maker.sample_rate * units.Hz
        assert chan.dtype == maker.data_type


# -- ChannelList ---------------------

class TestChannelList:
    """Test `ChannelList`."""

    TEST_CLASS = ChannelList
    ENTRY_CLASS = Channel

    NAMES = (
        "X1:GWPY-CHANNEL_1",
        "X1:GWPY-CHANNEL_2",
        "X1:GWPY-CHANNEL_3",
    )
    SAMPLE_RATES = (1, 4, 8)

    @pytest.fixture
    @classmethod
    def instance(cls):
        """Create a new instance of the class under test."""
        return cls.TEST_CLASS([
            cls.ENTRY_CLASS(n, sample_rate=s) for
            n, s in zip(cls.NAMES, cls.SAMPLE_RATES, strict=True)
        ])

    def test_from_names(self):
        """Test `ChannelList.from_names()`."""
        cl = self.TEST_CLASS.from_names(*self.NAMES)
        assert cl == list(map(self.ENTRY_CLASS, self.NAMES))

        cl2 = self.TEST_CLASS.from_names(",".join(self.NAMES))
        assert cl == cl2

    def test_find(self, instance):
        """Test `ChannelList.find()`."""
        assert instance.find(self.NAMES[2]) == 2

    def test_find_error(self, instance):
        """Test `ChannelList.find()` error handling."""
        with pytest.raises(
            ValueError,
            match="blah",
        ):
            instance.find("blah")

    def test_sieve(self, instance):
        """Test `ChannelList.sieve()`."""
        cl = instance.sieve(name="GWPY-CHANNEL")
        assert cl == instance

        cl = instance.sieve(name="X1:GWPY-CHANNEL_2", exact_match=True)
        assert cl[0] is instance[1]

        cl = instance.sieve(name="GWPY-CHANNEL", sample_range=[2, 16])
        assert cl == instance[1:]

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_query_nds2(self):
        """Test `ChannelList.query_nds2()`."""
        # test query_nds2
        c = self.TEST_CLASS.query_nds2(
            ["X1:test", "Y1:test"],
            host="test",
        )
        assert len(c) == 2
        assert c[0].name == "X1:test"
        assert c[0].sample_rate == 16 * units.Hz

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_query_nds2_error(self):
        """Test `ChannelList.query_nds2` error handling."""
        # check errors
        assert len(
            self.TEST_CLASS.query_nds2(["Z1:test"], host="test"),
        ) == 0

    @pytest.mark.requires("nds2")
    @pytest.mark.usefixtures("nds2_connection")
    def test_query_nds2_availability(self):
        """Test `ChannelList.query_nds2_availability`."""
        avail = self.TEST_CLASS.query_nds2_availability(
            ["X1:test"],
            1000000000,
            1000000010,
            host="test",
        )
        utils.assert_dict_equal(
            avail,
            {"X1:test": SegmentList([Segment(1000000000, 1000000008)])},
            utils.assert_segmentlist_equal,
        )

    def test_read_write_omega_config(self, tmp_path):
        """Test `ChannelList.read(... format='omega-scan')`."""
        tmp = tmp_path / "config.ini"
        # write OMEGA_CONFIG to file and read it back
        with tmp.open("w") as f:
            f.write(OMEGA_CONFIG)
        cl = self.TEST_CLASS.read(tmp, format="omega-scan")
        assert len(cl) == 2
        assert cl[0].name == "L1:GDS-CALIB_STRAIN"
        assert cl[0].sample_rate == 4096 * units.Hertz
        assert cl[0].frametype == "L1_HOFT_C00"
        assert cl[0].params == {
            "channelName": "L1:GDS-CALIB_STRAIN",
            "frameType": "L1_HOFT_C00",
            "sampleFrequency": 4096,
            "searchTimeRange": 64,
            "searchFrequencyRange": (0, float("inf")),
            "searchQRange": (4, 96),
            "searchMaximumEnergyLoss": 0.2,
            "whiteNoiseFalseRate": 1e-3,
            "searchWindowDuration": 0.5,
            "plotTimeRanges": (1, 4, 16),
            "plotFrequencyRange": (),
            "plotNormalizedEnergyRange": (0, 25.5),
            "alwaysPlotFlag": 1,
        }
        assert cl[1].name == "L1:PEM-CS_SEIS_LVEA_VERTEX_Z_DQ"
        assert cl[1].frametype == "L1_R"

        # write omega config again using ChannelList.write and read it back
        # and check that the two lists match
        with tmp.open("w") as f:
            cl.write(f, format="omega-scan")
        cl2 = type(cl).read(tmp, format="omega-scan")
        assert cl == cl2

    def test_read_write_clf(self, tmp_path):
        """Test `ChannelList.read(... format='ini')`."""
        tmp = tmp_path / "config.clf"
        # write clf to file and read it back
        with tmp.open("w") as f:
            f.write(CLF)
        cl = ChannelList.read(tmp)
        assert len(cl) == 4
        a = cl[0]
        assert a.name == "H1:GDS-CALIB_STRAIN"
        assert a.sample_rate == 16384 * units.Hz
        assert a.frametype == "H1_HOFT_C00"
        assert a.frequency_range[0] == 4. * units.Hz
        assert a.frequency_range[1] == float("inf") * units.Hz
        assert a.safe is False
        assert a.params == {
            "qhigh": "150",
            "safe": "unsafe",
            "fidelity": "clean",
        }
        b = cl[1]
        assert b.name == "H1:ISI-GND_STS_HAM2_X_DQ"
        assert b.frequency_range[0] == .1 * units.Hz
        assert b.frequency_range[1] == 60. * units.Hz
        c = cl[2]
        assert c.name == "H1:ISI-GND_STS_HAM2_Y_DQ"
        assert c.sample_rate == 256 * units.Hz
        assert c.safe is False
        d = cl[3]
        assert d.name == "H1:ISI-GND_STS_HAM2_Z_DQ"
        assert d.safe is True
        assert d.params["fidelity"] == "glitchy"

        # write clf again using ChannelList.write and read it back
        # and check that the two lists match
        with tmp.open("w") as f:
            cl.write(f)
        cl2 = type(cl).read(tmp)
        assert cl == cl2
