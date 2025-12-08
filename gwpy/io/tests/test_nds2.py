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

"""Unit tests for :mod:`gwpy.io.nds2`."""

from __future__ import annotations

import os
import warnings
from typing import (
    Generic,
    TypeVar,
)
from unittest import mock

import pytest

from ...detector import Channel
from ...segments import (
    Segment,
    SegmentList,
)
from ...testing import (
    mocks,
    utils,
)
from ...time import LIGOTimeGPS
from ...utils.tests.test_enum import TestNumpyTypeEnum as _TestNumpyTypeEnum
from .. import nds2 as io_nds2

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

EnumType = TypeVar("EnumType", bound=io_nds2._Nds2Enum)


class _TestNds2Enum(Generic[EnumType]):
    """Base class for tests of :class:`gwpy.io.nds2._Nds2Enum`."""

    TEST_CLASS: type[EnumType]

    def test_any(self):
        assert self.TEST_CLASS.any() == 2 * max(self.TEST_CLASS).value - 1

    def test_find_unknown(self):
        """Test 'UNKNOWN' for :meth:`gwpy.io.nds2.Nds2ChannelType.find`."""
        assert self.TEST_CLASS.find("blah").name == "UNKNOWN"


class TestNds2ChannelType(_TestNds2Enum[io_nds2.Nds2ChannelType]):
    """Tests of :class:`gwpy.io.nds2.Nds2DataType`."""

    TEST_CLASS: type[io_nds2.Nds2ChannelType] = io_nds2.Nds2ChannelType

    def test_nds2name(self):
        """Test the `nds2name` property."""
        assert self.TEST_CLASS.MTREND.nds2name == "m-trend"

    def test_nds2names(self):
        """Test the `nds2names` method."""
        expected = sorted(x.nds2name for x in self.TEST_CLASS)
        assert sorted(self.TEST_CLASS.nds2names()) == expected

    @pytest.mark.parametrize(("input_", "expected"), [
        (TEST_CLASS.MTREND.value, TEST_CLASS.MTREND),
        (TEST_CLASS.MTREND.name, TEST_CLASS.MTREND),
        (TEST_CLASS.MTREND.nds2name, TEST_CLASS.MTREND),
        ("mtrend", TEST_CLASS.MTREND),
        ("rds", TEST_CLASS.RDS),
        ("RDS", TEST_CLASS.RDS),
        ("reduced", TEST_CLASS.RDS),
        ("REDUCED", TEST_CLASS.RDS),
    ])
    def test_find(self, input_, expected):
        """Test :meth:`gwpy.io.nds2.Nds2ChannelType.find`."""
        assert self.TEST_CLASS.find(input_) == expected


class TestNds2DataType(
    _TestNds2Enum[io_nds2.Nds2DataType],
    _TestNumpyTypeEnum,
):
    """Tests of :class:`gwpy.io.nds2.Nds2DataType`."""

    TEST_CLASS = io_nds2.Nds2DataType  # type: ignore[assignment]

    def test_find_errors(self):
        """Test errors from :meth:`gwpy.io.nds2.Nds2DataType.find`."""
        pytest.skip(f"not implemented for {self.TEST_CLASS.__name__}")


@pytest.mark.parametrize(("key", "value", "result"), [
    pytest.param(
        "NDSSERVER",
        "test1.ligo.org:80,test2.ligo.org:43",
        [("test1.ligo.org", 80), ("test2.ligo.org", 43)],
        id="basic",
    ),
    pytest.param(
        "NDSSERVER",
        "test1.ligo.org:80,test2.ligo.org:43,test.ligo.org,test2.ligo.org:43",
        [("test1.ligo.org", 80), ("test2.ligo.org", 43), ("test.ligo.org", None)],
        id="repeat",
    ),
    pytest.param(
        "TESTENV",
        "test1.ligo.org:80,test2.ligo.org:43",
        [("test1.ligo.org", 80), ("test2.ligo.org", 43)],
        id="env-key",
    ),
])
def test_parse_nds_env(key, value, result):
    """Test `gwpy.io.nds2.parse_nds_env`."""
    with mock.patch.dict(os.environ, {key: value}):
        if key == "NDSSERVER":
            hosts = io_nds2.parse_nds_env()
        else:
            hosts = io_nds2.parse_nds_env(env=key)
    assert hosts == result


@mock.patch.dict(os.environ, clear=True)
@pytest.mark.parametrize(("ifo", "include_gwosc", "result"), [
    pytest.param(
        None,
        True,
        [("nds.ligo.caltech.edu", None), ("nds.gwosc.org", None)],
        id="default",
    ),
    pytest.param(
        "L1",
        False,
        [("nds.ligo-la.caltech.edu", None), ("nds.ligo.caltech.edu", None)],
        id="L1",
    ),
])
def test_host_resolution_order(ifo, include_gwosc, result):
    """Test `gwpy.io.nds2.host_resolution_order` basic usage."""
    assert io_nds2.host_resolution_order(
        ifo,
        include_gwosc=include_gwosc,
    ) == result


@mock.patch.dict(
    os.environ,
    {"NDSSERVER": "test1.ligo.org:80,test2.ligo.org:43"},
)
@pytest.mark.parametrize(("ifo", "include_gwosc", "result"), [
    pytest.param(
        None,
        False,
        [
            ("test1.ligo.org", 80),
            ("test2.ligo.org", 43),
            ("nds.ligo.caltech.edu", None),
        ],
        id="noifo",
    ),
    pytest.param(
        "L1",
        True,
        [
            ("test1.ligo.org", 80),
            ("test2.ligo.org", 43),
            ("nds.ligo-la.caltech.edu", None),
            ("nds.ligo.caltech.edu", None),
            ("nds.gwosc.org", None),
        ],
        id="L1",
    ),
])
def test_host_resolution_order_env(ifo, include_gwosc, result):
    """Test `host_resolution_order` environment parsing."""
    assert io_nds2.host_resolution_order(
        ifo,
        include_gwosc=include_gwosc,
    ) == result


@mock.patch.dict(
    os.environ,
    {"TESTENV": "test1.ligo.org:80,test2.ligo.org:43"},
)
def test_host_resolution_order_named_env():
    """Test `host_resolution_order` with a named environment variable."""
    hro = io_nds2.host_resolution_order(
        "V1",
        env="TESTENV",
        include_gwosc=False,
    )
    assert hro == [
        ("test1.ligo.org", 80),
        ("test2.ligo.org", 43),
        ("nds.ligo.caltech.edu", None),
    ]


@mock.patch.dict(
    os.environ,
    {"TESTENV": "test1.ligo.org:80,test2.ligo.org:43"},
)
@pytest.mark.parametrize(("ifo", "epoch", "env", "result"), [
    pytest.param(
        "L1",
        "Jan 1 2015",
        None,
        [
            ("nds.ligo.caltech.edu", None),
            ("nds.ligo-la.caltech.edu", None),
        ],
        id="old",
    ),
    pytest.param(
        "L1",
        "now",
        "TESTENV",
        [
            ("test1.ligo.org", 80),
            ("test2.ligo.org", 43),
            ("nds.ligo-la.caltech.edu", None),
            ("nds.ligo.caltech.edu", None),
        ],
        id="now",
    ),
])
def test_host_resolution_order_epoch(ifo, epoch, env, result):
    """Test `gwpy.io.nds2.host_resolution_order` epoch parsing."""
    assert io_nds2.host_resolution_order(
        ifo,
        epoch=epoch,
        env=env,
        include_gwosc=False,
    ) == result


@mock.patch.dict(
    os.environ,
    {"TESTENV": "test1.ligo.org:80,test2.ligo.org:43"},
)
def test_host_resolution_order_warning():
    """Test `gwpy.io.nds2.host_resolution_order` warnings."""
    # test warnings for unknown IFO
    with pytest.warns(
        UserWarning,
        match="no default host found for ifo 'X1'",
    ):
        # should produce warning
        hro = io_nds2.host_resolution_order(
            "X1",
            env=None,
            include_gwosc=False,
        )
    assert hro == [("nds.ligo.caltech.edu", None)]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        # should _not_ produce warning
        hro = io_nds2.host_resolution_order("X1", env="TESTENV")


@pytest.mark.requires("nds2")
@pytest.mark.parametrize(("host", "port", "callport"), [
    pytest.param("nds.test.gwpy", None, None, id="None"),
    pytest.param("nds.test.gwpy", 31200, 31200, id="31200"),
    pytest.param("x1nds9", None, 8088, id="nds1"),
])
@mock.patch("nds2.connection")
def test_connect(connector, host, port, callport):
    """Test `gwpy.io.nds2.connect`."""
    io_nds2.connect(host, port=port)
    if callport is None:
        connector.assert_called_once_with(host)
    else:
        connector.assert_called_once_with(host, callport)


@pytest.mark.requires("nds2")
@mock.patch("gwpy.io.nds2.connect")
def test_auth_connect(connect):
    """Test `gwpy.io.nds2.auth_connect`."""
    io_nds2.auth_connect("host", 0)
    connect.assert_called_once_with("host", 0)


@pytest.mark.requires("nds2")
@mock.patch("gwpy.io.nds2.kinit")
@mock.patch(
    "gwpy.io.nds2.connect",
    side_effect=(
        RuntimeError("Request SASL authentication something something"),
        True,
    ),
)
def test_auth_connect_kinit(connect, kinit, caplog):
    """Test `gwpy.io.nds2.auth_connect` with a callout to `gwpy.io.kerberos.kinit`."""
    with caplog.at_level("WARNING"):
        assert io_nds2.auth_connect("host", 0)
    assert "attempting Kerberos kinit()" in caplog.text
    kinit.assert_called_with()
    assert connect.call_count == 2
    connect.assert_called_with("host", 0)


@pytest.mark.requires("nds2")
@mock.patch(
    "gwpy.io.nds2.connect",
    side_effect=RuntimeError("Anything else"),
)
def test_auth_connect_error(connect):
    """Test errors from `gwpy.io.nds2.auth_connect`."""
    with pytest.raises(
        RuntimeError,
        match="Anything else",
    ):
        io_nds2.auth_connect("host", 0)
    connect.assert_called_once_with("host", 0)


@pytest.mark.requires("nds2")
def test_find_channels(nds2_connection):
    """Test `gwpy.io.nds2.find_channels`."""
    # call function and check result
    chans = io_nds2.find_channels(
        ["X1:test"],
        host="test.nds2",
    )
    assert len(chans) == 1
    assert chans[0].name == "X1:test"

    # check callouts in find_channels
    nds2_connection.set_epoch.assert_any_call("ALL")

    # check callouts in _find_channels
    nds2_connection.find_channels.assert_called_once_with(
        "X1:test",
        io_nds2.Nds2ChannelType.any(),
        io_nds2.Nds2DataType.any(),
        io_nds2.MIN_SAMPLE_RATE,
        io_nds2.MAX_SAMPLE_RATE,
    )


@pytest.mark.requires("nds2")
def test_find_channels_nds1(nds2_connection):
    """Test NDS1 name handling in `find_channels`."""
    nds2_connection.get_protocol.return_value = 1
    io_nds2.find_channels(
        ["X1:test,m-trend"],
        host="test.nds2",
        sample_rate=16,
        dtype=float,
    )
    nds2_connection.find_channels.assert_called_with(
        "X1:test,m-trend",
        io_nds2.Nds2ChannelType.MTREND.value,
        io_nds2.Nds2DataType.FLOAT64.value,
        16,
        16,
    )


@pytest.mark.requires("nds2")
def test_find_channels_online(nds2_connection):
    """Test handling of 'online' channels in `find_channels`."""
    # add an 'online' version of thee same channel
    buff = nds2_connection._buffers[0]
    nds2_connection._buffers.append(
        mocks.nds2_buffer(
            buff.name,
            buff.data,
            LIGOTimeGPS(buff.gps_seconds, buff.gps_nanoseconds),
            buff.sample_rate,
            buff.channel.signal_units,
            channel_type=int(io_nds2.Nds2ChannelType.ONLINE),
        ),
    )
    assert io_nds2.find_channels(
        ["X1:test"],
        host="test",
        unique=True,
    ) == [nds2_connection._buffers[0].channel]


@pytest.mark.requires("nds2")
def test_find_channels_unique(nds2_connection):
    """Test handling of 'online' channels in `find_channels`."""
    # add a second copy of the same channel
    # so that find_channels returns two things
    nds2_connection._buffers.append(nds2_connection._buffers[0])

    with pytest.raises(
        ValueError,
        match=r"^unique NDS2 channel match not found for 'X1:test'$",
    ):
        io_nds2.find_channels(["X1:test"], host="test", unique=True)


@pytest.mark.requires("nds2")
def test_get_availability(nds2_connection):
    """Test `gwpy.io.nds2.get_availability`."""
    # validate call and parsing of results
    segs = io_nds2.get_availability(
        ["X1:test"],
        0,
        1,
        host="test",
    )
    utils.assert_dict_equal(
        segs,
        {"X1:test": SegmentList([Segment(1000000000.0, 1000000008.0)])},
        utils.assert_segmentlist_equal,
    )

    # check callouts
    nds2_connection.set_epoch.assert_has_calls([
        mock.call(0, 1),
        mock.call(
            nds2_connection.current_epoch().gps_start,
            nds2_connection.current_epoch().gps_stop,
        ),
    ])
    nds2_connection.get_availability.assert_called_once_with(["X1:test,raw"])


@pytest.mark.requires("nds2")
def test_get_availability_real():
    """Test `gwpy.io.nds2.get_availability` with real remote data."""
    try:
        segs = io_nds2.get_availability(
            ["L1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M"],
            1238166018,
            1238192569,
            host="nds.gwosc.org",
        )
    except RuntimeError as exc:  # something went wrong with NDS2
        pytest.skip(str(exc))
    if sum(len(seglist) for seglist in segs.values()) == 0:  # EMPTY response
        pytest.skip("no data received from nds.gwosc.org")
    utils.assert_dict_equal(
        segs,
        {"L1:ISI-GND_STS_ITMY_Z_BLRMS_30M_100M": SegmentList([
            Segment(1238166018, 1238170289),
            Segment(1238175433, 1238192569),
        ])},
        utils.assert_segmentlist_equal,
    )


@pytest.mark.parametrize(("start", "end", "out"), [
    pytest.param(
        0,
        60,
        (0, 60),
        id="no change",
    ),
    pytest.param(
        1,
        60,
        (0, 60),
        id="expand start",
    ),
    pytest.param(
        0,
        61,
        (0, 120),
        id="expand end",
    ),
    pytest.param(
        59,
        61,
        (0, 120),
        id="expand both",
    ),
    pytest.param(
        1167264018,
        1198800018,
        (1167264000, 1198800060),
        id="expand both GPS",
    ),
])
def test_minute_trend_times(start, end, out):
    """Test `gwpy.io.nds2.minute_trend_times`."""
    assert io_nds2.minute_trend_times(start, end) == out


@pytest.mark.requires("nds2")
def test_get_nds2_name():
    """Test `gwpy.io.nds2._get_nds2_name`."""
    # Note: we can't use parametrize because mocks.nds2_channel requires
    #       the nds2-client and is executed _before_ the skip decorator is
    #       applied
    for channel, name in [
        ("test", "test"),
        (Channel("X1:TEST", type="m-trend"), "X1:TEST,m-trend"),
        (mocks.nds2_channel("X1:TEST", 16, "NONE"), "X1:TEST,raw"),
    ]:
        assert io_nds2._get_nds2_name(channel) == name  # type: ignore[arg-type]


@pytest.mark.requires("nds2")
def test_get_nds2_names():
    """Test `gwpy.io.nds2._get_nds2_names`."""
    channels = [
        "test",
        Channel("X1:TEST", type="m-trend"),
        mocks.nds2_channel("X1:TEST", 16, "NONE"),
    ]
    names = [
        "test",
        "X1:TEST,m-trend",
        "X1:TEST,raw",
    ]
    assert list(io_nds2._get_nds2_names(channels)) == names  # type: ignore[arg-type]
