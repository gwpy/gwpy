# Copyright (c) 2026 Cardiff University
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

"""Tests for :mod:`gwpy.timeseries.io.arrakis`."""

from __future__ import annotations

import os
from math import ceil
from unittest import mock

import numpy
import pytest

from ...detector import Channel
from ...testing import utils
from .. import (
    StateVector,
    TimeSeries,
    TimeSeriesDict,
)
from ..io import arrakis as io_arrakis

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

START = 1000000000
END = START + 4
SAMPLE_RATE = 16.


def _arrakis_block(
    names: list[str],
    start: float = START,
    duration: float = 4,
    sample_rate: float = SAMPLE_RATE,
    gaps: list[str] | None = None,
):
    """Build a real `arrakis.SeriesBlock` with random data for ``names``.

    ``gaps`` (if given) lists the names that should be returned as a
    masked array with the first second flagged as a gap.
    """
    from arrakis import (
        Channel as ArrakisChannel,
        SeriesBlock as ArrakisBlock,
    )

    gaps = gaps or []
    rng = numpy.random.default_rng(seed=0)
    channels = {
        name: ArrakisChannel(name, numpy.dtype("float64"), sample_rate)
        for name in names
    }
    data = {}
    for name in names:
        values = rng.random(size=int(duration * sample_rate))
        if name in gaps:
            mask = numpy.zeros(values.shape, dtype=bool)
            mask[:int(sample_rate)] = True
            values = numpy.ma.masked_array(values, mask=mask)
        data[name] = values
    return ArrakisBlock(int(start * 1e9), data, channels)


@pytest.fixture
def arrakis_client():
    """Mock `arrakis.Client` so that no network connection is attempted."""
    with mock.patch("arrakis.Client") as mock_client_class:
        mock_client_class.return_value.initial_url = "grpc://mock"
        yield mock_client_class


# -- identify_arrakis_sources --------

def test_identify_wrong_origin():
    """`identify_sources` only matches the 'get' origin."""
    assert io_arrakis.identify_sources("read") is None


@pytest.mark.requires("arrakis")
def test_identify_with_client():
    """`identify_sources` uses a directly-provided `Client`."""
    import arrakis

    client = mock.create_autospec(arrakis.Client, instance=True)
    result = io_arrakis.identify_sources("get", client=client)
    assert result == [{"client": client, "priority": 1}]


@pytest.mark.requires("arrakis")
def test_identify_with_url():
    """`identify_sources` parses an explicit URL."""
    result = io_arrakis.identify_sources("get", url="test.arrakis:1234")
    assert result == [{"url": "grpc://test.arrakis:1234", "priority": 10}]


@pytest.mark.requires("arrakis")
def test_identify_with_env():
    """`identify_sources` falls back to $ARRAKIS_SERVER."""
    with mock.patch.dict(os.environ, {"ARRAKIS_SERVER": "test.arrakis:1234"}):
        result = io_arrakis.identify_sources("get")
    assert result == [{"url": "grpc://test.arrakis:1234", "priority": 10}]


@pytest.mark.requires("arrakis")
def test_identify_no_source():
    """`identify_sources` returns nothing if no source is configured."""
    with mock.patch.dict(os.environ):
        os.environ.pop("ARRAKIS_SERVER", None)
        result = io_arrakis.identify_sources("get")
    assert result == []


# -- fetch_series / fetch_block ------

@pytest.mark.requires("arrakis")
def test_fetch_series(arrakis_client):
    """`fetch_series` returns a single `TimeSeries` from Arrakis."""
    block = _arrakis_block(["X1:TEST"])
    arrakis_client.return_value.fetch.return_value = block

    start = START + 0.5
    end = start + 1

    ts = io_arrakis.fetch_series("X1:TEST", start, end)

    utils.assert_quantity_sub_equal(
        ts,
        TimeSeries.from_arrakis(block["X1:TEST"]).crop(start, end),
    )
    arrakis_client.return_value.fetch.assert_called_once_with(
        ["X1:TEST"],
        int(start),
        ceil(end),
        on_gap="raise",
    )


@pytest.mark.requires("arrakis")
def test_fetch_block(arrakis_client):
    """`fetch_block` returns a `TimeSeriesDict` keyed by the original channels."""
    chan2 = Channel("X1:TEST-2")
    channels = ["X1:TEST-1", chan2]
    start = START + 0.5
    end = start + 1
    block = _arrakis_block(
        [str(chan) for chan in channels],
        start=start,
        duration=1,
    )
    arrakis_client.return_value.fetch.return_value = block

    tsd = io_arrakis.fetch_block(channels, start, end)

    assert isinstance(tsd, TimeSeriesDict)
    assert list(tsd.keys()) == ["X1:TEST-1", chan2]
    utils.assert_quantity_sub_equal(
        tsd["X1:TEST-1"],
        TimeSeries.from_arrakis(block["X1:TEST-1"]).crop(start, end),
    )


@pytest.mark.requires("arrakis")
def test_fetch_block_existing_client(arrakis_client):
    """`fetch_block(..., client=...)` reuses the given client."""
    # NOTE: use arrakis.client.Client (not arrakis.Client) here because the
    #       arrakis_client fixture has already replaced arrakis.Client with
    #       a mock, and autospeccing a mock raises InvalidSpecError.
    import arrakis.client

    existing = mock.create_autospec(arrakis.client.Client, instance=True)
    existing.initial_url = "grpc://existing"
    existing.fetch.return_value = _arrakis_block(["X1:TEST"])

    io_arrakis.fetch_block(["X1:TEST"], START, END, client=existing)

    arrakis_client.assert_not_called()
    existing.fetch.assert_called_once()


@pytest.mark.requires("arrakis")
def test_fetch_block_new_client(arrakis_client):
    """`fetch_block` opens a new `Client` when none is given."""
    arrakis_client.return_value.fetch.return_value = _arrakis_block(["X1:TEST"])

    io_arrakis.fetch_block(["X1:TEST"], START, END, url="test.arrakis:1234")

    arrakis_client.assert_called_once_with("test.arrakis:1234")


@pytest.mark.requires("arrakis")
def test_fetch_block_pad(arrakis_client):
    """`fetch_block(..., pad=...)` fills gaps and requests fill-mode gap handling."""
    import arrakis.client as arrakis_client_mod

    block = _arrakis_block(["X1:TEST"], gaps=["X1:TEST"])
    arrakis_client.return_value.fetch.return_value = block

    tsd = io_arrakis.fetch_block(["X1:TEST"], START, END, pad=-1.0)

    arrakis_client.return_value.fetch.assert_called_once_with(
        ["X1:TEST"],
        float(START),
        float(END),
        on_gap=arrakis_client_mod.ONGAP_DEFAULT,
    )
    assert (tsd["X1:TEST"].value[:int(SAMPLE_RATE)] == -1.0).all()


@pytest.mark.requires("arrakis")
def test_fetch_series_statevector(arrakis_client):
    """`fetch_series(..., series_class=StateVector)` returns a `StateVector`."""
    arrakis_client.return_value.fetch.return_value = _arrakis_block(["X1:TEST"])

    sv = io_arrakis.fetch_series(
        "X1:TEST",
        START,
        END,
        series_class=StateVector,
    )

    assert isinstance(sv, StateVector)


# -- get() registry wiring -----------

@pytest.mark.requires("arrakis")
def test_get_arrakis(arrakis_client):
    """`TimeSeries.get(..., source="arrakis")` dispatches to the Arrakis reader."""
    block = _arrakis_block(["X1:TEST"])
    arrakis_client.return_value.fetch.return_value = block

    with mock.patch.dict(os.environ, {"ARRAKIS_SERVER": "test.arrakis:1234"}):
        ts = TimeSeries.get(
            "X1:TEST",
            START,
            END,
            source="arrakis",
        )

    utils.assert_quantity_sub_equal(
        ts,
        TimeSeries.from_arrakis(block["X1:TEST"]),
    )
