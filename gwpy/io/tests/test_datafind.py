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

"""Unit tests for :mod:`gwpy.io.datafind`."""

import os
from pathlib import Path
from unittest import mock

import gwdatafind
import pytest

from ...testing.errors import pytest_skip_flaky_network
from ...testing.utils import TEST_GWF_FILE
from .. import datafind as io_datafind

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

GWDATAFIND_PATH = "gwpy.io.datafind.gwdatafind"
MOCK_ENV = {
    "VIRGODATA": "tmp",
    "GWDATAFIND_SERVER": "test:80",
}


def _mock_gwdatafind(func):
    """Decorate a function to use a mocked GWDataFind API."""
    @mock.patch.dict("os.environ", MOCK_ENV)
    @mock.patch(
        f"{GWDATAFIND_PATH}.find_types",
        mock.MagicMock(
            return_value=[Path(TEST_GWF_FILE).name.split("-")[1]],
        ),
    )
    @mock.patch(
        f"{GWDATAFIND_PATH}.find_urls",
        mock.MagicMock(return_value=[TEST_GWF_FILE]),
    )
    @mock.patch(
        f"{GWDATAFIND_PATH}.find_latest",
        mock.MagicMock(return_value=[TEST_GWF_FILE]),
    )
    @mock.patch(
        "gwpy.io.datafind.iter_channel_names",
        mock.MagicMock(return_value=["L1:LDAS-STRAIN", "H1:LDAS-STRAIN"]),
    )
    @mock.patch(
        "gwpy.io.datafind.num_channels",
        mock.MagicMock(return_value=1),
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# -- tests ---------------------------

@mock.patch.dict("os.environ", clear=True)
def test_gwdatafind_module_error():
    """Test `_gwdatafind_module()` error propagation."""
    with pytest.raises(
        RuntimeError,
        match=r"^unknown datafind configuration",
    ):
        io_datafind.find_frametype("X1:TEST")


# -- find_frametype

@_mock_gwdatafind
def test_find_frametype():
    """Test `find_frametype()`."""
    assert io_datafind.find_frametype(
        "L1:LDAS-STRAIN",
        allow_tape=True,
    ) == "HW100916"


@_mock_gwdatafind
def test_find_frametype_return_all():
    """Test `find_frametype(..., return_all=True)`."""
    assert io_datafind.find_frametype(
        "L1:LDAS-STRAIN",
        return_all=True,
    ) == ["HW100916"]


@_mock_gwdatafind
def test_find_frametype_multiple():
    """Test `find_frametype()` with list inputs."""
    assert io_datafind.find_frametype(
        ["H1:LDAS-STRAIN", "L1:LDAS-STRAIN"],
        allow_tape=True,
    ) == {
        "H1:LDAS-STRAIN": "HW100916",
        "L1:LDAS-STRAIN": "HW100916",
    }


@_mock_gwdatafind
def test_find_frametype_error_not_found():
    """Test `find_frametype()` error handing for missing channels."""
    with pytest.raises(
        ValueError,
        match=(
            r"^Cannot locate the following channel\(s\) "
            "in any known frametype:\n    X1:TEST$"
        ),
    ):
        io_datafind.find_frametype("X1:TEST", allow_tape=True)


@_mock_gwdatafind
def test_find_frametype_error_bad_channel():
    """Test `find_frametype()` error handling for bad channel names."""
    with pytest.raises(
        ValueError,
        match=(
            "^Cannot parse interferometer prefix from channel name "
            r"'bad channel name', cannot proceed with find\(\)$"
        ),
    ):
        io_datafind.find_frametype("bad channel name")


@_mock_gwdatafind
def test_find_frametype_error_files_on_tape():
    """Test `find_frametype()` error handling for ``allow_tape=False``."""
    patch = mock.patch("gwpy.io.datafind.on_tape", return_value=True)
    raises = pytest.raises(
        ValueError,
        match=r"\[files on tape have not been checked",
    )
    with patch, raises:
        io_datafind.find_frametype("X1:TEST", allow_tape=False)


# -- find_best_frametype

@_mock_gwdatafind
def test_find_best_frametype():
    """Test that `find_best_frametype` works in general."""
    assert io_datafind.find_best_frametype(
        "L1:LDAS-STRAIN",
        968654552,
        968654553,
    ) == "HW100916"


@pytest.mark.requires("lalframe")
@pytest_skip_flaky_network
@pytest.mark.skipif(
    "GWDATAFIND_SERVER" not in os.environ,
    reason="No GWDataFind server configured on this host",
)
@pytest.mark.parametrize(("channel", "expected"), [
    ("H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend", "H1_T"),
    ("H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend", "H1_M"),
])
def test_find_best_frametype_ligo_trend(channel, expected):
    """Test that `find_best_frametype` correctly matches trends.

    Currently the LIGO trend data are only available from a restricted server,
    so this test only works on LIGO computing resources.
    """
    # check that this server knows about trends and we can authenticate
    try:
        assert expected in gwdatafind.find_types("H"), (
            f"gwdatafind server doesn't know about {expected}"
        )
    except (
        AssertionError,
        RuntimeError,
    ) as exc:
        pytest.skip(str(exc))

    assert io_datafind.find_best_frametype(
        channel,
        1262276680,  # GW200105 -4s
        1262276688,  # GW200105 +4s
    ) == expected


# -- find_latest

@_mock_gwdatafind
def test_find_latest():
    """Test `find_latest()`."""
    assert io_datafind.find_latest(
        "HLV",
        "HW100916",
    ) == TEST_GWF_FILE


@mock.patch(
    f"{GWDATAFIND_PATH}.find_latest",
    mock.MagicMock(side_effect=IndexError),
)
@_mock_gwdatafind
def test_find_latest_error():
    """Test `find_latest()` error handling."""
    with pytest.raises(
        RuntimeError,
        match=r"^no files found for X-MISSING$",
    ):
        io_datafind.find_latest("X1", "MISSING")


# -- find_types

@mock.patch.dict("os.environ", MOCK_ENV)
def test_find_types():
    """Check that `find_types` works with default arguments."""
    types = ["a", "b", "c"]
    with mock.patch(f"{GWDATAFIND_PATH}.find_types", return_value=types):
        assert io_datafind.find_types(
            "X",
        ) == sorted(types)


@mock.patch.dict("os.environ", MOCK_ENV)
def test_find_types_priority():
    """Check that `find_types` prioritises trends properly."""
    types = ["L1_R", "L1_T", "L1_M"]
    with mock.patch(f"{GWDATAFIND_PATH}.find_types", return_value=types):
        assert io_datafind.find_types(
            "X",
            trend="m-trend",
        ) == ["L1_M", "L1_R", "L1_T"]


# -- utilities --------------

def test_on_tape_false():
    """Check `on_tape` works with a normal file."""
    assert io_datafind.on_tape(TEST_GWF_FILE) is False


@mock.patch("pathlib.Path.stat")
def test_on_tape_true(stat):
    """Test that `on_tape` returns `True` for a file with zero blocks."""
    stat.return_value.st_blocks = 0
    assert io_datafind.on_tape(TEST_GWF_FILE) is True


@pytest.mark.parametrize(("ftype", "trend", "priority"), [
    pytest.param("L1_HOFT_C00", None, 1, id="hoft"),
    pytest.param("H1_HOFT_C02_T1700406_v3", None, 1, id="cleaned"),
    pytest.param("H1_M", "m-trend", 0, id="m-trend"),
    pytest.param("K1_T", "s-trend", 0, id="s-trend"),
    pytest.param("K1_R", "s-trend", 5, id="raw@s-trend"),
    pytest.param("K1_M", None, 10, id="m-trend@raw"),
    pytest.param("K1_C", None, 6, id="commissioning"),
    pytest.param("SOMETHING_GRB051103", None, 10, id="lowpriority"),
    pytest.param("something else", None, 5, id="other"),
])
def test_type_priority(ftype, trend, priority):
    """Test that `_type_priority` works for various cases."""
    assert io_datafind._type_priority(
        ftype,
        trend=trend,
    )[0] == priority


# -- end-to-end tests with real data -

@pytest_skip_flaky_network
@pytest.mark.requires("lalframe", "requests_pelican")
@mock.patch.dict("os.environ", {"GWPY_CACHE": "false"})
def test_find_frametype_gwosc_osdf(caplog):
    """Test `find_frametype` with GWOSC data over OSDF."""
    caplog.set_level("DEBUG", logger="gwpy.io.datafind")
    try:
        assert io_datafind.find_frametype(
            "H1:GWOSC-4KHZ_R1_STRAIN",
            host="datafind.gwosc.org",
            urltype="osdf",
            gpstime=1269362688,
            frametype_match="_4KHZ_",
        ) == "H1_GWOSC_O3b_4KHZ_R1"
    except ValueError:
        # Astropy < 7.1 doesn't handle TimeoutError well, so check for
        # network-related issues in the logs and skip the test if so.
        for record in caplog.records:
            if "Failed to download file" in (msg := record.getMessage()):
                pytest.skip(msg)
        raise
