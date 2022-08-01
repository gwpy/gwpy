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
from unittest import mock

import pytest

import gwdatafind

from ...testing.errors import (
    pytest_skip_cvmfs_read_error,
    pytest_skip_network_error,
)
from ...testing.utils import (
    TEST_GWF_FILE,
)
from .. import datafind as io_datafind

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

GWDATAFIND_PATH = "gwpy.io.datafind.gwdatafind"
MOCK_ENV = {
    'VIRGODATA': 'tmp',
    'GWDATAFIND_SERVER': 'test:80',
}


def _mock_gwdatafind(func):
    """Decorate a function to use a mocked GWDataFind API.
    """
    @mock.patch.dict("os.environ", MOCK_ENV)
    @mock.patch(
        f"{GWDATAFIND_PATH}.find_types",
        mock.MagicMock(
            return_value=[os.path.basename(TEST_GWF_FILE).split("-")[1]],
        )
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
        'gwpy.io.datafind.iter_channel_names',
        mock.MagicMock(return_value=['L1:LDAS-STRAIN', 'H1:LDAS-STRAIN']),
    )
    @mock.patch(
        'gwpy.io.datafind.num_channels',
        mock.MagicMock(return_value=1),
    )
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def _gwosc_gwdatafind(func):
    """Decorate a function to use the GWOSC GWDataFind server.

    That server returns paths from CVMFS (``/cvmfs/gwosc.osgstorage.org``) so
    we need to add various protections.
    """
    @mock.patch.dict(  # point GWDataFind at GWOSC server
        "os.environ",
        {"GWDATAFIND_SERVER": "datafind.gw-openscience.org:80"},
    )
    @pytest_skip_cvmfs_read_error  # skip CVMFS problems
    @pytest_skip_network_error  # skip network problems
    @pytest.mark.skipif(  # skip missing CVMFS repo
        not os.path.isdir('/cvmfs/gwosc.osgstorage.org/'),
        reason="GWOSC CVMFS repository not available",
    )
    @pytest.mark.cvmfs  # mark test as requiring cvmfs
    @pytest.mark.requires("LDAStools.frameCPP")  # skip if no frameCPP
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


# -- tests --------------------------------------------------------------------

@mock.patch.dict("os.environ", clear=True)
def test_gwdatafind_module_error():
    """Test that _gwdatafind_module() will propagate an exception if
    it can't work out how to find data.
    """
    with pytest.raises(RuntimeError) as exc:
        io_datafind.find_frametype("TEST")
    assert str(exc.value).startswith("unknown datafind configuration")


# -- find_frametype

@_mock_gwdatafind
def test_find_frametype():
    """Test that `find_frametype` works with basic input.
    """
    assert io_datafind.find_frametype(
        'L1:LDAS-STRAIN',
        allow_tape=True,
    ) == 'HW100916'


@_mock_gwdatafind
def test_find_frametype_return_all():
    """Test that the ``return_all` keyword for `find_frametype` returns a list.
    """
    assert io_datafind.find_frametype(
        'L1:LDAS-STRAIN',
        return_all=True,
    ) == ['HW100916']


@_mock_gwdatafind
def test_find_frametype_multiple():
    """Test that `find_frametype` can handle channels as MIMO.
    """
    assert io_datafind.find_frametype(
        ['H1:LDAS-STRAIN', 'L1:LDAS-STRAIN'],
        allow_tape=True,
    ) == {
        'H1:LDAS-STRAIN': 'HW100916',
        'L1:LDAS-STRAIN': 'HW100916',
    }


@_mock_gwdatafind
def test_find_frametype_error_not_found():
    """Test that `find_frametype` raises the right error for a missing channel.
    """
    # test missing channel raises sensible error
    with pytest.raises(ValueError) as exc:
        io_datafind.find_frametype('X1:TEST', allow_tape=True)
    assert str(exc.value) == (
        'Cannot locate the following channel(s) '
        'in any known frametype:\n    X1:TEST')


@_mock_gwdatafind
def test_find_frametype_error_bad_channel():
    """Test that `find_frametype` raises the right error when the channel
    name isn't parsable.
    """
    with pytest.raises(ValueError) as exc:
        io_datafind.find_frametype('bad channel name')
    assert str(exc.value) == ('Cannot parse interferometer prefix '
                              'from channel name \'bad channel name\','
                              ' cannot proceed with find()')


@_mock_gwdatafind
def test_find_frametype_error_files_on_tape():
    """Test that `find_frametype` raises the right error when the only
    discovered data are on tape, and we asked for not on tape.
    """
    # check that allow_tape errors get handled properly
    with mock.patch('gwpy.io.datafind.on_tape', return_value=True):
        with pytest.raises(ValueError) as exc:
            io_datafind.find_frametype('X1:TEST', allow_tape=False)
        assert '[files on tape have not been checked' in str(exc.value)


# -- find_best_frametype

@_mock_gwdatafind
def test_find_best_frametype():
    """Test that `find_best_frametype` works in general.
    """
    assert io_datafind.find_best_frametype(
        'L1:LDAS-STRAIN',
        968654552,
        968654553,
    ) == 'HW100916'


@_gwosc_gwdatafind
def test_find_best_frametype_with_gaps():
    """Test that `find_best_frametype` works across gaps.

    It tries to find something that covers at least some of the gap.
    """
    assert io_datafind.find_best_frametype(
        "L1:GWOSC-4KHZ_R1_STRAIN",
        1187733504,
        1187733504 + 4097,  # one second too long
    ) == "L1_GWOSC_O2_4KHZ_R1"


@_gwosc_gwdatafind
def test_find_best_frametype_with_gaps_multiple():
    """Test that `find_best_frametype` works across gaps with multiple
    channels.
    """
    assert io_datafind.find_best_frametype(
        ("L1:GWOSC-4KHZ_R1_STRAIN", "L1:GWOSC-4KHZ_R1_DQMASK"),
        1187733504,
        1187733504 + 4097,  # one second too long
    ) == {
        "L1:GWOSC-4KHZ_R1_STRAIN": "L1_GWOSC_O2_4KHZ_R1",
        "L1:GWOSC-4KHZ_R1_DQMASK": "L1_GWOSC_O2_4KHZ_R1",
    }


@pytest.mark.requires("LDAStools.frameCPP")
@pytest_skip_network_error
@pytest.mark.skipif(
    "GWDATAFIND_SERVER" not in os.environ,
    reason='No GWDataFind server configured on this host',
)
@pytest.mark.parametrize('channel, expected', [
    ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend', 'H1_T'),
    ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend', 'H1_M'),
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
    with pytest.raises(RuntimeError) as exc:
        io_datafind.find_latest("X1", "MISSING")
    assert str(exc.value) == "no files found for X-MISSING"


# -- find_types

@mock.patch.dict("os.environ", MOCK_ENV)
def test_find_types():
    """Check that `find_types` works with default arguments.
    """
    types = ["a", "b", "c"]
    with mock.patch(f"{GWDATAFIND_PATH}.find_types", return_value=types):
        assert io_datafind.find_types(
            "X",
        ) == sorted(types)


@mock.patch.dict("os.environ", MOCK_ENV)
def test_find_types_priority():
    """Check that `find_types` prioritises trends properly.
    """
    types = ["L1_R", "L1_T", "L1_M"]
    with mock.patch(f"{GWDATAFIND_PATH}.find_types", return_value=types):
        assert io_datafind.find_types(
            "X",
            trend="m-trend",
        ) == ["L1_M", "L1_R", "L1_T"]


# -- utilities --------------

def test_on_tape_false():
    """Check `on_tape` works with a normal file.
    """
    assert io_datafind.on_tape(TEST_GWF_FILE) is False


def test_on_tape_true():
    """Test that `on_tape` returns `True` for a file with zero blocks.
    """
    with mock.patch("os.stat") as os_stat:
        os_stat.return_value.st_blocks = 0
        assert io_datafind.on_tape(TEST_GWF_FILE) is True


def test_on_tape_windows():
    """Test that `on_tape` always returns `False` on Windows.
    """
    class _Stat():
        def __init__(self, path):
            pass

        @property
        def st_blocks(self):
            raise AttributeError("st_blocks")

    with mock.patch("os.stat", _Stat):
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
    """Test that `_type_priority` works for various cases.
    """
    assert io_datafind._type_priority(ifo, ftype, trend=trend)[0] == priority
