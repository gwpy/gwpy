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

from ...testing.utils import (
    TEST_GWF_FILE,
    skip_missing_dependency,
)
from .. import datafind as io_datafind

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

GWDATAFIND_PATH = "gwpy.io.datafind.gwdatafind"
MOCK_ENV = {
    'VIRGODATA': 'tmp',
    'LIGO_DATAFIND_SERVER': 'test:80',
}


def _mock_gwdatafind(func):
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


# -- tests --------------------------------------------------------------------

@_mock_gwdatafind
def test_find_frametype():
    # simple test
    assert io_datafind.find_frametype(
        'L1:LDAS-STRAIN',
        allow_tape=True,
    ) == 'HW100916'


@_mock_gwdatafind
def test_find_frametype_return_all():
    assert io_datafind.find_frametype(
        'L1:LDAS-STRAIN',
        return_all=True,
    ) == ['HW100916']


@_mock_gwdatafind
def test_find_frametype_multiple():
    # test multiple channels
    assert io_datafind.find_frametype(
        ['H1:LDAS-STRAIN'],
        allow_tape=True,
    ) == {'H1:LDAS-STRAIN': 'HW100916'}


@_mock_gwdatafind
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


@_mock_gwdatafind
def test_find_best_frametype():
    assert io_datafind.find_best_frametype(
        'L1:LDAS-STRAIN',
        968654552,
        968654553,
    ) == 'HW100916'


# -- utilities --------------

@skip_missing_dependency('LDAStools.frameCPP')
@pytest.mark.skipif(
    (
        "GWDATAFIND_SERVER" not in os.environ
        or "LIGO_DATAFIND_SERVER" not in os.environ
    ),
    reason='No GWDataFind server configured on this host',
)
@pytest.mark.parametrize('channel, expected', [
    ('H1:GDS-CALIB_STRAIN', {'H1_HOFT_C00', 'H1_ER_C00_L1'}),
    ('L1:IMC-PWR_IN_OUT_DQ', {'L1_R'}),
    ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,s-trend', {'H1_T'}),
    ('H1:ISI-GND_STS_ITMY_X_BLRMS_30M_100M.mean,m-trend', {'H1_M'}),
])
def test_find_best_frametype_ligo(channel, expected):
    try:
        ft = io_datafind.find_best_frametype(
            channel,
            1262276680,  # GW200105 -4s
            1262276688,  # GW200105 +4s
        )
    except ValueError as exc:  # pragma: no-cover
        if str(exc).lower().startswith('cannot locate'):
            pytest.skip(str(exc))
        raise
    except RuntimeError as exc:  # pragma: no-cover
        if "credential" in str(exc):
            pytest.skip(str(exc))
        raise
    assert ft in expected


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


def test_on_tape_false():
    """Check `on_tape` works with a normal file.
    """
    assert io_datafind.on_tape(TEST_GWF_FILE) is False


def test_on_tape_true():
    with mock.patch("os.stat") as os_stat:
        os_stat.return_value.st_blocks = 0
        assert io_datafind.on_tape(TEST_GWF_FILE) is True


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
