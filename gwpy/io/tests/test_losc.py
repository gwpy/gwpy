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

"""Unit tests for :mod:`gwpy.io.losc`
"""

from ssl import SSLError

from six.moves.urllib.error import URLError

import pytest

from .. import losc as io_losc

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def test_fetch_json():
    try:
        jdata = io_losc.fetch_json(
            'https://losc.ligo.org/archive/1126257414/1126261510/json/')
    except (URLError, SSLError) as exc:
        pytest.skip(str(exc))
    assert sorted(list(jdata.keys())) == ['events', 'runs']
    assert jdata['events']['GW150914'] == {
        'DQbits': 7,
        'GPStime': 1126259462.4,
        'INJbits': 5,
        'UTCtime': u'2015-09-14T09:50:45.400000',
        'detectors': [u'L1', u'H1'],
        'frametype': u'%s_HOFT_C02',
    }

    with pytest.raises(ValueError) as exc:
        io_losc.fetch_json(
            'https://losc.ligo.org/archive/1126257414/1126261510/')
    assert str(exc.value).startswith('Failed to parse LOSC JSON')


@pytest.mark.parametrize('segment, detector, strict, result', [
    ((1126257414, 1126261510), 'H1', False,
     ('GW150914', 'O1', 'O1_16KHZ', 'tenyear')),
    ((1126250000, 1126270000), 'H1', False,
     ('O1', 'O1_16KHZ', 'tenyear', 'GW150914')),
    ((1126250000, 1126270000), 'H1', True,
     ('O1', 'O1_16KHZ', 'tenyear',)),
    ((1126250000, 1126270000), 'V1', False,
     ('tenyear',)),
])
def test_find_datasets(segment, detector, strict, result):
    try:
        sets = io_losc.find_datasets(*segment,
                                     detector=detector, strict=strict)
    except (URLError, SSLError) as exc:
        pytest.skip(str(exc))
    assert sets == result


def test_event_gps():
    try:
        gps = io_losc.event_gps('GW170817')
    except (URLError, SSLError) as exc:
        pytest.skip(str(exc))
    assert gps == 1187008882.43
