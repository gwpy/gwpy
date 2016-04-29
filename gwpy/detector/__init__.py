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

"""This module defines the :class:`~gwpy.detector.Channel`.
"""

# This module used to define the `LaserInterferometer` class, but it was
# removed # pre-release because it never got used, or implemented properly.

import datetime

from ..utils.deps import with_import
from . import units
from .channel import *
from .io import *

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

# local time-zone for ground-based laser interferometers
TIMEZONE = {
    'C1': 'US/Pacific',
    'G1': 'Europe/Berlin',
    'H1': 'US/Pacific',
    'L1': 'US/Central',
    'V1': 'Europe/Rome',
}


def get_timezone(ifo):
    try:
        return TIMEZONE[ifo]
    except KeyError as e:
        e.args = ('No time-zone information for %r detector' % ifo,)
        raise


@with_import('pytz')
def get_timezone_offset(ifo, dt=None):
    dt = dt or datetime.datetime.now()
    offset = pytz.timezone(get_timezone(ifo)).utcoffset(dt)
    return offset.days * 86400 + offset.seconds + offset.microseconds * 1e-6
