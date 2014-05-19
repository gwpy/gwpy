# coding=utf-8
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

"""tconvert: a utility to convert to and from GPS times.

This method is inspired by the original tconvert utility, written by
Peter Shawhan.
"""

import datetime

from dateutil import parser as dateparser

from .. import version
from . import (Time, LIGOTimeGPS)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def tconvert(gpsordate='now'):
    """Convert GPS times to :class:`datetime.datetime` and vice-versa.
    """
    # convert from GPS into datetime
    if isinstance(gpsordate, (float, LIGOTimeGPS)):
        if float(gpsordate).is_integer():
            gpsordate = int(gpsordate)
        return Time(gpsordate, format='gps', scale='utc').datetime
    # convert str to datetime
    elif isinstance(gpsordate, str):
        return str_to_gps(gpsordate)
    # convert datetime to GPS
    if isinstance(gpsordate, datetime.date):
        gpsordate = Time(gpsordate, scale='utc')
    # convert Time to GPS
    if isinstance(gpsordate, Time) and not gpsordate.format == 'gps':
        return time_to_gps(gpsordate)
    try:
        return Time(gpsordate, format='gps', scale='utc').datetime
    except Exception as e:
        try:
            return LIGOTimeGPS(Time(gpsordate).utc.gps)
        except:
            raise e


def time_to_gps(t):
    """Convert a `Time` into `LIGOTimeGPS`.
    """
    t = t.utc
    dt = t.datetime
    gps = t.gps
    if ((isinstance(dt, datetime.datetime) and not dt.microsecond) or
            type(dt) is datetime.date):
        gps = int(gps)
    return LIGOTimeGPS(gps)


def str_to_gps(datestr):
    """Convert a `str` representing a datetime into `LIGOTimeGPS`.
    """
    datestr = datestr.lower()
    if datestr == 'now':
        date = datetime.datetime.now()
    elif datestr == 'today':
        date = datetime.date.today()
    elif datestr == 'tomorrow':
        today = datetime.date.today()
        date = today + datetime.timedelta(days=1)
    elif datestr == "yesterday":
        today = datetime.date.today()
        date = today - datetime.timedelta(days=1)
    else:
        date = dateparser.parser(datestr)
    return time_to_gps(Time(date, scale='utc'))
