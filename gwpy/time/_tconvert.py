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

"""tconvert: a utility to convert to and from GPS times.

This method is inspired by the original tconvert utility, written by
Peter Shawhan.
"""

import datetime

from dateutil import parser as dateparser

from astropy.units import Quantity

from . import (Time, LIGOTimeGPS)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['tconvert', 'to_gps', 'from_gps']


def tconvert(gpsordate='now'):
    """Convert GPS times to ISO-format date-times and vice-versa.

    Parameters
    ----------
    gpsordate : `float`, `LIGOTimeGPS`, `Time`, `datetime.datetime`, ...
        input gps or date to convert

    Returns
    -------
    date : `datetime.datetime` or `LIGOTimeGPS`
        converted gps or date

    Notes
    -----
    If the input object is a `float` or `LIGOTimeGPS`,
    it will get converted from GPS format into a
    `datetime.datetime`, otherwise the input will be converted
    into `LIGOTimeGPS`.
    """
    # convert from GPS into datetime
    try:
        gps = LIGOTimeGPS(gpsordate)
    except (ValueError, TypeError):
        if hasattr(gpsordate, 'gpsSeconds'):
            return from_gps(gpsordate)
        else:
            return to_gps(gpsordate)
    else:
        return from_gps(gps)


def to_gps(t, *args, **kwargs):
    """Convert any input date/time into a `LIGOTimeGPS`.

    Any input object that can be cast as a
    `~astropy.time.Time` (with `str` going through the
    `datetime.datetime`) are acceptable.

    Parameters
    ----------
    t : `float`, `datetime.datetime`, `~astropy.time.Time`, `str`
        the input time, any object that can be converted into a
        `LIGOTimeGPS`, `~astropy.time.Time`, or `datetime.datetime`,
        is acceptable.
    *args, **kwargs
        other arguments to pass to pass to `~astropy.time.Time` if given

    Returns
    -------
    gps : `LIGOTimeGPS`
        the number of GPS seconds (non-integer) since the start of the
        epoch (January 6 1980).

    Raises
    ------
    TypeError
        if a `str` input cannot be parsed as a `datetime.datetime`.
    ValueError
        if the input cannot be cast as a `~astropy.time.Time` or
        `LIGOTimeGPS`.
    """
    # allow Time conversion to override type-checking
    if args or kwargs:
        return Time(t, *args, **kwargs).utc.gps
    # if lal.LIGOTimeGPS
    if hasattr(t, 'gpsSeconds'):
        return LIGOTimeGPS(t.gpsSeconds, t.gpsNanoSeconds)
    # or convert numeric string to float (e.g. '123.456')
    try:
        t = float(t)
    except (TypeError, ValueError):
        pass
    # or convert str into datetime.datetime
    if isinstance(t, str):
        t = str_to_datetime(t)
    # or convert tuple into datetime.datetime
    elif isinstance(t, (tuple, list)):
        t = datetime.datetime(*t)
    # and then into lal.LIGOTimeGPS or Time
    if isinstance(t, datetime.date):
        # try and use LAL, it's more reliable (possibly)
        try:
            from lal import UTCToGPS
        except ImportError:
            if not isinstance(t, datetime.datetime):
                t = datetime.datetime.combine(t, datetime.time.min)
            t = Time(t, scale='utc')
        else:
            return to_gps(UTCToGPS(t.timetuple()))
    # and then into LIGOTimeGPS
    if isinstance(t, Time):
        return time_to_gps(t)
    # extract quantity to a float in seconds
    if isinstance(t, Quantity):
        t = t.to('second').value
    # if all else fails...
    return LIGOTimeGPS(t)


def from_gps(gps):
    """Convert a GPS time into a `datetime.datetime`.

    Parameters
    ----------
    gps : `LIGOTimeGPS`
        GPS time to convert

    Returns
    -------
    datetime : `datetime.datetime`
        ISO-format datetime equivalent of input GPS time
    """
    try:
        gps = LIGOTimeGPS(gps)
    except (ValueError, TypeError):
        gps = LIGOTimeGPS(float(gps))
    try:
        from lal import GPSToUTC
    except ImportError:
        dt = Time(gps.seconds, gps.nanoseconds * 1e-9,
                  format='gps', scale='utc').datetime
    else:
        dt = datetime.datetime(*GPSToUTC(gps.seconds)[:6])
        dt += datetime.timedelta(seconds=gps.nanoseconds * 1e-9)
    if float(gps).is_integer():
        return dt.replace(microsecond=0)
    else:
        return dt


def time_to_gps(t):
    """Convert a `Time` into `LIGOTimeGPS`.

    Parameters
    ----------
    t : `~astropy.time.Time`
        formatted `Time` object to convert

    Returns
    -------
    gps : `LIGOTimeGPS`
        Nano-second precision `LIGOTimeGPS` time
    """
    t = t.utc
    dt = t.datetime
    gps = t.gps
    if ((isinstance(dt, datetime.datetime) and not dt.microsecond) or
            type(dt) is datetime.date):
        gps = int(gps)
    return LIGOTimeGPS(gps)


def str_to_datetime(datestr):
    """Convert a `str` representing a datetime into a `datetime.datetime`.

    Parameters
    ----------
    datestr : `str`
        date-like string parseable by :meth:`dateutil.parser.parse`, or
        one of the following special cases

        - 'now' : second precision for current time
        - 'today'
        - 'tomorrow'
        - 'yesterday'

    Returns
    -------
    date : `datetime.datetime`
        `datetime.datetime` version of the input ``datestr``

    Raises
    ------
    TypeError
        if ``datestr`` cannot be parsed by :meth:`dateutil.parser.parse`
    """
    datestr = str(datestr).lower()
    if datestr == 'now':
        date = datetime.datetime.utcnow().replace(microsecond=0)
    elif datestr == 'today':
        date = datetime.date.today()
    elif datestr == 'tomorrow':
        today = datetime.date.today()
        date = today + datetime.timedelta(days=1)
    elif datestr == "yesterday":
        today = datetime.date.today()
        date = today - datetime.timedelta(days=1)
    else:
        try:
            date = dateparser.parse(datestr)
        except TypeError as e:
            e.args = ("Cannot parse date string %r: %s"
                      % (datestr, e.args[0]),)
            raise
    return date
