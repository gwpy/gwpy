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

from .. import version
from . import (Time, LIGOTimeGPS)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version
__all__ = ['tconvert', 'to_gps', 'from_gps']


def tconvert(gpsordate='now'):
    """Convert GPS times to ISO-format date-times and vice-versa.

    Parameters
    ----------
    gpsordate : `float`, `LIGOTimeGPS`, `Time`, `datetime.datetime`, ...
        input gps or date to convert

    Returns
    -------
    date : :class:`datetime.datetime` or :class:`~gwpy.time.LIGOTimeGPS`
        converted gps or date

    Notes
    -----
    If the input object is a `float` or :class:`~gwpy.time.LIGOTimeGPS`,
    it will get converted from GPS format into a
    :class:`datetime.datetime`, otherwise the input will be converted
    into :class:`~gwpy.time.LIGOTimeGPS`.
    """
    # convert from GPS into datetime
    if isinstance(gpsordate, (float, LIGOTimeGPS)):
        return from_gps(gpsordate)
    else:
        return to_gps(gpsordate)


def to_gps(t, **tparams):
    """Convert any input date/time into a :class:`~gwpy.time.LIGOTimeGPS`.

    Any input object that can be cast as a
    :class:`~astropy.time.core.Time` (with `str` going through the
    :class:`datetime.datetime`) are acceptable.

    Parameters
    ----------
    t : `datetime.datetime`, `~astropy.time.core.Time`, `str`
        the input time, any object that can be converted into a
        :class:`~astropy.time.core.Time` (using
        :class:`datetime.datetime` as an intermediary as needed) is
        acceptable.
    scale : `str`, optional, default: ``'utc'``
        time-scale of input value.
    **targs
        other keyword arguments to pass to `~astropy.time.core.Time`.

    Returns
    -------
    gps : `float`
        the number of GPS seconds (non-integer) since the start of the
        epoch (January 6 1980).

    Raises
    ------
    TypeError
        if a `str` input cannot be parsed as a `datetime.datetime`.
    ValueError
        if the input cannot be cast as a `~astropy.time.core.Time` or
        `LIGOTimeGPS`.
    """
    # allow Time conversion to override type-checking
    if tparams:
        return Time(t, **tparams).utc.gps
    # if lal.LIGOTimeGPS
    if hasattr(t, 'gpsSeconds'):
        return LIGOTimeGPS(t.gpsSeconds, t.gpsNanoSeconds)
    # or convert str into datetime.datetime
    if isinstance(t, str):
        t = str_to_datetime(t)
    # and then into Time
    if isinstance(t, datetime.date):
        if not isinstance(t, datetime.datetime):
            t = datetime.datetime.combine(t, datetime.time.min)
        t = Time(t, scale='utc')
    # and then into LIGOTimeGPS
    if isinstance(t, Time):
        return time_to_gps(t)
    # if all else fails...
    return LIGOTimeGPS(t)


def from_gps(gps):
    """Convert a GPS time into a :class:`datetime.datetime`.

    Parameters
    ----------
    gps : :class:`~gwpy.time.LIGOTimeGPS`
        GPS time to convert

    Returns
    -------
    datetime : :class:`datetime.datetime`
        ISO-format datetime equivalent of input GPS time
    """
    try:
        gps = LIGOTimeGPS(gps)
    except (ValueError, TypeError):
        gps = LIGOTimeGPS(float(gps))
    dt = Time(gps.seconds, gps.nanoseconds * 1e-9,
              format='gps', scale='utc').datetime
    if float(gps).is_integer():
        return dt.replace(microsecond=0)
    else:
        return dt


def time_to_gps(t):
    """Convert a `Time` into `LIGOTimeGPS`.

    Parameters
    ----------
    t : :class:`~astropy.time.core.Time`
        formatted `Time` object to convert

    Returns
    -------
    gps : :class:`~gwpy.time.LIGOTimeGPS`
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
    date : :class:`datetime.datetime`
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
            e.args = ("Cannot parse date string %r: %s" % (datestr, e.args[0]),)
            raise
    return date
