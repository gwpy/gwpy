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
from decimal import Decimal

from dateutil import parser as dateparser

from astropy.units import Quantity

from . import (Time, LIGOTimeGPS)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['tconvert', 'to_gps', 'from_gps']


def tconvert(gpsordate='now'):
    """Convert GPS times to ISO-format date-times and vice-versa.

    Parameters
    ----------
    gpsordate : `float`, `astropy.time.Time`, `datetime.datetime`, ...
        input gps or date to convert, many input types are supported

    Returns
    -------
    date : `datetime.datetime` or `LIGOTimeGPS`
        converted gps or date

    Notes
    -----
    If the input object is a `float` or `LIGOTimeGPS`, it will get
    converted from GPS format into a `datetime.datetime`, otherwise
    the input will be converted into `LIGOTimeGPS`.

    Examples
    --------
    Integers and floats are automatically converted from GPS to
    `datetime.datetime`:

    >>> from gwpy.time import tconvert
    >>> tconvert(0)
    datetime.datetime(1980, 1, 6, 0, 0)
    >>> tconvert(1126259462.3910)
    datetime.datetime(2015, 9, 14, 9, 50, 45, 391000)

    while strings are automatically converted to `~gwpy.time.LIGOTimeGPS`:

    >>> to_gps('Sep 14 2015 09:50:45.391')
    LIGOTimeGPS(1126259462, 391000000)

    Additionally, a few special-case words as supported, which all return
    `~gwpy.time.LIGOTimeGPS`:

    >>> tconvert('now')
    >>> tconvert('today')
    >>> tconvert('tomorrow')
    >>> tconvert('yesterday')
    """
    # convert from GPS into datetime
    try:
        float(gpsordate)  # if we can 'float' it, then its probably a GPS time
    except (TypeError, ValueError):
        return to_gps(gpsordate)
    return from_gps(gpsordate)


def to_gps(t, *args, **kwargs):
    """Convert any input date/time into a `LIGOTimeGPS`.

    Any input object that can be cast as a `~astropy.time.Time`
    (with `str` going through the `datetime.datetime`) are acceptable.

    Parameters
    ----------
    t : `float`, `~datetime.datetime`, `~astropy.time.Time`, `str`
        the input time, any object that can be converted into a
        `LIGOTimeGPS`, `~astropy.time.Time`, or `~datetime.datetime`,
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

    Examples
    --------
    >>> to_gps('Jan 1 2017')
    LIGOTimeGPS(1167264018, 0)
    >>> to_gps('Sep 14 2015 09:50:45.391')
    LIGOTimeGPS(1126259462, 391000000)

    >>> import datetime
    >>> to_gps(datetime.datetime(2017, 1, 1))
    LIGOTimeGPS(1167264018, 0)

    >>> from astropy.time import Time
    >>> to_gps(Time(57754, format='mjd'))
    LIGOTimeGPS(1167264018, 0)
    """
    # allow Time conversion to override type-checking
    if args or kwargs:
        return Time(t, *args, **kwargs).utc.gps
    # if lal.LIGOTimeGPS, just return it
    if isinstance(t, LIGOTimeGPS):
        return t
    # if Decimal, cast to LIGOTimeGPS and return
    if isinstance(t, Decimal):
        return LIGOTimeGPS(str(t))

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
            gps = to_gps(UTCToGPS(t.timetuple()))
            if hasattr(t, 'microsecond'):
                return gps + t.microsecond * 1e-6
            return gps
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
    gps : `LIGOTimeGPS`, `int`, `float`
        GPS time to convert

    Returns
    -------
    datetime : `datetime.datetime`
        ISO-format datetime equivalent of input GPS time

    Examples
    --------
    >>> from_gps(1167264018)
    datetime.datetime(2017, 1, 1, 0, 0)
    >>> from_gps(1126259462.3910)
    datetime.datetime(2015, 9, 14, 9, 50, 45, 391000)
    """
    try:
        gps = LIGOTimeGPS(gps)
    except (ValueError, TypeError):
        gps = LIGOTimeGPS(float(gps))
    try:
        from lal import GPSToUTC
    except ImportError:
        date = Time(gps.gpsSeconds, gps.gpsNanoSeconds * 1e-9,
                    format='gps', scale='utc').datetime
    else:
        date = datetime.datetime(*GPSToUTC(gps.gpsSeconds)[:6])
        date += datetime.timedelta(seconds=gps.gpsNanoSeconds * 1e-9)
    if float(gps).is_integer():
        return date.replace(microsecond=0)
    return date


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
    # if datetime format has zero microseconds, force int(gps) to remove
    # floating point precision errors from gps
    if type(dt) is datetime.date or (  # pylint: disable=unidiomatic-typecheck
            isinstance(dt, datetime.datetime) and not dt.microsecond):
        return LIGOTimeGPS(int(gps))
    # use repr() to remove hidden floating point precision problems
    return LIGOTimeGPS(repr(gps))


def str_to_datetime(datestr):
    """Convert a `str` representing a datetime into a `datetime.datetime`.

    Parameters
    ----------
    datestr : `str`
        date-like string parseable by :meth:`dateutil.parser.parse`, or
        one of the following special cases

            - ``'now'`` : second precision for current time
            - ``'today'``
            - ``'tomorrow'``
            - ``'yesterday'``

    Returns
    -------
    date : `datetime.datetime`
        `datetime.datetime` version of the input ``datestr``

    Raises
    ------
    TypeError
        if ``datestr`` cannot be parsed by :meth:`dateutil.parser.parse`.
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
        except (ValueError, TypeError) as exc:
            exc.args = ("Cannot parse date string %r: %s"
                        % (datestr, exc.args[0]),)
            raise
    return date
