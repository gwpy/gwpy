# Copyright (C) Louisiana State University (2014-2017)
#               Cardiff University (2017-)
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

from __future__ import annotations

import datetime
from collections.abc import Callable
from decimal import Decimal
from numbers import Number
from typing import (
    SupportsFloat,
    cast,
)

from astropy.time import Time
from astropy.units import Quantity
from dateparser import parse as dateparser_parse

from . import LIGOTimeGPS

GpsConvertible = SupportsFloat | datetime.date | str

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "tconvert",
    "to_gps",
    "from_gps",
]


def tconvert(
    gpsordate: GpsConvertible = "now",
) -> datetime.date | LIGOTimeGPS:
    """Convert GPS times to ISO-format date-times and vice-versa.

    Parameters
    ----------
    gpsordate : `float`, `astropy.time.Time`, `datetime.datetime`, ...
        Input gps or date to convert, many input types are supported.

    Returns
    -------
    date : `datetime.datetime` or `LIGOTimeGPS`
        Converted gps or date.

    Notes
    -----
    If the input object is a `float` or `LIGOTimeGPS`, it will get
    converted from GPS format into a `datetime.datetime`, otherwise
    the input will be converted into `LIGOTimeGPS`.

    .. warning::

       This method cannot convert exact leap seconds to
       `datetime.datetime`, that object doesn't support it,
       so you should consider using `astropy.time.Time` directly.

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

    >>> to_gps("Sep 14 2015 09:50:45.391")
    LIGOTimeGPS(1126259462, 391000000)

    Additionally, a few special-case words as supported, which all return
    `~gwpy.time.LIGOTimeGPS`:

    >>> tconvert("now")
    >>> tconvert("today")
    >>> tconvert("tomorrow")
    >>> tconvert("yesterday")
    """
    # convert from GPS into datetime
    try:
        # if we can 'float' it, then its probably a GPS time
        float(gpsordate)  # type: ignore[arg-type]
    except (
        TypeError,
        ValueError,
    ):
        return to_gps(gpsordate)
    return from_gps(cast("SupportsFloat", gpsordate))


def to_gps(
    t: GpsConvertible,
    *args,
    **kwargs,
) -> LIGOTimeGPS:
    """Convert any input date/time into a `LIGOTimeGPS`.

    Any input object that can be cast as a `~astropy.time.Time`
    (with `str` going through the `datetime.datetime`) are acceptable.

    Parameters
    ----------
    t : `float`, `~datetime.datetime`, `~astropy.time.Time`, `str`
        The input time, any object that can be converted into a
        `LIGOTimeGPS`, `~astropy.time.Time`, or `~datetime.datetime`,
        is acceptable.

    args, kwargs
        Other arguments to pass to pass to `~astropy.time.Time` if given.

    Returns
    -------
    gps : `LIGOTimeGPS`
        The number of GPS seconds (non-integer) since the start of the
        epoch (January 6 1980).

    Raises
    ------
    TypeError
        If a `str` input cannot be parsed as a `datetime.datetime`.
    ValueError
        If the input cannot be cast as a `~astropy.time.Time` or
        `LIGOTimeGPS`.

    Examples
    --------
    >>> to_gps("Jan 1 2017")
    LIGOTimeGPS(1167264018, 0)
    >>> to_gps("Sep 14 2015 09:50:45.391")
    LIGOTimeGPS(1126259462, 391000000)

    >>> import datetime
    >>> to_gps(datetime.datetime(2017, 1, 1))
    LIGOTimeGPS(1167264018, 0)

    >>> from astropy.time import Time
    >>> to_gps(Time(57754, format="mjd"))
    LIGOTimeGPS(1167264018, 0)
    """
    # -- convert input to Time, or something we can pass to LIGOTimeGPS

    if isinstance(t, str):
        try:  # if str represents a number, leave it for LIGOTimeGPS to handle
            float(t)
        except ValueError:  # str -> datetime.date
            t = _str_to_datetime(t)

    # tuple -> datetime.date
    if isinstance(t, tuple | list):
        t = datetime.datetime(*t)

    # datetime.datetime -> Time
    if isinstance(t, datetime.date):
        t = _datetime_to_time(t)

    # Quantity -> float
    if isinstance(t, Quantity):
        t = t.to("second").value

    # Number/Decimal -> str
    if isinstance(t, Decimal | Number):
        t = str(t)

    # -- convert to LIGOTimeGPS

    if isinstance(t, Time):
        return _time_to_gps(t, *args, **kwargs)
    try:
        return LIGOTimeGPS(t)  # type: ignore[call-arg,call-overload]
    except (TypeError, ValueError):
        return LIGOTimeGPS(float(t))  # type: ignore[arg-type]


def from_gps(
    gps: SupportsFloat,
) -> datetime.datetime:
    """Convert a GPS time into a `datetime.datetime`.

    Parameters
    ----------
    gps : `LIGOTimeGPS`, `int`, `float`
        GPS time to convert

    Returns
    -------
    datetime : `datetime.datetime`
        ISO-format datetime equivalent of input GPS time

    Notes
    -----
    .. warning::

       This method cannot convert exact leap seconds to
       `datetime.datetime`, that object doesn't support it,
       so you should consider using `astropy.time.Time` directly.

    Examples
    --------
    >>> from_gps(1167264018)
    datetime.datetime(2017, 1, 1, 0, 0)
    >>> from_gps(1126259462.3910)
    datetime.datetime(2015, 9, 14, 9, 50, 45, 391000)
    """
    try:
        ltgps = LIGOTimeGPS(gps)  # type: ignore[call-arg,call-overload]
    except (
        ValueError,
        TypeError,
        RuntimeError,
    ):
        ltgps = LIGOTimeGPS(float(gps))
    sec, nano = ltgps.gpsSeconds, ltgps.gpsNanoSeconds
    try:
        date = Time(sec, format="gps", scale="utc").datetime
    except ValueError as exc:
        if "within a leap second" in str(exc):
            exc.args = (
                "cannot represent leap second using datetime.datetime, "
                "consider using "
                f"astropy.time.Time({gps}, format=\"gps\", scale=\"utc\") "
                "directly",
            )
        raise
    return date + datetime.timedelta(microseconds=nano * 1e-3)


# -- utilities ----------------------------------------------------------------
# special case strings

def _now() -> datetime.datetime:
    try:
        now = datetime.datetime.now(datetime.UTC)
    except AttributeError:  # python < 3.11
        now = datetime.datetime.utcnow()
    return now.replace(microsecond=0)


def _today() -> datetime.date:
    return datetime.date.today()


def _today_delta(**delta) -> datetime.date:
    return _today() + datetime.timedelta(**delta)


def _tomorrow() -> datetime.date:
    return _today_delta(days=1)


def _yesterday() -> datetime.date:
    return _today_delta(days=-1)


DATE_STRINGS: dict[str, Callable] = {
    "now": _now,
    "today": _today,
    "tomorrow": _tomorrow,
    "yesterday": _yesterday,
}


def _str_to_datetime(
    datestr: str,
) -> datetime.datetime:
    """Convert `str` to `datetime.datetime`."""
    # try known string
    try:
        return DATE_STRINGS[str(datestr).lower()]()
    except KeyError:  # any other string
        pass

    result = dateparser_parse(
        datestr,
        settings={
            "TIMEZONE": "UTC",
            "RETURN_AS_TIMEZONE_AWARE": True,
            "TO_TIMEZONE": "UTC",
            "PREFER_DATES_FROM": "current_period",
        }
    )
    if result is None:
        raise ValueError(f"failed to parse '{datestr}' as datetime")
    return result


def _datetime_to_time(
    dtm: datetime.date,
) -> Time:
    """Convert `datetime.date` to `astropy.time.Time`."""
    # astropy.time.Time requires datetime.datetime
    if not isinstance(dtm, datetime.datetime):
        dtm = datetime.datetime.combine(dtm, datetime.time.min)
    return Time(dtm, scale="utc")


def _time_to_gps(
    time: Time,
) -> LIGOTimeGPS:
    """Convert a `Time` into `LIGOTimeGPS`.

    This method uses `datetime.datetime` underneath, which restricts
    to microsecond precision by design. This should probably be fixed...

    Parameters
    ----------
    time : `~astropy.time.Time`
        Formatted `Time` object to convert.

    Returns
    -------
    gps : `LIGOTimeGPS`
        Nano-second precision `LIGOTimeGPS` time.
    """
    time = time.utc
    date = time.datetime
    micro = date.microsecond if isinstance(date, datetime.datetime) else 0
    return LIGOTimeGPS(int(time.gps), int(micro * 1e3))
