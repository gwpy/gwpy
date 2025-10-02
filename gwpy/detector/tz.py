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

"""Timezone utilities for GWOs."""

import datetime

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# local time-zone for ground-based laser interferometers
TIMEZONE: dict[str, str] = {
    "C1": "US/Pacific",
    "G1": "Europe/Berlin",
    "H1": "US/Pacific",
    "I1": "Asia/Kolkata",
    "K1": "Japan",
    "L1": "US/Central",
    "V1": "Europe/Rome",
}


def get_timezone(ifo: str) -> str:
    """Return the timezone for the given interferometer prefix.

    Parameters
    ----------
    ifo : `str`
        Prefix of interferometer, e.g. ``'X1'``

    Returns
    -------
    timezone : `str`
        The name of the timezone for ``ifo``.

    Raises
    ------
    ValueError
        If ``ifo`` is not recognised.

    Examples
    --------
    >>> get_timezone("G1")
    'Europe/Berlin'
    """
    try:
        return TIMEZONE[ifo]
    except KeyError as exc:
        msg = f"Unrecognised ifo: '{ifo}'"
        raise ValueError(msg) from exc


def get_timezone_offset(ifo: str, dt: datetime.datetime | None = None) -> float:
    """Return the offset in seconds between UTC and the given interferometer.

    Parameters
    ----------
    ifo : `str`
        Prefix of interferometer, e.g. ``'X1'``

    dt : `datetime.datetime`, optional
        The time at which to calculate the offset.
        Default is `datetime.datetime.now`.

    Returns
    -------
    offset : `float`
        The offset in seconds between the timezone of the interferometer
        and UTC.
    """
    import pytz
    if dt is None:
        dt = datetime.datetime.now(tz=datetime.UTC)
    if dt.tzinfo is None:
        dt = dt.astimezone(datetime.UTC)
    tz = pytz.timezone(get_timezone(ifo))
    return dt.astimezone(tz).utcoffset().total_seconds()
