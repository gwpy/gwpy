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

"""This module provides time conversion utilities.

The :class:`~astropy.time.core.Time` object from the astropy package
is imported for user convenience, and a GPS time conversion function
is provided.

All other time conversions can be easily completed using the `Time`
object.
"""

from dateutil import parser as dateparser
from astropy.time import Time

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version
__all__ = ['Time', 'gps']


def gps(input, scale='utc', **targs):
    """Convert any input date/time into a GPS float.

    Any input object that can be cast as a
    :class:`~astropy.time.core.Time` (with `str` going through the
    :class:`datetime.datetime`) are acceptable.

    Parameters
    ----------
    input : `datetime.datetime`, `~astropy.time.core.Time`, `str`
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
    ValueError
        if the input cannot be cast as a `~astropy.time.core.Time`.
    """
    if isinstance(input, (unicode, str)):
        input = dateparser.parse(input)
    if not isinstance(input, Time):
        targs.setdefault('copy', True)
        input = Time(input, scale=scale, **targs)
    return input.utc.gps
