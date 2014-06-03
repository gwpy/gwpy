# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""Common tools for time-axis array.
"""

import warnings
from math import (ceil, floor)

import numpy

from astropy import units
from astropy.time import Time

from .. import version
from ..data import Series
from ..segments import Segment

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


def is_contiguous(self, other):
    """Check whether other is contiguous with self.
    """
    self.is_compatible(other)
    if abs(float(self.span[1] - other.span[0])) < 1e-5:
        return 1
    elif abs(float(other.span[1] - self.span[0])) < 1e-5:
        return -1
    else:
        return 0


def append(self, other, gap='raise', inplace=True, pad=0.0, resize=True):
    """Connect another `TimeSeries` onto the end of the current one.

    Parameters
    ----------
    other : `TimeSeries`
        the second data set to connect to this one
    gap : `str`, optional, default: ``'raise'``
        action to perform if there's a gap between the other series
        and this one. One of

            - ``'raise'`` - raise an `Exception`
                - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

    inplace : `bool`, optional, default: `True`
        perform operation in-place, modifying current `TimeSeries,
        otherwise copy data and return new `TimeSeries`
    pad : `float`, optional, default: ``0.0``
        value with which to pad discontiguous `TimeSeries`

    Returns
    -------
    series : `TimeSeries`
        time-series containing joined data sets
    """
    # check metadata
    self.is_compatible(other)
    # make copy if needed
    if inplace:
        new = self
    else:
        new = self.copy()
    # fill gap
    if new.is_contiguous(other) != 1:
        if gap == 'pad':
            ngap = (other.span[0] - new.span[1]) // new.dt.value
            if ngap < 1:
                raise ValueError("Cannot append TimeSeries that starts "
                                 "before this one.")
            gapshape = list(new.shape)
            gapshape[0] = int(ngap)
            padding = numpy.ones(gapshape).view(new.__class__) * pad
            padding.epoch = new.span[1]
            padding.dt = new.dt
            padding.unit = new.unit
            new.append(padding, inplace=True, resize=resize)
        elif gap == 'ignore':
            pass
        elif new.span[0] < other.span[0] < new.span[1]:
            raise ValueError("Cannot append overlapping TimeSeries")
        else:
            raise ValueError("Cannot append discontiguous TimeSeries\n"
                             "    TimeSeries 1 span: %s\n"
                             "    TimeSeries 2 span: %s"
                             % (self.span, other.span))
    # check empty other
    if not other.size:
        return new
    # resize first
    if resize:
        s = list(new.shape)
        s[0] = new.shape[0] + other.shape[0]
        try:
            new.resize(s, refcheck=False)
        except ValueError as e:
            if 'resize only works on single-segment arrays' in str(e):
                new = new.copy()
                new.resize(s, refcheck=False)
            else:
                raise
    else:
        new.data[:-other.shape[0]] = new.data[other.shape[0]:]
    new[-other.shape[0]:] = other.data
    try:
        if isinstance(self, Series):
            times = new._index
        else:
            timees = new._xindex
    except AttributeError:
        if not resize:
            new.x0 = new.x0.value + other.shape[0] * new.dx.value
    else:
        if resize:
            new.times.resize(s, refcheck=False)
        else:
            new.times[:-other.shape[0]] = new.times[other.shape[0]:]
        new.times[-other.shape[0]:] = other.times.data
        new.epoch = new.times[0]
    return new


def prepend(self, other, gap='raise', inplace=True, pad=0.0):
    """Connect another `TimeSeries` onto the start of the current one.

    Parameters
    ----------
    other : `TimeSeries`
        the second data set to connect to this one
    gap : `str`, optional, default: ``'raise'``
        action to perform if there's a gap between the other series
        and this one. One of

            - ``'raise'`` - raise an `Exception`
            - ``'ignore'`` - remove gap and join data
            - ``'pad'`` - pad gap with zeros

    inplace : `bool`, optional, default: `True`
        perform operation in-place, modifying current `TimeSeries,
        otherwise copy data and return new `TimeSeries`
    pad : `float`, optional, default: ``0.0``
        value with which to pad discontiguous `TimeSeries`

    Returns
    -------
    series : `TimeSeries`
        time-series containing joined data sets
    """
    # check metadata
    self.is_compatible(other)
    # make copy if needed
    if inplace:
        new = self
    else:
        new = self.copy()
    # fill gap
    if new.is_contiguous(other) != -1:
        if gap == 'pad':
            ngap = (new.span[0]-other.span[1]) // new.dt.value
            if ngap < 1:
                raise ValueError("Cannot prepend TimeSeries that starts "
                                 "after this one.")
            gapshape = list(new.shape)
            gapshape[0] = ngap
            padding = numpy.ones(gapshape).view(new.__class__) * pad
            padding.epoch = other.span[1]
            padding.dt = new.dt
            padding.unit = new.unit
            new.prepend(padding, inplace=True)
        elif gap == 'ignore':
            pass
        elif other.span[0] < new.span[0] < other.span[1]:
            raise ValueError("Cannot prepend overlapping TimeSeries")
        else:
            raise ValueError("Cannot prepend discontiguous TimeSeries")
    # resize first
    N = new.shape[0]
    s = list(new.shape)
    s[0] = new.shape[0] + other.shape[0]
    new.resize(s, refcheck=False)
    new[-N:] = new.data[:N]
    new[:other.shape[0]] = other.data
    return new


def update(self, other, inplace=True):
    """Update this `TimeSeries` by appending new data from an other
    and dropping the same amount of data off the start.

    """
    return self.append(other, inplace=inplace, resize=False)


def crop(self, start=None, end=None, copy=False):
    """Crop this `TimeSeries` to the given GPS ``[start, end)``
    `Segment`.

    Parameters
    ----------
    start : `Time`, `float`
        GPS start time to crop `TimeSeries` at left
    end : `Time`, `float`
        GPS end time to crop `TimeSeries` at right

    Returns
    -------
    timeseries : `TimeSeries`
        A new `TimeSeries` with the same metadata but different GPS
        span

    Notes
    -----
    If either ``start`` or ``end`` are outside of the original
    `TimeSeries` span, warnings will be printed and the limits will
    be restricted to the :attr:`TimeSeries.span`
    """
    # check type
    if isinstance(start, Time):
        start = start.gps
    if isinstance(end, Time):
        end = end.gps
    # pin early starts to time-series start
    if start == self.span[0]:
        start = None
    elif start is not None and start < self.span[0]:
        warnings.warn('TimeSeries.crop given GPS start earlier than '
                      'start time of the input TimeSeries. Crop will '
                      'begin when the TimeSeries actually starts.')
        start = None
    # pin late ends to time-series end
    if end == self.span[1]:
        end = None
    if start is not None and end > self.span[1]:
        warnings.warn('TimeSeries.crop given GPS end later than '
                      'end time of the input TimeSeries. Crop will '
                      'end when the TimeSeries actually ends.')
        end = None
    # find start index
    if start is None:
        idx0 = None
    else:
        idx0 = float(start - self.span[0]) // self.dt.value
    # find end index
    if end is None:
        idx1 = None
    else:
        idx1 = float(end - self.span[0]) // self.dt.value
        if idx1 >= self.size:
            idx1 = None
    # crop
    if copy:
        return self[idx0:idx1].copy()
    else:
        return self[idx0:idx1]
