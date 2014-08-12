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

"""Methods to calculate a rate TimeSeries from a LIGO_LW Table.
"""

import operator as _operator
from math import ceil

import numpy

from .. import version
from .utils import (EVENT_TABLES, get_table_column)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version
__all__ = ['event_rate', 'binned_event_rates']

OPERATORS = {'<': _operator.lt, '<=': _operator.le, '=': _operator.eq,
             '>=': _operator.ge, '>': _operator.gt, '==': _operator.is_,
             '!=': _operator.is_not}


def event_rate(self, stride, start=None, end=None, timecolumn='time'):
    """Calculate the rate `~gwpy.timeseries.TimeSeries` for this `Table`.

    Parameters
    ----------
    stride : `float`
        size (seconds) of each time bin
    start : `float`, :class:`~gwpy.time.LIGOTimeGPS`, optional
        GPS start epoch of rate :class:`~gwpy.timeseries.TimeSeries`
    end : `float`, :class:`~gwpy.time.LIGOTimeGPS`, optional
        GPS end time of rate :class:`~gwpy.timeseries.TimeSeries`.
        This value will be rounded up to the nearest sample if needed.
    timecolumn : `str`, optional, default: ``time``
        name of time-column to use when binning events

    Returns
    -------
    rate : :class:`~gwpy.timeseries.TimeSeries`
        a `TimeSeries` of events per second (Hz)
    """
    from gwpy.timeseries import TimeSeries
    # get time data
    times = get_table_column(self, timecolumn)
    # generate time bins
    if not start:
        start = times.min()
    if not end:
        end = times.max()
    nsamp = int(ceil((end - start) / stride))
    timebins = numpy.arange(nsamp + 1) * stride + start
    # histogram data and return
    out = TimeSeries(numpy.histogram(times, bins=timebins)[0] / float(stride),
                     epoch=start, sample_rate=1/float(stride), unit='Hz',
                     name='Event rate')
    return out


def binned_event_rates(self, stride, column, bins, operator='>=',
                       start=None, end=None, timecolumn='time'):
    """Calculate an event rate `~gwpy.timeseries.TimeSeriesDict` over
    a number of bins.

    Parameters
    ----------
    stride : `float`
        size (seconds) of each time bin
    column : `str`
        name of column by which to bin.
    bins : `list`
        a list of `tuples <tuple>` marking containing bins, or a list of
        `floats <float>` defining bin edges against which an math operation
        is performed for each event.
    operator : `str`, `callable`
        one of:

        - ``'<'``, ``'<='``, ``'>'``, ``'>='``, ``'=='``, ``'!='``,
          for a standard mathematical operation,
        - ``'in'`` to use the list of bins as containing bin edges, or
        - a callable function that takes compares an event value
          against the bin value and returns a boolean.

        .. note::

           If ``bins`` is given as a list of tuples, this argument
           is ignored.

    start : `float`, :class:`~gwpy.time.LIGOTimeGPS`, optional
        GPS start epoch of rate `~gwpy.timeseries.TimeSeries`.
    end : `float`, `~gwpy.time.LIGOTimeGPS`, optional
        GPS end time of rate `~gwpy.timeseries.TimeSeries`.
        This value will be rounded up to the nearest sample if needed.
    timecolumn : `str`, optional, default: ``time``
        name of time-column to use when binning events

    Returns
    -------
    rates : :class:`~gwpy.timeseries.TimeSeriesDict`
        a dict of (bin, `~gwpy.timeseries.TimeSeries`) pairs describing a
        rate of events per second (Hz) for each of the bins.
    """
    from gwpy.timeseries import (TimeSeries, TimeSeriesDict)
    from gwpy.plotter.table import get_column_string
    # get time data
    times = get_table_column(self, timecolumn)

    # get channel
    try:
        channel = self[0].channel
    except (IndexError, AttributeError):
        channel = None

    # generate time bins
    if not start:
        start = times.min()
    if not end:
        end = times.max()
    nsamp = int(ceil((end - start) / stride))
    timebins = numpy.arange(nsamp + 1) * stride + start
    # generate column bins
    if not bins:
        bins = [(-numpy.inf, numpy.inf)]
    if operator == 'in' and not isinstance(bins[0], tuple):
        bins2 = []
        for i, bin_ in enumerate(bins[:-1]):
            bins2.append((bin_, bins[i+1]))
        bins = bins2
    elif isinstance(operator, (unicode, str)):
        op = OPERATORS[operator]
    else:
        op = operator
    coldata = get_table_column(self, column)
    colstr = get_column_string(column)
    # generate one TimeSeries per bin
    out = TimeSeriesDict()
    for bin_ in bins:
        if isinstance(bin_, tuple):
            bintimes = times[(coldata >= bin_[0]) & (coldata < bin_[1])]
        else:
            bintimes = times[op(coldata, bin_)]
        out[bin_] = TimeSeries(
            numpy.histogram(bintimes, bins=timebins)[0] / float(stride),
            epoch=start, sample_rate=1/float(stride), unit='Hz',
            name='%s $%s$ %s' % (colstr, operator, bin_), channel=channel)
    return out


# attach methods to lsctables
for table in EVENT_TABLES:
    table.event_rate = event_rate
    table.binned_event_rates = binned_event_rates
