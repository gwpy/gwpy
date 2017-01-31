# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

"""Extend :mod:`astropy.table` with the `EventTable`
"""

import operator as _operator
from math import ceil

import numpy

from astropy.table import (Table, Column)

from ..io import (reader, writer)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['EventColumn', 'EventTable']

OPERATORS = {'<': _operator.lt, '<=': _operator.le, '=': _operator.eq,
             '>=': _operator.ge, '>': _operator.gt, '==': _operator.is_,
             '!=': _operator.is_not}


class EventColumn(Column):
    """Custom `Column` that allows filtering with segments
    """
    def in_segmentlist(self, segmentlist):
        """Return the index of values lying inside the given segmentlist

        A `~gwpy.segments.Segment` represents a semi-open interval,
        so for any segment `[a, b)`, a value `x` is 'in' the segment if

            a <= x < b

        """
        segmentlist = type(segmentlist)(segmentlist).coalesce()
        idx = self.argsort()
        contains = numpy.zeros(self.shape[0], dtype=bool)
        j = 0
        try:
            a, b = segmentlist[j]
        except IndexError:  # no segments, return all False
            return contains
        i = 0
        while i < contains.shape[0]:
            x = idx[i]
            v = self[x]
            # if before start, move to next value
            if v < a:
                i += 1
                continue
            # if after end, find the next segment and check value again
            if v >= b:
                j += 1
                try:
                    a, b = segmentlist[j]
                    continue
                except IndexError:
                    break
            # otherwise value must be in this segment
            contains[x] = True
            i += 1
        return contains

    def not_in_segmentlist(self, segmentlist):
        """Return the index of values not lying inside the given segmentlist

        See `~EventColumn.in_segmentlist` for more details
        """
        return self.in_segmentlist(~segmentlist)


class EventTable(Table):
    """Container for a table of events
    """
    Column = EventColumn

    # -- i/o ------------------------------------

    read = classmethod(reader(doc="""
        read data into an `eventtable`

        Parameters
        ----------
        source : `str`, `list`, :class:`~glue.lal.Cache`
            file or list of files from which to read events

        format : `str`, optional
            the format of the given source files; if not given, an attempt
            will be made to automatically identify the format

        columns : `list` of `str`, optional
            list of column names ro read; should represent a sub-set of
            all available columns

        .. note::

           Keyword arguments other than those listed here may be required
           depending on the `format`

        Returns
        -------
        table : `EventTable`

        Raises
        ------
        astropy.io.registry.IORegistryError
            if the `format` cannot be automatically identified

        Notes
        -----"""))

    write = writer(doc="""
        write this table to a file

        parameters
        ----------
        target: `str`
            filename for output data file

        format : `str`, optional
            format for output data; if not given, an attempt will be made
            to automatically identify the format based on the `target`
            filename

        **kwargs
            other keyword arguments will be passed directly to the
            underlying writer method for the given format

        raises
        ------
        astropy.io.registry.ioregistryerror
            if the `format` cannot be automatically identified

        notes
        -----""")

    # -- ligolw compatibility -------------------

    def get_column(self, name):
        """Return the `Column` with the given name

        This method is provided only for compatibility with the
        :class:`glue.ligolw.table.Table`.

        Parameters
        ----------
        name : `str`
            the name of the column to return

        Returns
        -------
        column : `astropy.table.Column`

        Raises
        ------
        KeyError
            if no column is found with the given name
        """
        return self[name]

    # -- extensions -----------------------------

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
        times = self[timecolumn]
        if not start:
            start = times.min()
        if not end:
            end = times.max()
        nsamp = int(ceil((end - start) / stride))
        timebins = numpy.arange(nsamp + 1) * stride + start
        # histogram data and return
        out = TimeSeries(
            numpy.histogram(times, bins=timebins)[0] / float(stride),
            t0=start, dt=stride, unit='Hz', name='Event rate')
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
        rates : ~gwpy.timeseries.TimeSeriesDict`
            a dict of (bin, `~gwpy.timeseries.TimeSeries`) pairs describing a
            rate of events per second (Hz) for each of the bins.
        """
        from gwpy.timeseries import TimeSeriesDict

        # work out time boundaries
        times = self[timecolumn]
        if not start:
            start = times.min()
        if not end:
            end = times.max()

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

        coldata = self[column]

        # generate one TimeSeries per bin
        out = TimeSeriesDict()
        for bin_ in bins:
            if isinstance(bin_, tuple):
                keep = (coldata >= bin_[0]) & (coldata < bin_[1])
            else:
                keep = op(coldata, bin_)
            out[bin_] = self[keep].event_rate(stride, start=start, end=end,
                                              timecolumn=timecolumn)
            out[bin_].name = '%s $%s$ %s' % (column, operator, bin_)

        return out

    def plot(self, x, y, *args, **kwargs):
        """Generate an `EventTablePlot` of this `Table`.

        Parameters
        ----------
        x : `str`
            name of column defining centre point on the X-axis

        y : `str`
            name of column defining centre point on the Y-axis

        width : `str`, optional
            name of column defining width of tile

        height : `str`, optional
            name of column defining height of tile

            .. note::

               The ``width`` and ``height`` positional arguments should
               either both be omitted, in which case a scatter plot will
               be drawn, or both given, in which case a collection of
               rectangles will be drawn.

        color : `str`, optional, default:`None`
            name of column by which to color markers

        **kwargs
            any other arguments applicable to the `Plot` constructor, and
            the `Table` plotter.

        Returns
        -------
        plot : `~gwpy.plotter.EventTablePlot`
            new plot for displaying tabular data.

        See Also
        --------
        gwpy.plotter.EventTablePlot
            for more details.
        """
        from gwpy.plotter import EventTablePlot
        return EventTablePlot(self, x, y, *args, **kwargs)

    def hist(self, column, **kwargs):
        """Generate a `HistogramPlot` of this `Table`.

        Parameters
        ----------
        column : `str`
            name of the column over which to histogram data

        **kwargs
            any other arguments applicable to the `HistogramPlot`

        Returns
        -------
        plot : `~gwpy.plotter.HistogramPlot`
            new plot displaying a histogram of this `Table`.
        """
        from gwpy.plotter import HistogramPlot
        return HistogramPlot(self, column, **kwargs)
