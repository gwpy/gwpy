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

from math import ceil

from six import string_types

import numpy

from astropy.table import (Table, Column, vstack)
from astropy.io.registry import write as io_write

from ..io.mp import read_multi as io_read_multi
from .filter import (filter_table, parse_operator)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = ['EventColumn', 'EventTable']


class EventColumn(Column):
    """Custom `Column` that allows filtering with segments
    """
    def in_segmentlist(self, segmentlist):
        """Return the index of values lying inside the given segmentlist

        A `~gwpy.segments.Segment` represents a semi-open interval,
        so for any segment `[a, b)`, a value `x` is 'in' the segment if

            a <= x < b

        """
        from .filters import in_segmentlist
        return in_segmentlist(self, segmentlist)

    def not_in_segmentlist(self, segmentlist):
        """Return the index of values not lying inside the given segmentlist

        See `~EventColumn.in_segmentlist` for more details
        """
        return self.in_segmentlist(~segmentlist)


class EventTable(Table):
    """A container for a table of events

    This differs from the basic `~astropy.table.Table` in two ways

    - GW-specific file formats are registered to use with
      `EventTable.read` and `EventTable.write`
    - columns of this table are of the `EventColumn` type, which provides
      methods for filtering based on a `~gwpy.segments.SegmentList` (not
      specifically time segments)

    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `EventTable`
    """
    Column = EventColumn

    # -- i/o ------------------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):  # pylint: disable=arguments-differ
        """Read data into an `EventTable`

        Parameters
        ----------
        source : `str`, `list`, :class:`~glue.lal.Cache`
            file or list of files from which to read events

        *args
            other positional arguments will be passed directly to the
            underlying reader method for the given format

        format : `str`, optional
            the format of the given source files; if not given, an attempt
            will be made to automatically identify the format

        selection : `str`, or `list` of `str`
            one or more column filters with which to downselect the
            returned table rows as they as read, e.g. ``'snr > 5'``;
            multiple selections should be connected by ' && ', or given as
            a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
            ``['snr > 5', 'frequency < 1000']``

        nproc : `int`, optional, default: 1
            number of CPUs to use for parallel reading of multiple files

        verbose : `bool`, optional
            print a progress bar showing read status, default: `False`

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
        -----"""
        # astropy's ASCII formats don't support on-the-fly selection, so
        # we pop the selection argument out here
        if str(kwargs.get('format')).startswith('ascii'):
            selection = kwargs.pop('selection', [])
            if isinstance(selection, string_types):
                selection = [selection]
        else:
            selection = []

        # read the table
        tab = io_read_multi(vstack, cls, source, *args, **kwargs)

        # apply the selection if required:
        if selection:
            tab = tab.filter(*selection)

        # and return
        return tab

    def write(self, target, *args, **kwargs):  # pylint: disable=arguments-differ
        """Write this table to a file

        Parameters
        ----------
        target: `str`
            filename for output data file

        *args
            other positional arguments will be passed directly to the
            underlying writer method for the given format

        format : `str`, optional
            format for output data; if not given, an attempt will be made
            to automatically identify the format based on the `target`
            filename

        **kwargs
            other keyword arguments will be passed directly to the
            underlying writer method for the given format

        Raises
        ------
        astropy.io.registry.IORegistryError
            if the `format` cannot be automatically identified

        Notes
        -----"""
        return io_write(self, target, *args, **kwargs)

    @classmethod
    def fetch(cls, format_, *args, **kwargs):
        """Fetch a table of events from a database

        Parameters
        ----------
        format : `str`, `~sqlalchemy.engine.Engine`
            the format of the remote data, see _Notes_ for a list of
            registered formats, OR an SQL database `Engine` object

        *args
            all other positional arguments are specific to the
            data format, see below for basic usage

        columns : `list` of `str`, optional
            the columns to fetch from the database table, defaults to all

        selection : `str`, or `list` of `str`, optional
            one or more column filters with which to downselect the
            returned table rows as they as read, e.g. ``'snr > 5'``;
            multiple selections should be connected by ' && ', or given as
            a `list`, e.g. ``'snr > 5 && frequency < 1000'`` or
            ``['snr > 5', 'frequency < 1000']``

        **kwargs
            all other positional arguments are specific to the
            data format, see the online documentation for more details


        Returns
        -------
        table : `EventTable`
            a table of events recovered from the remote database

        Examples
        --------
        >>> from gwpy.table import EventTable

        To download a table of all blip glitches from the Gravity Spy database:

        >>> EventTable.fetch('gravityspy', 'glitches', selection='Label=Blip')

        To download a table from any SQL-type server

        >>> from sqlalchemy.engine import create_engine
        >>> engine = create_engine(...)
        >>> EventTable.fetch(engine, 'mytable')

        Notes
        -----"""
        # handle open database engine
        try:
            from sqlalchemy.engine import Engine
        except ImportError:
            pass
        else:
            if isinstance(format_, Engine):
                from .io.sql import fetch
                return cls(fetch(format_, *args, **kwargs))

        # standard registered fetch
        from .io.fetch import get_fetcher
        fetcher = get_fetcher(format_, cls)
        return fetcher(*args, **kwargs)

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

        start : `float`, `~gwpy.time.LIGOTimeGPS`, optional
            GPS start epoch of rate `~gwpy.timeseries.TimeSeries`

        end : `float`, `~gwpy.time.LIGOTimeGPS`, optional
            GPS end time of rate `~gwpy.timeseries.TimeSeries`.
            This value will be rounded up to the nearest sample if needed.

        timecolumn : `str`, optional, default: ``time``
            name of time-column to use when binning events

        Returns
        -------
        rate : `~gwpy.timeseries.TimeSeries`
            a `TimeSeries` of events per second (Hz)
        """
        from gwpy.timeseries import TimeSeries
        times = self[timecolumn]
        if times.dtype is numpy.dtype('O'):  # cast to ufunc-compatible type
            times = times.astype('float64', copy=False)
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

        start : `float`, `~gwpy.time.LIGOTimeGPS`, optional
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
        if not start:
            start = self[timecolumn].min()
        if not end:
            end = self[timecolumn].max()

        # generate column bins
        if not bins:
            bins = [(-numpy.inf, numpy.inf)]
        if operator == 'in' and not isinstance(bins[0], tuple):
            bins = [(bin_, bins[i+1]) for i, bin_ in enumerate(bins[:-1])]
        elif isinstance(operator, string_types):
            op_func = parse_operator(operator)
        else:
            op_func = operator

        coldata = self[column]

        # generate one TimeSeries per bin
        out = TimeSeriesDict()
        for bin_ in bins:
            if isinstance(bin_, tuple):
                keep = (coldata >= bin_[0]) & (coldata < bin_[1])
            else:
                keep = op_func(coldata, bin_)
            out[bin_] = self[keep].event_rate(stride, start=start, end=end,
                                              timecolumn=timecolumn)
            out[bin_].name = ' '.join((column, str(operator), str(bin_)))

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
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plotter.EventTableAxes.plot_table
            for documentation of keyword arguments used to display the table
            (calls out to :meth:`~matplotlib.axes.Axes.scatter`)
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

    def filter(self, *column_filters):
        """Apply one or more column slice filters to this `EventTable`

        Multiple column filters can be given, and will be applied
        concurrently

        Parameters
        ----------
        column_filter : `str`, `tuple`
            a column slice filter definition, e.g. ``'snr > 10``, or
            a filter tuple definition, e.g. ``('snr', <my_func>, <arg>)``

        Notes
        -----
        See :ref:`gwpy-table-filter` for more details on using filter tuples

        Returns
        -------
        table : `EventTable`
            a new table with only those rows matching the filters

        Examples
        --------
        To filter an existing `EventTable` (``table``) to include only
        rows with ``snr`` greater than `10`, and ``frequency`` less than
        `1000`:

        >>> table.filter('snr>10', 'frequency<1000')

        Custom operations can be defined using filter tuple definitions:

        >>> from gwpy.table.filters import in_segmentlist
        >>> filter(my_table, ('time', in_segmentlist, segs))
        """
        return filter_table(self, *column_filters)
