# -*- coding: utf-8 -*-
# Copyright (C) Louisiana State University (2017)
#               Cardiff University (2017-2022)
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

import warnings
from functools import wraps
from operator import attrgetter
from math import ceil

import numpy

from gwosc.api import DEFAULT_URL as DEFAULT_GWOSC_URL

from astropy.table import (Table, vstack)
from astropy.io import registry

from ..io.mp import read_multi as io_read_multi
from ..time import gps_types
from .filter import (filter_table, parse_operator)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'

TIME_LIKE_COLUMN_NAMES = [
    "time",  # standard
    "gps",  # GWOSC catalogues
    "peakGPS",  # gravityspy
]


# -- utilities ----------------------------------------------------------------

def inherit_io_registrations(cls):
    parent = cls.__mro__[1]
    for row in registry.get_formats(data_class=parent):
        name = row["Format"]
        # read
        if row["Read"].lower() == "yes":
            registry.register_reader(
                name,
                cls,
                registry.get_reader(name, parent),
                force=False,
            )
        # write
        if row["Write"].lower() == "yes":
            registry.register_writer(
                name,
                cls,
                registry.get_writer(name, parent),
                force=False,
            )
        # identify
        if row["Auto-identify"].lower() == "yes":
            registry.register_identifier(
                name,
                cls,
                registry._identifiers[(name, parent)],
                force=False,
            )
    return cls


def _rates_preprocess(func):
    @wraps(func)
    def wrapped_func(self, *args, **kwargs):
        timecolumn = kwargs.get('timecolumn')
        start = kwargs.get('start')
        end = kwargs.get('end')

        # get timecolumn if we are going to need it
        if (
            (timecolumn is None and (start is None or end is None))
            or not self.colnames
        ):
            try:
                kwargs['timecolumn'] = self._get_time_column()
            except ValueError as exc:
                exc.args = ('{0}, please give `timecolumn` '
                            'keyword'.format(exc.args[0]),)
                raise
        # otherwise use anything (it doesn't matter)
        kwargs.setdefault('timecolumn', self.colnames[0])

        # set start and end
        times = self[kwargs['timecolumn']]
        if start is None:
            kwargs['start'] = times.min()
        if end is None:
            kwargs['end'] = times.max()

        return func(self, *args, **kwargs)
    return wrapped_func


# -- Table --------------------------------------------------------------------

@inherit_io_registrations
class EventTable(Table):
    """A container for a table of events.

    This object expands the default :class:`~astropy.table.Table`
    with extra read/write formats, and methods to perform filtering,
    rate calculations, and visualisation.

    See also
    --------
    astropy.table.Table
        for details on parameters for creating an `EventTable`
    """
    # -- utilities ------------------------------

    def _is_time_column(self, name):
        """Return `True` if a column in this table represents 'time'

        This method checks the name of the column against a hardcoded list
        of time-like names, then checks the `dtype` of the column (or the
        first element in the column) against a hardcoded list of time-like
        dtypes (`gwpy.time.gps_types`).
        """
        # if the name looks like a time column, accept that
        if name.lower() in TIME_LIKE_COLUMN_NAMES:
            return True

        # if the dtype of this column looks right, accept that
        if self[name].dtype in gps_types:
            return True
        try:
            return isinstance(self[name][0], gps_types)
        except IndexError:
            return False

    def _get_time_column(self):
        """Return the name of the 'time' column in this table.

        This method tries the following:

        - look for a single column named 'time', 'gps', or 'peakGPS'
        - look for a single column with a GPS type (e.g. `LIGOTimeGPS`)

        So, its not foolproof.

        Raises a `ValueError` if either 0 or multiple matches are found.
        """
        matches = list(filter(self._is_time_column, self.columns))
        try:
            time, = matches
        except ValueError:
            tcolnames = ", or ".join(
                ", ".join(map(repr, TIME_LIKE_COLUMN_NAMES)).rsplit(", ", 1),
            )
            msg = (
                "cannot identify time column for table, no columns "
                "named {}, or with a GPS dtype".format(tcolnames)
            )
            if len(matches) > 1:
                msg = msg.replace("no columns", "multiple columns")
            raise ValueError(msg)
        return time

    # -- i/o ------------------------------------

    @classmethod
    def read(cls, source, *args, **kwargs):  # pylint: disable=arguments-differ
        """Read data into an `EventTable`

        Parameters
        ----------
        source : `str`, `list`
            Source of data, any of the following:

            - `str` path of single data file,
            - `str` path of LAL-format cache file,
            - `list` of paths.

        *args
            other positional arguments will be passed directly to the
            underlying reader method for the given format

        format : `str`, optional
            the format of the given source files; if not given, an attempt
            will be made to automatically identify the format

        columns : `list` of `str`, optional
            the list of column names to read

        selection : `str`, or `list` of `str`, optional
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
        IndexError
            if ``source`` is an empty list

        Notes
        -----"""
        return io_read_multi(vstack, cls, source, *args, **kwargs)

    def write(self, target, *args, **kwargs):
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
        return registry.write(self, target, *args, **kwargs)

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

        >>> EventTable.fetch(
        ...     'gravityspy',
        ...     'glitches',
        ...     selection=['ml_label=Blip', 'ml_confidence>0.9'],
        ... )

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
        out = fetcher(*args, **kwargs)
        if not isinstance(out, cls):
            if issubclass(cls, type(out)):
                try:
                    return cls(out)
                except Exception as exc:
                    exc.args = (
                        "could not convert fetch() output to {0}: {1}".format(
                            cls.__name__, str(exc),
                        ),
                    )
                    raise
            raise TypeError(
                "fetch() should return a {0} instance".format(cls.__name__),
            )
        return out

    @classmethod
    def fetch_open_data(cls, catalog, columns=None, selection=None,
                        host=DEFAULT_GWOSC_URL, **kwargs):
        """Fetch events from an open-data catalogue hosted by GWOSC.

        Parameters
        ----------
        catalog : `str`
            the name of the catalog to fetch, e.g. ``'GWTC-1-confident'``

        columns : `list` of `str`, optional
            the list of column names to read

        selection : `str`, or `list` of `str`, optional
            one or more column filters with which to downselect the
            returned events as they as read, e.g. ``'mass1 < 30'``;
            multiple selections should be connected by ' && ', or given as
            a `list`, e.g. ``'mchirp < 3 && distance < 500'`` or
            ``['mchirp < 3', 'distance < 500']``

        host : `str`, optional
            the open-data host to use
        """
        from .io.losc import fetch_catalog
        tab = fetch_catalog(catalog, columns=columns, selection=selection,
                            host=host, **kwargs)
        if type(tab) is cls:  # don't copy unless we need to
            return tab
        return cls(tab)

    # -- ligolw compatibility -------------------

    def get_column(self, name):
        """Return the `Column` with the given name

        This method is provided only for compatibility with the
        :class:`ligo.lw.table.Table`.

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

    @_rates_preprocess
    def event_rate(self, stride, start=None, end=None, timecolumn=None):
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

        timecolumn : `str`, optional
            name of time-column to use when binning events, attempts
            are made to guess this

        Returns
        -------
        rate : `~gwpy.timeseries.TimeSeries`
            a `TimeSeries` of events per second (Hz)

        Raises
        ------
        ValueError
            if the ``timecolumn`` cannot be guessed from the table contents
        """
        # NOTE: decorator sets timecolumn, start, end to non-None values
        from gwpy.timeseries import TimeSeries
        times = self[timecolumn]
        if times.dtype.name == 'object':  # cast to ufuncable type
            times = times.astype('longdouble', copy=False)
        nsamp = int(ceil((end - start) / stride))
        timebins = numpy.arange(nsamp + 1) * stride + start
        # create histogram
        return TimeSeries(
            numpy.histogram(times, bins=timebins)[0] / float(stride),
            t0=start, dt=stride, unit='Hz', name='Event rate')

    @_rates_preprocess
    def binned_event_rates(self, stride, column, bins, operator='>=',
                           start=None, end=None, timecolumn=None):
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
        # NOTE: decorator sets timecolumn, start, end to non-None values

        from gwpy.timeseries import TimeSeriesDict

        # generate column bins
        if not bins:
            bins = [(-numpy.inf, numpy.inf)]
        if operator == 'in' and not isinstance(bins[0], tuple):
            bins = [(bin_, bins[i+1]) for i, bin_ in enumerate(bins[:-1])]
        elif isinstance(operator, str):
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

    def plot(self, *args, **kwargs):
        """DEPRECATED, use `EventTable.scatter`
        """
        warnings.warn('{0}.plot was renamed {0}.scatter and will be removed '
                      'in an upcoming release'.format(type(self).__name__),
                      DeprecationWarning)
        return self.scatter(*args, **kwargs)

    def scatter(self, x, y, **kwargs):
        """Make a scatter plot of column ``x`` vs column ``y``.

        Parameters
        ----------
        x : `str`
            name of column defining centre point on the X-axis

        y : `str`
            name of column defining centre point on the Y-axis

        color : `str`, optional, default:`None`
            name of column by which to color markers

        **kwargs
            any other keyword arguments, see below

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            the newly created figure

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plot.Axes.scatter
            for documentation of keyword arguments used to display the table
        """
        color = kwargs.pop('color', None)
        if color is not None:
            kwargs['c'] = self[color]
        return self._plot('scatter', self[x], self[y], **kwargs)

    def tile(self, x, y, w, h, **kwargs):
        """Make a tile plot of this table.

        Parameters
        ----------
        x : `str`
            name of column defining anchor point on the X-axis

        y : `str`
            name of column defining anchor point on the Y-axis

        w : `str`
            name of column defining extent on the X-axis (width)

        h : `str`
            name of column defining extent on the Y-axis (height)

        color : `str`, optional, default:`None`
            name of column by which to color markers

        **kwargs
            any other keyword arguments, see below

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            the newly created figure

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes
        gwpy.plot.Axes.tile
            for documentation of keyword arguments used to display the table
        """
        color = kwargs.pop('color', None)
        if color is not None:
            kwargs['color'] = self[color]
        return self._plot('tile', self[x], self[y], self[w], self[h], **kwargs)

    def _plot(self, method, *args, **kwargs):
        from matplotlib import rcParams
        from ..plot import Plot
        from ..plot.tex import label_to_latex

        if self._is_time_column(args[0].name):
            # map X column to GPS axis
            kwargs.setdefault('figsize', (12, 6))
            kwargs.setdefault('xscale', 'auto-gps')

        kwargs['method'] = method
        plot = Plot(*args, **kwargs)

        # set default labels
        ax = plot.gca()
        for axis, col in zip(
                filter(attrgetter('isDefault_label'), (ax.xaxis, ax.yaxis)),
                args[:2],
        ):
            name = col.name
            if rcParams['text.usetex']:
                name = r'\texttt{{{0}}}'.format(label_to_latex(col.name))
            if col.unit is not None:
                name += ' [{0}]'.format(col.unit.to_string('latex_inline'))
            axis.set_label_text(name)
            axis.isDefault_label = True

        return plot

    def hist(self, column, **kwargs):
        """Generate a `HistogramPlot` of this `Table`.

        Parameters
        ----------
        column : `str`
            Name of the column over which to histogram data

        method : `str`, optional
            Name of `~matplotlib.axes.Axes` method to use to plot the
            histogram, default: ``'hist'``.

        **kwargs
            Any other keyword arguments, see below.

        Returns
        -------
        plot : `~gwpy.plot.Plot`
            The newly created figure.

        See also
        --------
        matplotlib.pyplot.figure
            for documentation of keyword arguments used to create the
            figure.
        matplotlib.figure.Figure.add_subplot
            for documentation of keyword arguments used to create the
            axes.
        gwpy.plot.Axes.hist
            for documentation of keyword arguments used to display the
            histogram, if the ``method`` keyword is given, this method
            might not actually be the one used.
        """
        from ..plot import Plot
        return Plot(self[column], method='hist', **kwargs)

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
        >>> table.filter(('time', in_segmentlist, segs))
        """
        return filter_table(self, *column_filters)

    def cluster(self, index, rank, window):
        """Cluster this `EventTable` over a given column, `index`, maximizing
        over a specified column in the table, `rank`.

        The clustering algorithm uses a pooling method to identify groups
        of points that are all separated in `index` by less than `window`.

        Each cluster of nearby points is replaced by the point in that cluster
        with the maximum value of `rank`.

        Parameters
        ----------
        index : `str`
            name of the column which is used to search for clusters

        rank : `str`
            name of the column to maximize over in each cluster

        window : `float`
            window to use when clustering data points, will raise
            ValueError if `window > 0` is not satisfied

        Returns
        -------
        table : `EventTable`
            a new table that has had the clustering algorithm applied via
            slicing of the original

        Examples
        --------
        To cluster an `EventTable` (``table``) whose `index` is
        `end_time`, `window` is `0.1`, and maximize over `snr`:

        >>> table.cluster('end_time', 'snr', 0.1)
        """
        if window <= 0.0:
            raise ValueError('Window must be a positive value')

        # If no rows, no need to cluster
        if len(self) == 0:
            return self.copy()

        # Generate index and rank vectors that are ordered
        orderidx = numpy.argsort(self[index])
        col = self[index][orderidx]
        param = self[rank][orderidx]

        # Find all points where the index vector changes by less than window
        clusterpoints = numpy.where(numpy.diff(col) <= window)[0]

        # If no such cluster points, no need to cluster
        if len(clusterpoints) == 0:
            return self.copy()

        # Divide points into clusters of adjacent points
        sublists = numpy.split(clusterpoints,
                               numpy.where(numpy.diff(clusterpoints) > 1)[0]+1)

        # Add end-points to each cluster and find the index of the maximum
        # point in each list
        padded_sublists = [numpy.append(s, numpy.array([s[-1]+1]))
                           for s in sublists]
        maxidx = [s[numpy.argmax(param[s])] for s in padded_sublists]

        # Construct a mask that removes all points within clusters and
        # replaces them with the maximum point from each cluster
        mask = numpy.ones_like(col, dtype=bool)
        mask[numpy.concatenate(padded_sublists)] = False
        mask[maxidx] = True

        return self[orderidx[mask]]
