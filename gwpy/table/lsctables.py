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

"""Drop in of :mod:`glue.ligolw.lsctables` to annotate Table classes.
"""

import warnings

from glue.ligolw.lsctables import *
from glue.ligolw.table import StripTableName as strip_table_name
from glue.ligolw.types import ToNumPyType as NUMPY_TYPE
from glue.ligolw.ilwd import get_ilwdchar_class

from ..io import reader
from ..time import to_gps

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__all__ = TableByName.keys()

NUMPY_TYPE['ilwd:char'] = numpy.dtype(int).name
NUMPY_TYPE['lstring'] = 'a20'


def to_recarray(self, columns=None, on_attributeerror='raise'):
    """Convert this table to a structured `numpy.recarray`

    This returned `~numpy.recarray` is a blank data container, mapping
    columns in the original LIGO_LW table to fields in the output, but
    mapping none of the instance methods of the origin table.

    Parameters
    ----------
    columns : `list` of `str`, optional
        the columns to populate, if not given, all columns present in the
        table are mapped
    on_attributeerror : `str`
        how to handle `AttributeError` when accessing rows, one of

        - 'raise' : raise normal exception
        - 'ignore' : skip over this column
        - 'warn' : print a warning instead of raising error

    """
    # get numpy-type columns
    if columns is None:
       columns = self.columnnames
    dtypes = [(str(c), NUMPY_TYPE[self.validcolumns[c]])
              for c in columns]
    # create array
    m = len(self)
    out = numpy.recarray((len(self),), dtype=dtypes)
    # and fill it
    for column in columns:
        orig_type = self.validcolumns[column]
        try:
            if orig_type == 'ilwd:char':  # numpy tries long() which breaks
                out[column] = map(int, self.getColumnByName(column))
            else:
                out[column] = self.getColumnByName(column)
        except AttributeError as e:
            if on_attributeerror == 'ignore':
                pass
            elif on_attributeerror == 'warn':
                warnings.warn('Caught %s: %s' % (type(e).__name__, str(e)))
            else:
                raise
    return out


def from_recarray(cls, array, columns=None):
    """Create a new table from a `numpy.recarray`

    Parameters
    ----------
    array : `numpy.recarray`
        an array of data
    column : `list` of `str`, optional
        the columns to populate, if not given, all columns present in the
        `~numpy.recarray` are mapped

    Notes
    -----
    The columns populated in the `numpy.recarray` must all map exactly to
    valid columns of the target `~glue.ligolw.table.Table`.
    """
    if columns is None:
        columns = list(array.dtype.names)
    out = New(cls, columns=columns)
    tblname = strip_table_name(out.tableName)
    ilwdchar = dict((col, get_ilwdchar_class(tblname, col))
                    for (col, llwtype) in zip(out.columnnames, out.columntypes)
                    if llwtype == 'ilwd:char')
    for rec in array:
        row = out.RowType()
        for col, llwtype in zip(out.columnnames, out.columntypes):
            if llwtype == 'ilwd:char':
                setattr(row, col, ilwdchar[col](rec[col]))
            else:
                setattr(row, col, rec[col])
        out.append(row)
    return out


def _plot_factory():
    def plot(self, *args, **kwargs):
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
               either both not given, in which case a scatter plot will
               be drawn, or both given, in which case a collections of
               rectangles will be drawn.

        color : `str`, optional, default:`None`
            name of column by which to color markers
        **kwargs
            any other arguments applicable to the `Plot` constructor, and
            the `Table` plotter.

        Returns
        -------
        plot : :class:`~gwpy.plotter.EventTablePlot`
            new plot for displaying tabular data.

        See Also
        --------
        gwpy.plotter.EventTablePlot : for more details.
        """
        from gwpy.plotter import EventTablePlot
        return EventTablePlot(self, *args, **kwargs)

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
        plot : :class:`~gwpy.plotter.HistogramPlot`
            new plot displaying a histogram of this `Table`.
        """
        from gwpy.plotter import HistogramPlot
        return HistogramPlot(self, column, **kwargs)

    return (plot, hist)


def _fetch_factory(table):
    def fetch(cls, channel, etg, start, end, verbose=False, **kwargs):
        """Find and read events into a `{0}`.

        Event XML files are searched for only on the LIGO Data Grid
        using the conventions set out in LIGO-T1300468.

        .. warning::

            This method will not work on machines outside of the LIGO Lab
            computing centres at Caltech, LHO, and LHO.

        Parameters
        ----------
        channel : `str`
            the name of the data channel to search for
        etg : `str`
            the name of the event trigger generator (ETG)
        start : `float`, `~gwpy.time.LIGOTimeGPS`
            the GPS start time of the search
        end : `float`, `~gwpy.time.LIGOTimeGPS`
            the GPS end time of the search
        verbose : `bool`, optional
            print verbose output, default: `False`
        **kwargs
            other keyword arguments to pass to :meth:`{0}.read`

        Returns
        -------
        table : :class:`{0}`
            a new `{0}` containing triggers read from the standard
            paths

        Raises
        ------
        ValueError
            if no channel-level directory is found for the given channel,
            indicating that nothing has ever been processed for this channel.

        See also
        --------
        {0}.read :
            for documentation of the available keyword arguments
        """
        from gwpy.segments import Segment
        from .io.trigfind import find_trigger_urls
        # check times
        start = to_gps(start)
        end = to_gps(end)
        # find files
        cache = find_trigger_urls(channel, etg, start, end, verbose=verbose)
        # construct filter
        infilt = kwargs.pop('filt', None)
        segment = Segment(float(start), float(end))
        if infilt is None:
            def filt(row):
                return float(row.get_peak()) in segment
        else:
            def filt(row):
                return infilt(row) and float(row.get_peak()) in segment
        # read and return
        return cls.read(cache, format='ligolw', filt=filt, **kwargs)
    fetch.__doc__ = fetch.__doc__.format(table.__name__)

    return classmethod(fetch)

# annotate lsctables with new methods
for table in TableByName.itervalues():
    # define the read classmethod with docstring
    table.read = classmethod(reader(doc="""
        Read data into a `{0}`.

        Parameters
        ----------
        f : `file`, `str`, `~glue.lal.CacheEntry`, `list`, `~glue.lal.Cache`
            object representing one or more files. One of

                - an open `file`
                - a `str` pointing to a file path on disk
                - a formatted `~glue.lal.CacheEntry` representing one file
                - a `list` of `str` file paths
                - a formatted `~glue.lal.Cache` representing many files

        columns : `list`, optional
            list of column name strings to read, default all.

        ifo : `str`, optional
            prefix of IFO to read

            .. warning::

               the ``ifo`` keyword argument is only applicable (but is
               required) when reading single-interferometer data from
               a multi-interferometer file

        filt : `function`, optional
            function by which to `filter` events. The callable must
            accept as input a row of the table event and return
            `True`/`False`.

        nproc : `int`, optional, default: ``1``
            number of parallel processes with which to distribute file I/O,
            default: serial process.

            .. warning::

               The ``nproc`` keyword argument is only applicable when
               reading a `list` (or `~glue.lal.Cache`) of files.

        contenthandler : `~glue.ligolw.ligolw.LIGOLWContentHandler`
            SAX content handler for parsing ``LIGO_LW`` documents.

            .. warning::

               The ``contenthandler`` keyword argument is only applicable
               when reading from ``LIGO_LW`` documents.

        **loadtxtkwargs
            when reading from ASCII, all other keyword arguments are passed
            directly to `numpy.loadtxt`

        Returns
        -------
        table : :class:`{0}`
            `{0}` of data with given columns filled

        Notes
        -----
        """.format(table.__name__)))

    table.to_recarray = to_recarray
    table.from_recarray = classmethod(from_recarray)

    if ('start_time' in table.validcolumns or
            'peak_time' in table.validcolumns or
            'end_time' in table.validcolumns):
        table.plot, table.hist = _plot_factory()

    if table == SnglBurstTable:
        table.fetch = _fetch_factory(table)
