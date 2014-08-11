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

from glue.ligolw.lsctables import *

from .. import version
from ..io import reader

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


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
        """Generate a `HistogrmaPlot` of this `Table`.

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

        Returns
        -------
        table : :class:`{0}`
            `{0}` of data with given columns filled

        Notes
        -----
        """.format(table.__name__)))
    if ('start_time' in table.validcolumns or
            'peak_time' in table.validcolumns or
            'end_time' in table.validcolumns):
        table.plot, table.hist = _plot_factory()
