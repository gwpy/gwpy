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

"""Plot data from a LIGO_LW-format XML event table.

The :mod:`glue.ligolw` library provides a complete set of tools to read,
write, and manipulate data tables using the LIGO_LW XML schema, the
standard for storing transient event triggers generated from many
gravitational-wave search algorithms.

This modules provides a set of Axes and a Plot wrapper in order to
display these tables in x-y format, with optional colouring.
"""

from __future__ import division

import re
from six import string_types

import numpy
from matplotlib import (cm, collections, pyplot)
from matplotlib.projections import register_projection

from glue.ligolw.table import Table

from ..time import LIGOTimeGPS
from .core import Plot
from .timeseries import (TimeSeriesAxes, TimeSeriesPlot)
from .spectrum import SpectrumPlot
from .utils import float_to_latex
from ..table.utils import (get_table_column, get_row_value)

__all__ = ['EventTableAxes', 'EventTablePlot']


class EventTableAxes(TimeSeriesAxes):
    """Custom `Axes` for an :class:`~gwpy.plotter.EventTablePlot`.

    The `EventTableAxes` inherit from
    :class:`~gwpy.plotter.TimeSeriesAxes` as a convenience to
    optionally displaying a time-column. That choice has no effect on the
    rest of the `Axes` functionality.
    """
    name = 'triggers'

    def plot(self, *args, **kwargs):
        """Plot data onto these axes

        Parameters
        ----------
        *args
            a single :class:`~glue.ligolw.table.Table` (or sub-class)
            or anything valid for
            :meth:`~gwpy.plotter.TimeSeriesPlot.plot`.
        **kwargs
            keyword arguments applicable to
            :meth:`~matplotlib.axes.Axes.plot`
        """
        if isinstance(args[0], Table):
            return self.plot_table(*args, **kwargs)
        else:
            return super(EventTableAxes, self).plot(*args, **kwargs)

    def plot_table(self, table, x, y, color=None, size_by=None,
                   size_by_log=None, size_range=None, **kwargs):
        """Plot a LIGO_LW-format event `Table` onto these `Axes`

        Parameters
        ----------
        table : :class:`~glue.ligolw.table.Table`
            LIGO_LW-format XML event `Table` to display
        x : `str`
            name of column to display on the X-axis
        y : `str`
            name of column to display on the Y-axis
        c : `str`, optional
            name of column by which to colour the data
        **kwargs
            any other arguments applicable to
            :meth:`~matplotlib.axes.Axes.scatter`

        Returns
        -------
        collection
        """
        if size_by is not None and size_by_log is not None:
            raise ValueError("size_by_color and size_by_log_color are "
                             "mutually exclusive options, please select one")
        # get x-y data
        xdata = get_table_column(table, x)
        ydata = get_table_column(table, y)

        # rank data by size or colour
        sizecol = size_by or size_by_log or (size_range and color)
        if color:
            cdata = get_table_column(table, color)
        if sizecol:
            sdata = get_table_column(table, sizecol)
        if color and sizecol:
            zipped = zip(xdata, ydata, cdata, sdata)
            zipped.sort(key=lambda row: row[2])
            try:
                xdata, ydata, cdata, sdata = map(numpy.asarray, zip(*zipped))
            except ValueError:
                pass
        elif sizecol:
            zipped = zip(xdata, ydata, sdata)
            zipped.sort(key=lambda row: row[-1])
            try:
                xdata, ydata, sdata = map(numpy.asarray, zip(*zipped))
            except ValueError:
                pass
        elif color:
            zipped = zip(xdata, ydata, cdata)
            zipped.sort(key=lambda row: row[-1])
            try:
                xdata, ydata, cdata = map(numpy.asarray, zip(*zipped))
            except ValueError:
                pass

        # work out sizing
        if sizecol:
            if size_range is None and sdata.size:
                size_range = [sdata.min(), sdata.max()]
            if size_range:
                # convert color value into a size between the given min and max
                s = kwargs.pop('s', 20)
                sizes = [s/10., s]
                sp = (sdata - size_range[0]) / (size_range[1] - size_range[0])
                sp[sp < 0] = 0
                sp[sp > 1] = 1
                if size_by_log is None:
                    sarray = sizes[0] + sp * (sizes[1] - sizes[0])
                else:
                    logsizes = numpy.log10(sizes)
                    sarray = 10 ** (logsizes[0] + sp * (
                                                    logsizes[1] - logsizes[0]))
                kwargs.setdefault('s', sarray)

        if color:
            return self.scatter(xdata, ydata, c=cdata, **kwargs)
        else:
            return self.scatter(xdata, ydata, **kwargs)

    def plot_tiles(self, table, x, y, width, height, color=None,
                   anchor='center', edgecolors='face', linewidth=0.8,
                   **kwargs):
        # get x/y data
        xdata = get_table_column(table, x)
        ydata = get_table_column(table, y)
        wdata = get_table_column(table, width)
        hdata = get_table_column(table, height)

        # get color and sort
        if color:
            cdata = get_table_column(table, color)
            zipped = zip(xdata, ydata, wdata, hdata, cdata)
            zipped.sort(key=lambda row: row[-1])
            try:
                xdata, ydata, wdata, hdata, cdata = map(numpy.asarray,
                                                        zip(*zipped))
            except ValueError:
                pass

        # construct vertices
        if anchor == 'll':
            verts = [((x, y), (x, y+height), (x+width, y+height),
                      (x+width, y)) for (x,y,width,height) in
                     zip(xdata, ydata, wdata, hdata)]
        elif anchor == 'lr':
            verts = [((x-width, y), (x-width, y+height), (x, y+height),
                      (x, y)) for (x,y,width,height) in
                     zip(xdata, ydata, wdata, hdata)]
        elif anchor == 'ul':
            verts = [((x, y-height), (x, y), (x+width, y),
                      (x+width, y-height)) for (x,y,width,height) in
                     zip(xdata, ydata, wdata, hdata)]
        elif anchor == 'ur':
            verts = [((x-width, y-height), (x-width, y), (x, y),
                      (x, y-height)) for (x,y,width,height) in
                     zip(xdata, ydata, wdata, hdata)]
        elif anchor == 'center':
            verts = [((x-width/2., y-height/2.), (x-width/2., y+height/2.),
                       (x+width/2., y+height/2.), (x+width/2., y-height/2.))
                     for (x,y,width,height) in zip(xdata, ydata, wdata, hdata)]
        else:
            raise ValueError("Unrecognised tile anchor '%s'." % anchor)

        # build collection
        cmap = kwargs.pop('cmap', cm.jet)
        coll = collections.PolyCollection(verts, edgecolors=edgecolors,
                                          linewidth=linewidth, **kwargs)
        if color:
            coll.set_array(cdata)
            coll.set_cmap(cmap)

        return self.add_collection(coll)

    def add_loudest(self, table, rank, x, y, *columns, **kwargs):
        """Display the loudest event according to some rank.

        The loudest event is displayed as a gold star at its
        position given by the values in columns ``x``, and ``y``,
        and those values are displayed in a text box.

        Parameters
        ----------
        table : `~glue.ligolw.table.Table`
            LIGO_LW-format XML event table in which to find the loudest
            event
        rank : `str`
            name of column to use for ranking
        x : `str`
            name of column to display on the X-axis
        y : `str`
            name of column to display on the Y-axis
        color : `str`, optional
            name of column by which to colour the data
        **kwargs
            any other arguments applicable to
            :meth:`~matplotlib.axes.Axes.text`

        Returns
        -------
        out : `tuple`
            (`collection`, `text`) tuple of items added to the `Axes`
        """
        ylim = self.get_ylim()
        row = table[get_table_column(table, rank).argmax()]
        disp = "Loudest event:"
        columns = [x, y, rank] + list(columns)
        scat = []
        for i, column in enumerate(columns):
            if not column or column in columns[:i]:
                continue
            if i:
                disp += ','
            val = get_row_value(row, column)
            if i < 2:
                scat.append([float(val)])
            column = get_column_string(column)
            if pyplot.rcParams['text.usetex'] and column.endswith('Time'):
                disp += (r" %s$= %s$" % (column, LIGOTimeGPS(val)))
            elif pyplot.rcParams['text.usetex']:
                disp += (r" %s$=$ %s" % (column, float_to_latex(val, '%.3g')))
            else:
                disp += " %s = %.2g" % (column, val)
        disp = disp.rstrip(',')
        pos = kwargs.pop('position', [0.5, 1.00])
        kwargs.setdefault('transform', self.axes.transAxes)
        kwargs.setdefault('verticalalignment', 'bottom')
        kwargs.setdefault('horizontalalignment', 'center')
        args = pos + [disp]
        self.scatter(*scat, marker='*', zorder=1000, facecolor='gold',
                     edgecolor='black',  s=200)
        self.text(*args, **kwargs)
        if self.get_title():
            pos = self.title.get_position()
            self.title.set_position((pos[0], pos[1] + 0.05))
        self.set_ylim(*ylim)

register_projection(EventTableAxes)


class _EventTableMetaPlot(type):
    """Meta-class for generating a new :class:`EventTablePlot`.

    This object allows the choice of parent class for the
    `EventTablePlot` to be made at runtime, dependent on the given
    x-column of the first displayed Table.
    """
    def __call__(cls, *args, **kwargs):
        """Execute the meta-class, given the arguments for the plot

        All ``*args`` and ``**kwargs`` are those passed to the
        `EventTablePlot` constructor, used to determine the appropriate
        parent class the that object at runtime.
        """
        # find x-column: copy the arguments and find the strings
        a2 = list(args)
        while len(a2):
            if isinstance(a2[0], string_types):
                break
            a2.pop(0)
        # if at least one string was found, treat it as the x-axis column name
        if 'base' in kwargs:
            plotclass = kwargs.pop('base')
        elif len(a2):
            xcol = a2[0]
            # initialise figure as a TimeSeriesPlot
            if re.search('time\Z', xcol, re.I):
                plotclass = TimeSeriesPlot
            # or as a SpectrumPlot
            elif re.search('(freq\Z|frequency\Z)', xcol, re.I):
                plotclass = SpectrumPlot
            # otherwise as a standard Plot
            else:
               plotclass = Plot
        else:
            plotclass = Plot
        cls.__bases__ = (plotclass,)
        return super(_EventTableMetaPlot, cls).__call__(*args, **kwargs)


class EventTablePlot(TimeSeriesPlot, SpectrumPlot, Plot):
    """`Figure` for displaying a :class:`~glue.ligolw.table.Table`.

    Parameters
    ----------
    table : :class:`~glue.ligolw.table.Table`
        LIGO_LW-format XML event `Table` to display
    x : `str`
        name of column to display on the X-axis
    y : `str`
        name of column to display on the Y-axis
    c : `str`, optional
        name of column by which to colour the data
    **kwargs
        any other arguments applicable to the `Plot` constructor, and
        the `Table` plotter.

    Returns
    -------
    plot : :class:`EventTablePlot`
        new plot for displaying tabular data.

    Notes
    -----
    The form of the returned `EventTablePlot` is decided at run-time,
    rather than when the module was imported.
    If tables are passed directly to the constructor, for example::

        >>> plot = EventTablePlot(table1, 'time', 'snr')

    the columns as given are used to determine the appropriate parent
    class for the output.

    If the input x-column (the first string argument) ends with 'time'
    the output is a child of the :class:`~gwpy.plotter.TimeSeriesPlot`,
    allowing easy formatting of GPS times, while if the x-column ends with
    'frequency', the output comes from the
    :class:`~gwpy.plotter.SpectrumPlot`, otherwise the parent is
    the core :class:`~gwpy.plotter.Plot`.
    """
    _DefaultAxesClass = EventTableAxes
    __metaclass__ = _EventTableMetaPlot

    def __init__(self, *args, **kwargs):
        """Generate a new `EventTablePlot`.
        """
        # extract plotting keyword arguments
        plotargs = dict()
        for arg in ['linewidth', 'facecolor', 'edgecolor', 'marker', 'cmap',
                    's', 'size_by', 'size_by_log', 'size_range', 'label',
                    'edgecolors', 'facecolors', 'linewidths', 'antialiaseds',
                    'offsets', 'transOffset', 'norm']:
            if arg in kwargs:
                val = kwargs.pop(arg, None)
                if val is not None:
                    plotargs[arg] = val

        # extract columns
        args = list(args)
        tables = []
        columns = {}
        tiles = False
        while len(args):
            arg = args[0]
            if isinstance(arg, string_types):
                break
            tables.append(args.pop(0))
        if len(tables) != 0 and len(args) < 2:
            raise ValueError("columnnames for 'x' and 'y' axes must be given "
                             "after any tables, e.g. "
                             "TablePlot(t1, t2, 'time', 'snr')")
        if len(args) in [3, 5]:
            kwargs.setdefault('color', args.pop(-1))
        elif len(args) == 4:
            tiles = True
        elif len(args) > 5:
            raise ValueError("No more than three column names should be given")
        if len(tables):
            columns = dict(zip(['x', 'y', 'width', 'height'], args))

        # extract column arguments
        epoch = kwargs.pop('epoch', None)
        sep = kwargs.pop('sep', False)
        columns['color'] = kwargs.pop('color', None)
        if columns['color'] and len(tables) > 1 and not sep:
            raise ValueError("Plotting multiple Tables on a single set of "
                             "Axes with a colorbar is not supported, "
                             "currently...")

        super(EventTablePlot, self).__init__(**kwargs)

        # plot data
        for table in tables:
            if len(args) == 2:
                self.add_table(table, columns['x'], columns['y'],
                               color=columns['color'], newax=sep, **plotargs)
            elif len(args) == 4:
                self.add_tiles(table, columns['x'], columns['y'],
                               columns['width'], columns['height'],
                               color=columns['color'], newax=sep, **plotargs)
        if len(tables):
            # set auto-scale
            for ax in self.axes:
                ax.autoscale(axis='both', tight=True)
            # set individual epoch for TimeSeriesAxes
            if isinstance(self, TimeSeriesPlot) and sep:
                for ax,table in zip(self.axes, tables):
                    axepoch = epoch
                    if axepoch is None:
                        tcol = numpy.asarray(get_table_column(table,
                                                              columns['x']))
                        if epoch is None:
                            epoch = tcol.min()
                        else:
                            epoch = min(epoch, tcol.min())
                    ax.set_epoch(epoch)
            # set global epoch for TimeSeriesPlot
            elif isinstance(self, TimeSeriesPlot):
                ax = self.axes[0]
                if epoch is None:
                    for table in tables:
                        tcol = numpy.asarray(get_table_column(table,
                                                              columns['x']))
                        if epoch is None:
                            epoch = tcol.min()
                        else:
                            epoch = min(epoch, tcol.min())
                ax.set_epoch(epoch)
            # remove labels
            if sep:
                for ax in self.axes[:-1]:
                    ax.set_xlabel("")

    def add_table(self, table, x, y, color=None, projection='triggers', ax=None,
                  newax=None, **kwargs):
        """Add a LIGO_LW Table to this Plot

        Parameters
        ----------
        table : :class:`~glue.ligolw.table.Table`
            LIGO_LW-format XML event `Table` to display
        x : `str`
            name of column to display on the X-axis
        y : `str`
            name of column to display on the Y-axis
        c : `str`, optional
            name of column by which to colour the data
        projection : `str`, optiona, default: ``'triggers'``
            name of the Axes projection on which to plot data
        ax : :class:`~gwpy.plotter.Axes`, optional
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs.
            other keyword arguments passed to the
            :meth:`EventTableAxes.plot_table` method

        Returns
        -------
        scatter : :class:`~matplotlib.collections.Collection`
            the displayed collection for this `Table`

        See Also
        --------
        :meth:`EventTableAxes.plot_table`
            for details on arguments and keyword arguments other than
            ``ax`` and ``newax`` for this method.
        """
        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection)
        return ax.plot_table(table, x, y, color=color, **kwargs)

    def add_tiles(self, table, x, y, width, height, color=None,
                  anchor='center', projection='triggers', ax=None,
                  newax=None, **kwargs):
        """Add a LIGO_LW Table to this Plot

        Parameters
        ----------
        table : :class:`~glue.ligolw.table.Table`
            LIGO_LW-format XML event `Table` to display
        x : `str`
            name of column for tile x-anchor
        y : `str`
            name of column for tile y-anchor
        width : `str`
            name of column for tile width
        height : `str`
            name of column for tile height
        color : `str`, optional
            name of column by which to colour the data
        anchor : `str`, optional, default: ``'center'``
            position of (x, y) vertex on tile, default 'center'.
            Other options: 'll', 'lr', 'ul', 'ur'.
        projection : `str`, optiona, default: ``'triggers'``
            name of the Axes projection on which to plot data
        ax : :class:`~gwpy.plotter.Axes`, optional
            the `Axes` on which to add these data, if this is not given,
            a guess will be made as to the best `Axes` to use. If no
            appropriate axes are found, new `Axes` will be created
        newax : `bool`, optional, default: `False`
            force data to plot on a fresh set of `Axes`
        **kwargs.
            other keyword arguments passed to the
            :meth:`EventTableAxes.plot_table` method

        Returns
        -------
        scatter : :class:`~matplotlib.collections.Collection`
            the displayed collection for this `Table`

        See Also
        --------
        :meth:`EventTableAxes.plot_table`
            for details on arguments and keyword arguments other than
            ``ax`` and ``newax`` for this method.
        """
        # find relevant axes
        if ax is None and not newax:
            try:
                ax = self._find_axes(projection)
            except IndexError:
                newax = True
        if newax:
            ax = self._add_new_axes(projection=projection)
        return ax.plot_tiles(table, x, y, width, height, color=color,
                             anchor=anchor, **kwargs)


def get_column_string(column):
    """
    Format the string columnName (e.g. xml table column) into latex format for.
    an axis label.

    Examples:

    >>> get_column_string('snr')
    'SNR'

    >>> get_column_string('bank_chisq_dof')
    r'Bank $\chi^2$ DOF'

    Arguments:

      columnName : string
        string to format
    """
    acro = ['snr', 'ra', 'dof', 'id', 'ms', 'far']
    greek = ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'eta',
             'theta', 'iota', 'kappa', 'lamda', 'mu', 'nu', 'xi', 'omicron',
             'pi', 'rho', 'sigma', 'tau', 'upsilon', 'phi', 'chi', 'psi',
             'omega']
    unit = ['ns']
    sub = ['flow', 'fhigh', 'hrss', 'mtotal', 'mchirp']

    tex = pyplot.rcParams['text.usetex']

    words = []
    for w in re.split('\s', column):
        if w.isupper():
            words.append(w)
        else:
            words.extend(re.split('_', w))

    # parse words
    for i, w in enumerate(words):
        # get acronym in lower case
        if w in acro:
            words[i] = w.upper()
        # get numerical unit
        elif w in unit:
            words[i] = '$(%s)$' % w
        # get character with subscript text
        elif w in sub and tex:
            words[i] = '%s$_{\mbox{\\small %s}}$' % (w[0], w[1:])
        # get greek word
        elif w in greek and tex:
            words[i] = '$\%s$' % w
        # get starting with greek word
        elif re.match('(%s)' % '|'.join(greek), w) and tex:
            if w[-1].isdigit():
                words[i] = '$\%s_{%s}$''' % tuple(re.findall(r"[a-zA-Z]+|\d+",w))
            elif w.endswith('sq'):
                words[i] = '$\%s^2$' % w.rstrip('sq')
        # get everything else
        else:
            if w.isupper():
                words[i] = w
            else:
                words[i] = w.title()
            # escape underscore
            words[i] = re.sub(r'(?<!\\)_', r'\_', words[i])
    return ' '.join(words)
