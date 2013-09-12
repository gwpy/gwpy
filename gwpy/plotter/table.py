# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Plot data from a Table
"""

import numpy
from matplotlib import pyplot

from .core import Plot
from .decorators import auto_refresh
from ..table import Table
from . import (tex, ticks)


class TablePlot(Plot):
    """Plot data directly from a Table
    """
    def __init__(self, *args, **kwargs):
        # extract plotting keyword arguments
        plotargs = dict()
        plotargs['facecolor'] = kwargs.pop('facecolor', None)
        plotargs['edgecolor'] = kwargs.pop('edgecolor', None)
        plotargs['marker'] = kwargs.pop('marker', None)
        plotargs['cmap'] = kwargs.pop('cmap', None)
        plotargs = dict(kvp for kvp in plotargs.iteritems() if
                        kvp[1] is not None)

        # extract slot-based arguments
        slotargs = dict()
        for key,val in kwargs.items():
            if key in self.__slots__:
                slotargs[key] = kwargs.pop(key, val)

        # extract columns
        tables = []
        columns = {}
        columns['x'], columns['y'] = args[-2:]
        if (not isinstance(columns['x'], basestring) or not
                isinstance(columns['y'], basestring)):
            raise ValueError("columnnames for 'x' and 'y' axes must be given "
                             "after any tables, e.g. "
                             "TablePlot(t1, t2, 'time', 'snr')")
        tables = args[:-2]

        # extract column arguments
        columns['color'] = kwargs.pop('colorcolumn', None)
        if columns['color'] and len(tables) > 1:
            raise ValueError("Plotting multiple Tables with a colorbar is "
                             "not supported, currently...")

        # initialise figure
        super(TablePlot, self).__init__(**kwargs)
        self._tables = []

        # plot figures
        for table in tables:
            self.add_table(table, columns['x'], columns['y'], columns['color'],
                           **plotargs)
            self._tables.append(table)

        self._xcolumn = columns['x']
        self._ycolumn = columns['y']
        self._colorcolumn = columns['color']

        # set slot arguments
        for key,val in sorted(slotargs.iteritems(), key=lambda x: x[0]):
            setattr(self, key, val)

    def add_table(self, table, x, y, color=None, **kwargs):
        # get xdata
        xdata = get_column(table, x)
        ydata = get_column(table, y)
        if color:
            cdata = get_column(table, color)
            zipped = zip(xdata, ydata, cdata)
            zipped.sort(key=lambda (x,y,c): c)
            xdata, ydata, cdata = map(numpy.asarray, zip(*zipped))
            self.add_markers(xdata, ydata, c=cdata, **kwargs)
        else:
            self.add_markers(xdata, ydata, **kwargs)

    def add_loudest(self, rank=None, columns=None, **kwargs):
        if rank is None:
            rank = self._colorcolumn
        if columns is None:
            columns = [self._xcolumn, self._ycolumn, self._colorcolumn]
        if len(self._tables) > 1:
            raise RuntimeError("Cannot display loudest event for a plot "
                               "over multiple tables")
        row = self._tables[0][self._tables[0][rank].argmax()]
        disp = "Loudest event:"
        for i,column in enumerate(columns):
            if i:
                disp += ','
            unit = self._tables[0][column].units
            if pyplot.rcParams['text.usetex']:
                unit = (unit and tex.unit_to_latex(unit) or '')
                disp += (r" ${\rm %s} = %s %s$"
                         % (column, row[column], unit))
            else:
                unit = (unit and str(unit) or '')
                disp += " %s = %.2g %s" % (column, row[column], unit).rstrip()
        pos = kwargs.pop('position', [0.01, 0.98])
        kwargs.setdefault('transform', self.axes.transAxes)
        kwargs.setdefault('verticalalignment', 'top')
        kwargs.setdefault('backgroundcolor', 'white')
        kwargs.setdefault('bbox', dict(facecolor='white', alpha=1.0,
                                       edgecolor='black', pad=6.0))
        args = pos + [disp]
        self.add_markers([row[self._xcolumn]], [row[self._ycolumn]],
                         marker='*', zorder=1000, color='gold',
                         s=80)
        #self.add_legend(alpha=0.5, loc='upper left', scatterpoints=1)
        self.axes.text(*args, **kwargs)


def get_column(table, column):
    """Extract a column from the given table
    """
    if isinstance(table, Table):
        return table[column]
    else:
        column = str(column).lower()
        if hasattr(table, "get_%s" % column):
            return numpy.asarray(getattr(table, "get_%s" % column)())
        elif column == "time":
            if re.match("(sim_inspiral|multi_inspiral)", table.tableName,
                        re.I):
                return table.get_end()
            elif re.match("(sngl_burst|multi_burst)", table.tableName, re.I):
                return table.get_peak()
            elif re.match("(sngl_ring|multi_ring)", table.tableName, re.I):
                return table.get_start()
        return numpy.asarray(table.getColumnByName(column))
