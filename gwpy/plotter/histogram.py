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

"""Docstring
"""

from __future__ import division

from math import log10

import numpy

from matplotlib.cbook import iterable
from matplotlib.projections import register_projection

from glue import iterutils
from glue.ligolw.table import Table

from ..table.utils import get_table_column
from .axes import Axes
from .core import Plot
from ..data import Series
from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class HistogramAxes(Axes):
    """A set of `Axes` on which to display a histogram.
    """
    name = 'histogram'

    def hist_series(self, series, **kwargs):
        """Add a histogram of the given 1-dimensional series to these `Axes`.

        Parameters
        ----------
        series : :class:`~gwpy.data.series.Series`
            1-dimensional `Series` of data.
        **kwargs
            common histogram keyword arguments to pass to
            :meth:`~HistogramAxes.hist`.

        See Also
        --------
        HistogramAxes.hist : for details on keyword arguments
        """
        return self.hist(series.data, **kwargs)

    def hist_table(self, table, column, **kwargs):
        """Add a histogram of the given :class:`~glue.ligolw.table.Table`.

        Parameters
        ----------
        table : :class:`~glue.ligolw.table.Table`
            LIGO_LW-format table of data to analyse.
        column : `str`
            name of ``table`` column to histogram.
        **kwargs
            common histogram keyword arguments to pass to
            :meth:`~HistogramAxes.hist`.

        See Also
        --------
        HistogramAxes.hist : for details on keyword arguments
        """
        data = get_table_column(table, column)
        return self.hist(data, **kwargs)

    def hist(self, x, **kwargs):
        logbins = kwargs.pop('logbins', False)
        bins = kwargs.get('bins', 30)
        weights = kwargs.get('weights', None)
        if isinstance(weights, (float, int)):
            kwargs['weights'] = []
            if isinstance(x, numpy.ndarray) or not iterable(x[0]):
                kwargs['weights'].append(numpy.ones_like(x) * weights)
            else:
                for x2 in x:
                    kwargs['weights'].append(numpy.ones_like(x2) * weights)
        if logbins and (bins is None or isinstance(bins, (float, int))):
            bins = bins or 30
            range_ = kwargs.pop('range', self.common_limits(x))
            kwargs['bins'] = self.bin_boundaries(range_[0], range_[1], bins,
                                                 log=True)
        if kwargs.get('histtype', None) == 'stepfilled':
            kwargs.setdefault('edgecolor', 'black')
        return super(HistogramAxes, self).hist(x, **kwargs)
    hist.__doc__ = Axes.hist.__doc__

    @staticmethod
    def common_limits(datasets, default_min=0, default_max=0):
        """Find the global maxima and minima of a list of datasets.

        Parameters
        ----------
        datasets : `iterable`
            list (or any other iterable) of data arrays to analyse.
        default_min : `float`, optional
            fall-back minimum value if datasets are all empty.
        default_max : `float`, optional
            fall-back maximum value if datasets are all empty.

        Returns
        -------
        (min, max) : `float`
            2-tuple of common minimum and maximum over all datasets.
        """
        if isinstance(datasets, numpy.ndarray) or not iterable(datasets[0]):
            datasets = [datasets]
        max_stat = max(list(iterutils.flatten(datasets)) + [-numpy.inf])
        min_stat = min(list(iterutils.flatten(datasets)) + [numpy.inf])
        if numpy.isinf(-max_stat):
            max_stat = default_max
        if numpy.isinf(min_stat):
            min_stat = default_min
        return min_stat, max_stat

    @staticmethod
    def bin_boundaries(lower, upper, num, log=False):
        """Determine the bin boundaries for the given interval.

        The returned array contains the left edge of all bins, as well as
        the right-most edge of the right-most bin.

        Parameters
        ----------
        lower : `float`
            minimum boundary for bins. This value will be the first
            element of the returned bins array.
        upper : `float`
            maximum boundary for bins. This value will be the last
            element of the returned bins array.
        num : `int`
            number of bins to generate
        log : `bool`, optional, default: `False`
            if `True` generate logarithmically-spaced bins, else
            generate linearly-spaced bins.

        Returns
        -------
        bins : :class:`~numpy.ndarray`
            array of bins (length ``len(num)`` + 1).
        """
        if log:
            return numpy.logspace(log10(lower), log10(upper), num+1,
                                  endpoint=True)
        else:
            return numpy.linspace(lower, upper, num+1, endpoint=True)

register_projection(HistogramAxes)


class HistogramPlot(Plot):
    """A plot showing a histogram of data
    """
    _DefaultAxesClass = HistogramAxes

    def __init__(self, *data, **kwargs):
        """Generate a new `HistogramPlot` from some ``data``.
        """
        # extract histogram arguments
        histargs = dict()
        for key in ['bins', 'range', 'normed', 'weights', 'cumulative',
                    'bottom', 'histtype', 'align', 'orientation', 'rwidth',
                    'log', 'color', 'label', 'stacked', 'logbins']:
            try:
                histargs[key] = kwargs.pop(key)
            except KeyError:
                pass
        # generate Figure
        super(HistogramPlot, self).__init__(**kwargs)
        # plot data
        if data:
            ax = self.gca()
            data = list(data)
        while data:
            dataset = data.pop(0)
            if isinstance(dataset, Series):
                ax.hist_series(dataset, **histargs)
            elif isinstance(dataset, Table):
                column = data.pop()
                ax.hist_table(dataset, column, **histargs)
            else:
                ax.hist(dataset, **histargs)
