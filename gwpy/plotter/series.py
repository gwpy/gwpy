# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2018)
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

"""An extension of the Plot and Axes classes for handling Series
"""

import numpy

from matplotlib import (colors, __version__ as mpl_version)
from matplotlib.projections import register_projection

from . import text
from .core import Plot
from .axes import Axes
from .decorators import auto_refresh
from ..segments import SegmentList

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['SeriesPlot', 'SeriesAxes']

USE_IMSHOW = mpl_version >= '2.0'  # imshow respects log scaling in >=2.0


class SeriesAxes(Axes):
    """Custom `Axes` for a `~gwpy.plotter.SeriesPlot`.
    """
    name = 'series'

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('xmargin', 0)
        super(SeriesAxes, self).__init__(*args, **kwargs)

    @auto_refresh
    def plot(self, *args, **kwargs):
        """Plot data onto these Axes.

        Parameters
        ----------
        args
            a single `~gwpy.types.Series` (or sub-class)
            or standard (x, y) data arrays

        kwargs
            keyword arguments applicable to :meth:`~matplotib.axes.Axes.plot`

        Returns
        -------
        Line2D
            the `~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        from ..types import Series

        # plot series
        if len(args) == 1 and isinstance(args[0], Series):
            return self.plot_series(*args, **kwargs)

        # plot everything else
        return super(SeriesAxes, self).plot(*args, **kwargs)

    @auto_refresh
    def plot_series(self, series, **kwargs):
        """Plot a `~gwpy.types.Series` onto these axes

        Parameters
        ----------
        series : `~gwpy.types.Series`
            data to plot

        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        line : `~matplotlib.lines.Line2D`
            the newly added line

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        kwargs.setdefault('label', text.to_string(series.name))
        line = self.plot(series.xindex.value, series.value, **kwargs)

        # update datalim to include end of final sample
        if not kwargs.get('linestyle', '-').startswith('steps-'):
            self.update_datalim(list(zip(series.xspan, (0, 0))), updatey=False)
            self.autoscale_view()

        # set default limits (protecting against 0-log)
        if len(self._get_artists()) == 1:
            self._set_lim_from_array(series, 'x')

        # set labels from units
        if not self.get_xlabel():
            self.set_xlabel(text.unit_as_label(series.xunit))
        if not self.get_ylabel():
            self.set_ylabel(text.unit_as_label(series.unit))

        return line

    @auto_refresh
    def plot_mmm(self, mean_, min_=None, max_=None, **kwargs):
        """Plot a `Series` onto these axes, with shaded regions

        The ``mean_`` `Series` is plotted normally, while the ``min_``
        and ``max_`` `Series` are plotted lightly below and above,
        with a fill between them and the ``mean_``.

        Parameters
        ----------
        mean_ : `~gwpy.types.Series`
            data to plot normally

        min_ : `~gwpy.types.Series`
            first data set to shade to ``mean_``

        max_ : `~gwpy.types.Series`
            second data set to shade to ``mean_``

        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        artists : `tuple`
            a 5-tuple containing:

            - `~matplotlib.lines.Line2d` for ``mean_``,
            - `~matplotlib.lines.Line2D` for ``min_``,
            - `~matplotlib.collections.PolyCollection` for ``min_`` shading,
            - `~matplotlib.lines.Line2D` for ``max_``, and
            - `~matplitlib.collections.PolyCollection` for ``max_`` shading

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        # plot mean
        meanline = self.plot_series(mean_, **kwargs)[0]

        # modify keywords for shading
        kwargs.pop('label', None)
        color = kwargs.pop('color', meanline.get_color())
        linewidth = kwargs.pop('linewidth', meanline.get_linewidth()) / 2

        def _plot_shade(series):
            line = self.plot(series, color=color, linewidth=linewidth,
                             **kwargs)
            coll = self.fill_between(series.xindex.value, series.value,
                                     mean_.value, alpha=.1, color=color,
                                     rasterized=kwargs.get('rasterized'))
            return line, coll

        # plot lower shade
        if min_ is not None:
            minline, mincoll = _plot_shade(min_)
        else:
            minline = mincoll = None

        # plot upper shade
        if max_ is not None:
            maxline, maxcoll = _plot_shade(max_)
        else:
            maxline = maxcoll = None

        return meanline, minline, mincoll, maxline, maxcoll

    def plot_array2d(self, array, imshow=USE_IMSHOW, **kwargs):
        """Plot a 2D array onto these axes

        Parameters
        ----------
        array : `~gwpy.types.Array2D`,
            Data to plot (e.g. a `~gwpy.spectrogram.Spectrogram`)

        imshow : `bool`, optional
            If `True`, use :meth:`~matplotlib.axes.Axes.imshow` to render
            the array as an image, otherwise use
            :meth:`~matplotlib.axes.Axes.pcolormesh`, default is `True`
            with `matplotlib >= 2.0`, otherwise `False`.

        norm : ``'log'``, `~matplotlib.colors.Normalize`
            A `~matplotlib.colors.Normalize`` instance used to scale the
            colour data, or ``'log'`` to use `LogNorm`.

        **kwargs
            Any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.imshow` (if ``imshow=True``),
            or :meth:`~matplotlib.Axes.pcolormesh` (``imshow=False``)

        Returns
        -------
        layer : `~matplotlib.collections.QuadMesh`, `~matplotlib.images.Image`
            the layer for this array

        See Also
        --------
        matplotlib.axes.Axes.imshow
        matplotlib.axes.Axes.pcolormesh
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        # allow normalisation as 'log'
        if kwargs.get('norm', None) == 'log':
            vmin = kwargs.get('vmin', None)
            vmax = kwargs.get('vmax', None)
            kwargs['norm'] = colors.LogNorm(vmin=vmin, vmax=vmax)

        # plot with imshow
        if imshow:
            layer = self._imshow_array2d(array, **kwargs)

        # plot with pcolormesh
        else:
            layer = self._pcolormesh_array2d(array, **kwargs)

        # format axes
        if not self.get_ylabel():
            self.set_ylabel(text.unit_as_label(array.yunit))
        if len(self._get_artists()) == 1:  # first plotted element
            self._set_lim_from_array(array, 'x')
            self._set_lim_from_array(array, 'y')

        return layer

    def _imshow_array2d(self, array, origin='lower', interpolation='none',
                        aspect='auto', **kwargs):
        # calculate extent
        extent = tuple(array.xspan) + tuple(array.yspan)
        if self.get_xscale() == 'log' and extent[0] == 0.:
            extent = (1e-300,) + extent[1:]
        if self.get_yscale() == 'log' and extent[2] == 0.:
            extent = extent[:2] + (1e-300,) + extent[3:]
        kwargs.setdefault('extent', extent)

        return self.imshow(array.value.T, **kwargs)

    def _pcolormesh_array2d(self, array, **kwargs):
        x = numpy.concatenate((array.xindex.value, [array.xspan[-1]]))
        y = numpy.concatenate((array.yindex.value, [array.yspan[-1]]))
        xcoord, ycoord = numpy.meshgrid(x, y, copy=False, sparse=True)
        return self.pcolormesh(xcoord, ycoord, array.value.T, **kwargs)

    def _set_lim_from_array(self, array, axis):
        """Set the axis limits using the index of an `~gwpy.types.Array`
        """
        # get limits from array span
        span = getattr(array, '{}span'.format(axis))
        scale = getattr(self, 'get_{}scale'.format(axis))()
        if scale == 'log' and not span[0]:  # protect log(0)
            index = getattr(array, '{}index'.format(axis)).value
            span = index[1], span[1]

        # set limits
        set_lim = getattr(self, 'set_{}lim'.format(axis))
        return set_lim(*span)


register_projection(SeriesAxes)


class SeriesPlot(Plot):
    """`Figure` for displaying a `~gwpy.types.Series`.

    Parameters
    ----------
    *series : `Series`
        any number of `~gwpy.types.Series` to display on the plot

    **kwargs
        other keyword arguments as applicable for the
        `~gwpy.plotter.Plot`
    """
    _DefaultAxesClass = SeriesAxes

    def __init__(self, *series, **kwargs):
        """Initialise a new SeriesPlot
        """
        kwargs.setdefault('projection', self._DefaultAxesClass.name)
        self._update_kwargs_from_data(kwargs, series)

        # extract custom keyword arguments
        sep = kwargs.pop('sep', False)

        # separate keyword arguments
        axargs, plotargs = self._parse_kwargs(kwargs)
        sharex = axargs.get('sharex', False)

        # generate figure
        super(SeriesPlot, self).__init__(**kwargs)

        # plot data
        self._init_axes(series, sep, axargs, plotargs)

        # set epoch
        for ax in self.axes[:-1]:
            if sharex:
                ax.set_xlabel("")
            if 'label' in kwargs:
                ax.legend()

    def _update_kwargs_from_data(self, kwargs, data):
        """Add `__init__` keywords based on input data
        """
        flat = self._get_axes_data(data, flat=True)
        if not flat:  # bail out with no data
            return kwargs

        # set xlim based on xspans
        kwargs.setdefault(
            'xlim', SegmentList([s.xspan for s in flat]).extent())

    def _init_axes(self, data, sep, axargs, plotargs):
        """Initalise these Axes from the input data and keywords
        """
        sharex = axargs.pop('sharex', False)
        sharey = axargs.pop('sharey', False)
        groups = self._get_axes_data(data, sep=sep)
        for group in groups:
            ax = self._add_new_axes(**axargs)
            for series in group:
                ax.plot(series, **plotargs)
            if sharex is True:
                axargs.setdefault('sharex', ax)
            if sharey is True:
                axargs.setdefault('sharey', ax)
