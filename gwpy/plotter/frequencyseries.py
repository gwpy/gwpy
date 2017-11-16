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

"""This module defines plotting classes for the data series defined in
`~gwpy.frequencyseries`
"""

import warnings

import numpy

from matplotlib.projections import register_projection
from matplotlib import colors

from . import text
from .core import Plot
from .axes import Axes
from .decorators import auto_refresh
from ..frequencyseries import (FrequencySeries, SpectralVariance)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['FrequencySeriesAxes', 'FrequencySeriesPlot']


class FrequencySeriesAxes(Axes):
    """Custom `Axes` for a `~gwpy.plotter.FrequencySeriesPlot`.
    """
    name = 'frequencyseries'

    # -------------------------------------------
    # GWpy class plotting methods

    @auto_refresh
    def plot(self, *args, **kwargs):
        """Plot data onto these Axes.

        Parameters
        ----------
        args
            a single `~gwpy.frequencyseries.FrequencySeries`
            (or sub-class) or standard (x, y) data arrays
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
        if len(args) == 1 and isinstance(args[0], FrequencySeries):
            return self.plot_frequencyseries(*args, **kwargs)
        if len(args) == 1 and isinstance(args[0], SpectralVariance):
            return self.plot_variance(*args, **kwargs)
        return super(FrequencySeriesAxes, self).plot(*args, **kwargs)

    @auto_refresh
    def plot_frequencyseries(self, spectrum, **kwargs):
        """Plot a `~gwpy.frequencyseries.FrequencySeries` onto these axes

        Parameters
        ----------
        spectrum : `~gwpy.frequencyseries.FrequencySeries`
            data to plot

        **kwargs
            any other keyword arguments acceptable for `~matplotlib.Axes.plot`

        Returns
        -------
        line : `~matplotlib.lines.Line2D`
            the newly added line

        See Also
        --------
        matplotlib.axes.Axes.plot
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        kwargs.setdefault('label', text.to_string(spectrum.name))
        if not kwargs.get('label', True):
            kwargs.pop('label')
        line = self.plot(spectrum.frequencies.value, spectrum.value, **kwargs)
        if len(self.lines) == 1:
            span = spectrum.xspan
            if self.get_xscale() == 'log' and not span[0]:
                span = (spectrum.df.value, span[-1])
            self.set_xlim(*span)
        if not self.get_xlabel():
            self.set_xlabel(text.unit_as_label(spectrum.xunit))
        if not self.get_ylabel():
            self.set_ylabel(text.unit_as_label(spectrum.unit))
        return line

    @auto_refresh
    def plot_spectrum(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        warnings.warn("{0}.plot_spectrum was renamed "
                      "{0}.plot_frequencyseries, "
                      "and will be removed in an upcoming release".format(
                          type(self).__name__))
        return self.plot_frequencyseries(*args, **kwargs)

    @auto_refresh
    def plot_frequencyseries_mmm(self, mean_, min_=None, max_=None, alpha=0.1,
                                 **kwargs):
        """Plot a `FrequencySeries` onto these axes, with shaded regions

        The ``mean_`` `FrequencySeries` is plotted normally, while the
        ``min_`` and ``max_`` series are plotted lightly below and above,
        with a fill between them and ``mean_``.

        Parameters
        ----------
        mean_ : `~gwpy.frequencyseries.FrequencySeries`
            data to plot normally

        min_ : `~gwpy.frequencyseries.FrequencySeries`
            first data set to shade to ``mean_``

        max_ : `~gwpy.frequencyseries.FrequencySeries`
            second data set to shade to ``mean_``

        alpha : `float`, optional
            weight of filled region, ``0.0`` for transparent through ``1.0``
            opaque

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
        meanline = self.plot_frequencyseries(mean_, **kwargs)[0]
        # plot min and max
        kwargs.pop('label', None)
        color = kwargs.pop('color', meanline.get_color())
        linewidth = kwargs.pop('linewidth', meanline.get_linewidth()) / 10
        if min_ is not None:
            minline = self.plot(min_.frequencies.value, min_.value,
                                color=color, linewidth=linewidth, **kwargs)
            if alpha:
                mincol = self.fill_between(min_.frequencies.value, mean_.value,
                                           min_.value, alpha=alpha,
                                           color=color,
                                           rasterized=kwargs.get('rasterized'))
            else:
                mincol = None
        else:
            minline = mincol = None
        if max_ is not None:
            maxline = self.plot(max_.frequencies.value, max_.value,
                                color=color, linewidth=linewidth, **kwargs)
            if alpha:
                maxcol = self.fill_between(max_.frequencies.value, mean_.value,
                                           max_.value, alpha=alpha,
                                           color=color,
                                           rasterized=kwargs.get('rasterized'))
            else:
                maxcol = None
        else:
            maxline = maxcol = None
        return meanline, minline, mincol, maxline, maxcol

    @auto_refresh
    def plot_spectrum_mmm(self, *args, **kwargs):
        # pylint: disable=missing-docstring
        warnings.warn("{0}.plot_spectrum_mmm was renamed "
                      "{0}.plot_frequencyseries_mmm, "
                      "and will be removed in an upcoming release".format(
                          type(self).__name__))
        return self.plot_frequencyseries_mmm(*args, **kwargs)

    @auto_refresh
    def plot_variance(self, specvar, norm='log', **kwargs):
        """Plot a `~gwpy.frequencyseries.SpectralVariance` onto
        these axes

        Parameters
        ----------
        spectrum : class:`~gwpy.frequencyseries.SpectralVariance`
            data to plot
        **kwargs
            any other eyword arguments acceptable for
            :meth:`~matplotlib.Axes.pcolormesh`

        Returns
        -------
        mesh : `~matplotlib.collections.MeshGrid`
            the collection that has just been added

        See Also
        --------
        matplotlib.axes.Axes.pcolormesh
            for a full description of acceptable ``*args`` and ``**kwargs``
        """
        if norm == 'log':
            vmin = kwargs.pop('vmin', None)
            vmax = kwargs.pop('vmax', None)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm
        x = numpy.concatenate((specvar.frequencies.value,
                               [specvar.x0.value +
                                specvar.dx.value * specvar.shape[0]]))
        y = specvar.bins.value
        xcoord, ycoord = numpy.meshgrid(x, y, copy=False, sparse=True)
        mesh = self.pcolormesh(xcoord, ycoord, specvar.value.T, **kwargs)
        if (len(self.collections) == 1 and
                isinstance(mesh.norm, colors.LogNorm)):
            cmap = mesh.get_cmap()
            try:  # only listed colormaps have cmap.colors
                cmap.set_bad(cmap.colors[0])
            except AttributeError:
                pass
        return mesh


register_projection(FrequencySeriesAxes)


class FrequencySeriesPlot(Plot):
    """`Figure` for displaying a `~gwpy.frequencyseries.FrequencySeries`
    """
    _DefaultAxesClass = FrequencySeriesAxes

    def __init__(self, *series, **kwargs):
        kwargs.setdefault('projection', self._DefaultAxesClass.name)
        # extract custom keyword arguments
        sep = kwargs.pop('sep', False)
        xscale = kwargs.pop(
            'xscale', kwargs.pop('logx', True) and 'log' or 'linear')
        yscale = kwargs.pop(
            'yscale', kwargs.pop('logy', True) and 'log' or 'linear')
        sharex = kwargs.pop('sharex', False)
        sharey = kwargs.pop('sharey', False)
        # separate custom keyword arguments
        # pylint: disable=unbalanced-tuple-unpacking
        axargs, plotargs = self._parse_kwargs(kwargs)
        axargs['xscale'] = xscale
        axargs['yscale'] = yscale

        # initialise figure
        super(FrequencySeriesPlot, self).__init__(**kwargs)

        # plot data
        axesdata = self._get_axes_data(series, sep=sep)
        for data in axesdata:
            ax = self._add_new_axes(**axargs)
            for series in data:
                ax.plot(series, **plotargs)
            if 'sharex' not in axargs and sharex is True:
                axargs['sharex'] = ax
            if 'sharey' not in axargs and sharey is True:
                axargs['sharey'] = ax

        axargs.pop('sharex', None)
        axargs.pop('sharey', None)
        axargs.pop('projection', None)

        for ax in self.axes:
            # format axes
            for key, val in axargs.items():
                getattr(ax, 'set_%s' % key)(val)
            # set grid
            if ax.get_xscale() in ['log']:
                ax.grid(True, axis='x', which='both')
            if ax.get_yscale() in ['log']:
                ax.grid(True, axis='y', which='both')
