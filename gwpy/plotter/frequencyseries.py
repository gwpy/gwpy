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

from matplotlib.projections import register_projection
from matplotlib import colors

from . import text
from .decorators import auto_refresh
from .series import (SeriesPlot, SeriesAxes)
from ..frequencyseries import SpectralVariance

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['FrequencySeriesAxes', 'FrequencySeriesPlot']


class FrequencySeriesAxes(SeriesAxes):
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
        # plot spectral variance
        if len(args) == 1 and isinstance(args[0], SpectralVariance):
            return self.plot_variance(*args, **kwargs)

        return super(FrequencySeriesAxes, self).plot(*args, **kwargs)

    plot_frequencyseries = SeriesAxes.plot_series

    @auto_refresh
    def plot_frequencyseries_mmm(self, mean_, min_=None, max_=None, **kwargs):
        warnings.warn('plot_frequencyseries_mmm has been deprecated, please '
                      'use instead plot_mmm()', DeprecationWarning)
        return self.plot_mmm(mean_, min_=min_, max_=max_, **kwargs)

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
    def plot_variance(self, specvar, norm='log', **kwargs):
        """Plot a `~gwpy.frequencyseries.SpectralVariance`

        Parameters
        ----------
        specvar : class:`~gwpy.frequencyseries.SpectralVariance`
            Data to plot

        **kwargs
            Any other eyword arguments acceptable for
            :meth:`~matplotlib.axes.Axes.pcolormesh`

        Returns
        -------
        mesh : `~matplotlib.collections.QuadMesh`
            The collection that has just been added

        See Also
        --------
        matplotlib.axes.Axes.pcolormesh
            For a full description of acceptable ``*args`` and ``**kwargs``
        """
        if kwargs.pop('imshow', False):
            raise TypeError("plotting a SpectralVariance with imshow() "
                            "is not supported")

        layer = super(FrequencySeriesAxes, self).plot_array2d(
            specvar, norm=norm, imshow=False, **kwargs)

        # allow masking of (e.g.) zeros on log norm
        if (len(self.collections) + len(self.images) == 1 and
                isinstance(layer.norm, colors.LogNorm)):
            cmap = layer.get_cmap()
            try:  # only listed colormaps have cmap.colors
                cmap.set_bad(cmap.colors[0])
            except AttributeError:
                pass

        return layer


register_projection(FrequencySeriesAxes)


class FrequencySeriesPlot(SeriesPlot):
    """`Figure` for displaying a `~gwpy.frequencyseries.FrequencySeries`
    """
    _DefaultAxesClass = FrequencySeriesAxes

    def __init__(self, *series, **kwargs):
        # customise defaults
        kwargs.setdefault('xscale',
                          kwargs.pop('logx', True) and 'log' or 'linear')
        kwargs.setdefault('yscale',
                          kwargs.pop('logy', True) and 'log' or 'linear')

        # generate figures
        super(FrequencySeriesPlot, self).__init__(*series, **kwargs)

        # update grids for log scales
        for ax in self.axes:
            # set grid
            if ax.get_xscale() in ['log']:
                ax.grid(True, axis='x', which='both')
            if ax.get_yscale() in ['log']:
                ax.grid(True, axis='y', which='both')
