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
`~gwpy.data`
"""

import copy
import warnings

import numpy

from matplotlib.projections import register_projection
from matplotlib import (cm, colors)

from . import (tex, rcParams)
from .utils import *
from .core import Plot
from .axes import Axes
from .decorators import auto_refresh
from ..frequencyseries import (FrequencySeries, SpectralVariance)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class FrequencySeriesAxes(Axes):
    """Custom `Axes` for a :class:`~gwpy.plotter.FrequencySeriesPlot`.
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
            a single :class:`~gwpy.frequencyseries.FrequencySeries`
            (or sub-class) or standard (x, y) data arrays
        kwargs
            keyword arguments applicable to :meth:`~matplotib.axes.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        if len(args) == 1 and isinstance(args[0], FrequencySeries):
            return self.plot_spectrum(*args, **kwargs)
        elif len(args) == 1 and isinstance(args[0], SpectralVariance):
            return self.plot_variance(*args, **kwargs)
        else:
            return super(FrequencySeriesAxes, self).plot(*args, **kwargs)

    @auto_refresh
    def plot_spectrum(self, spectrum, **kwargs):
        """Plot a :class:`~gwpy.frequencyseries.FrequencySeries` onto these axes

        Parameters
        ----------
        spectrum : :class:`~gwpy.frequencyseries.FrequencySeries`
            data to plot
        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        Line2D
            the :class:`~matplotlib.lines.Line2D` for this line layer

        See Also
        --------
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        if tex.USE_TEX:
            kwargs.setdefault('label', tex.label_to_latex(spectrum.name))
        else:
            kwargs.setdefault('label', spectrum.name)
        if not kwargs.get('label', True):
            kwargs.pop('label')
        line = self.plot(spectrum.frequencies.value, spectrum.value, **kwargs)
        if len(self.lines) == 1:
            try:
                self.set_xlim(spectrum.frequencies[0].value,
                              spectrum.frequencies[-1].value +
                              spectrum.df.value)
            except IndexError:
                pass
        if 'label' in kwargs:
            self.legend()
        return line

    @auto_refresh
    def plot_spectrum_mmm(self, mean_, min_=None, max_=None, alpha=0.1,
                          **kwargs):
        """Plot a `FrequencySeries` onto these axes, with (min, max) shaded
        regions

        The `mean_` `FrequencySeries` is plotted normally, while the `min_`
        and `max_ spectra are plotted lightly below and above,
        with a fill between them and the mean_.

        Parameters
        ----------
        mean_ : :class:`~gwpy.frequencyseries.FrequencySeries
            data to plot normally
        min_ : :class:`~gwpy.frequencyseries.FrequencySeries
            first data set to shade to mean_
        max_ : :class:`~gwpy.frequencyseries.FrequencySeries
            second data set to shade to mean_
        alpha : `float`, optional
            weight of filled region, ``0.0`` for transparent through ``1.0``
            opaque
        **kwargs
            any other keyword arguments acceptable for
            :meth:`~matplotlib.Axes.plot`

        Returns
        -------
        artists : `tuple`
            a 5-tuple containing (Line2d for mean_, `Line2D` for min_,
            `PolyCollection` for min_ shading, `Line2D` for max_, and
            `PolyCollection` for max_ shading)

        See Also
        --------
        :meth:`matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        # plot mean
        line1 = self.plot_spectrum(mean_, **kwargs)[0]
        # plot min and max
        kwargs.pop('label', None)
        color = kwargs.pop('color', line1.get_color())
        linewidth = kwargs.pop('linewidth', line1.get_linewidth()) / 10
        if min_ is not None:
            a = self.plot(min_.frequencies.value, min_.value, color=color,
                          linewidth=linewidth, **kwargs)
            if alpha:
                b = self.fill_between(min_.frequencies.value, mean_.value,
                                      min_.value, alpha=alpha, color=color,
                                      rasterized=kwargs.get('rasterized'))
            else:
                b = None
        else:
            a = b = None
        if max_ is not None:
            c = self.plot(max_.frequencies.value, max_.value, color=color,
                          linewidth=linewidth, **kwargs)
            if alpha:
                d = self.fill_between(max_.frequencies.value, mean_.value,
                                      max_.value, alpha=alpha, color=color,
                                      rasterized=kwargs.get('rasterized'))
            else:
                d = None
        else:
            c = d = None
        return line1, a, b, c, d

    @auto_refresh
    def plot_variance(self, specvar, norm='log', **kwargs):
        """Plot a :class:`~gwpy.frequencyseries.SpectralVariance` onto
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
        MeshGrid
            the :class:`~matplotlib.collections.MeshGridD` for this layer

        See Also
        --------
        :meth:`matplotlib.axes.Axes.pcolormesh`
            for a full description of acceptable ``*args` and ``**kwargs``
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
        X, Y = numpy.meshgrid(x, y, copy=False, sparse=True)
        mesh = self.pcolormesh(X, Y, specvar.value.T, **kwargs)
        if len(self.collections) == 1:
            self.set_yscale('log', nonposy='mask')
            self.set_xlim(x[0], x[-1])
            self.set_ylim(y[0], y[-1])
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
        axargs, plotargs = self._parse_kwargs(kwargs)

        # initialise figure
        super(FrequencySeriesPlot, self).__init__(**kwargs)

        # plot data
        x0 = []
        axesdata = self._get_axes_data(series, sep=sep)
        for data in axesdata:
            ax = self._add_new_axes(**axargs)
            for fs in data:
                ax.plot(fs, **plotargs)
            x0.append(min([fs.df.value for fs in data]))
            if 'sharex' not in axargs and sharex is True:
                axargs['sharex'] = ax
            if 'sharey' not in axargs and sharey is True:
                axargs['sharey'] = ax
        if sharex:
            x0 = [min(x0)]*len(x0)
        axargs.pop('sharex', None)
        axargs.pop('sharey', None)
        axargs.pop('projection', None)

        for i, ax in enumerate(self.axes):
            # format axes
            for key, val in axargs.iteritems():
                getattr(ax, 'set_%s' % key)(val)
            # fix log frequency scale with f0 = DC
            if xscale in ['log']:
                xlim = list(ax.get_xlim())
                if not xlim[0]:
                    xlim[0] = x0[i]
                ax.set_xlim(*xlim)
            # set axis scales
            ax.set_xscale(xscale)
            ax.set_yscale(yscale)


# -- deprecated classes

class SpectrumPlot(FrequencySeriesPlot):
    def __init__(self, *args, **kwargs):
        warnings.warn("The SpectrumPlot object was replaced by the "
                      "FrequencySeriesPlot, and will be removed in an "
                      "upcoming release.", DeprecationWarning)
        super(SpectrumPlot, self).__init__(*args, **kwargs)


class SpectrumAxes(FrequencySeriesAxes):
    name = 'spectrum'

    def __init__(self, *args, **kwargs):
        warnings.warn("The SpectrumAxes object was replaced by the "
                      "FrequencySeriesAxes, and will be removed in an "
                      "upcoming release.", DeprecationWarning)
        super(SpectrumAxes, self).__init__(*args, **kwargs)

register_projection(SpectrumAxes)
