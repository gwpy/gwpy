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

from matplotlib.projections import register_projection
from matplotlib import (cm, colors)

from . import tex
from .utils import *
from .core import Plot
from .axes import Axes
from .decorators import auto_refresh
from ..spectrum import (Spectrum, SpectralVariance)

from .. import version
__version__ = version.version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class SpectrumAxes(Axes):
    """Custom `Axes` for a :class:`~gwpy.plotter.SpectrumPlot`.
    """
    name = 'spectrum'

    # -------------------------------------------
    # GWpy class plotting methods

    @auto_refresh
    def plot(self, *args, **kwargs):
        """Plot data onto these Axes.

        Parameters
        ----------
        args
            a single :class:`~gwpy.spectrum.core.spectrum` (or sub-class)
            or standard (x, y) data arrays
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
        if len(args) == 1 and isinstance(args[0], Spectrum):
            return self.plot_spectrum(*args, **kwargs)
        elif len(args) == 1 and isinstance(args[0], SpectralVariance):
            return self.plot_variance(*args, **kwargs)
        else:
            return super(SpectrumAxes, self).plot(*args, **kwargs)

    @auto_refresh
    def plot_spectrum(self, spectrum, **kwargs):
        """Plot a :class:`~gwpy.spectrum.core.Spectrum` onto these axes

        Parameters
        ----------
        spectrum : :class:`~gwpy.spectrum.core.Spectrum`
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
        line = self.plot(spectrum.frequencies, spectrum.data, **kwargs)
        if len(self.lines) == 1:
            try:
                self.set_xlim(spectrum.frequencies[0],
                              spectrum.frequencies[-1] + spectrum.df.value)
            except IndexError:
                pass
        if 'label' in kwargs:
            self.legend()
        return line

    @auto_refresh
    def plot_spectrum_mmm(self, mean_, min_=None, max_=None, alpha=0.1,
                          **kwargs):
        """Plot a `Spectrum` onto these axes, with (min, max) shaded
        regions

        The `mean_` `Spectrum` is plotted normally, while the `min_`
        and `max_ spectra are plotted lightly below and above,
        with a fill between them and the mean_.

        Parameters
        ----------
        mean_ : :class:`~gwpy.spectrum.core.Spectrum
            data to plot normally
        min_ : :class:`~gwpy.spectrum.core.Spectrum
            first data set to shade to mean_
        max_ : :class:`~gwpy.spectrum.core.Spectrum
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
            a = self.plot(min_.frequencies, min_.data, color=color,
                          linewidth=linewidth, **kwargs)
            if alpha:
                b = self.fill_between(min_.frequencies, mean_.data, min_.data,
                                      alpha=alpha, color=color)
            else:
                b = None
        else:
            a = b = None
        if max_ is not None:
            c = self.plot(max_.frequencies, max_.data, color=color,
                          linewidth=linewidth, **kwargs)
            if alpha:
                d = self.fill_between(max_.frequencies, mean_.data, max_.data,
                                      alpha=alpha, color=color)
            else:
                d = None
        else:
            c = d = None
        return line1, a, b, c, d

    @auto_refresh
    def plot_variance(self, specvar, **kwargs):
        """Plot a :class:`~gwpy.spectrum.hist.SpectralVariance` onto
        these axes

        Parameters
        ----------
        spectrum : class:`~gwpy.spectrum.hist.SpectralVariance`
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
        cmap = kwargs.pop('cmap', None)
        if cmap is None:
            cmap = copy.deepcopy(cm.jet)
            cmap.set_bad(cmap(0.0))
        kwargs['cmap'] = cmap
        norm = kwargs.pop('norm', None)
        if norm is None and specvar.logy:
            vmin = kwargs.pop('vmin', None)
            vmax = kwargs.pop('vmax', None)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax)
        kwargs['norm'] = norm
        x = numpy.concatenate((specvar.frequencies.data,
                               [specvar.x0.value +
                                specvar.dx.value * specvar.shape[0]]))
        y = specvar.bins
        X, Y = numpy.meshgrid(x, y)
        mesh = self.pcolormesh(X, Y, specvar.data.T, **kwargs)
        if len(self.collections) == 1:
            if specvar.logy:
                self.set_yscale('log', nonposy='mask')
            self.set_xlim(x[0], x[-1])
            self.set_ylim(y[0], y[-1])
        return mesh


register_projection(SpectrumAxes)


class SpectrumPlot(Plot):
    """`Figure` for displaying a :class:`~gwpy.spectrum.core.Spectrum`.
    """
    _DefaultAxesClass = SpectrumAxes

    def __init__(self, *series, **kwargs):
        # extract plotting keyword arguments
        plotargs = dict()
        for key in ['linewidth', 'linestyle', 'color', 'label', 'cmap',
                    'vmin', 'vmax']:
            if key in kwargs:
                plotargs[key] = kwargs.pop(key)
        sep = kwargs.pop('sep', False)
        logx = kwargs.pop('logx', True)
        logy = kwargs.pop('logy', True)

        # initialise figure
        super(SpectrumPlot, self).__init__(**kwargs)
        self._series = []

        # plot time series
        for i, spectrum in enumerate(series):
            self.add_spectrum(spectrum, newax=sep, **plotargs)
            self.axes[-1].fmt_xdata = lambda f: ('%s [%s]'
                                                 % (f, spectrum.xunit))
            self.axes[-1].fmt_ydata = lambda a: ('%s [%s]'
                                                 % (a, spectrum.unit))
            if logx:
                xlim = list(self.axes[-1].get_xlim())
                if not xlim[0]:
                    xlim[0] = spectrum.df.value
                self.axes[-1].set_xscale('log')
                self.axes[-1].set_xlim(*xlim)
            if logy:
                self.axes[-1].set_yscale('log')
