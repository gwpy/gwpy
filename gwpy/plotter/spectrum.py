# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines plotting classes for the data series defined in
`~gwpy.data`
"""

import re
import warnings
import copy

from matplotlib.projections import register_projection
from matplotlib.projections import register_projection
from matplotlib import (cm, colors)

from .utils import *
from .core import Plot
from .axes import Axes
from ..spectrum import (Spectrum, SpectralVariance)

from ..version import version as __version__
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


class SpectrumPlot(Plot):
    """Plot data from a LAL TimeSeries object
    """
    def __init__(self, *series, **kwargs):
        # extract plotting keyword arguments
        plotargs = dict()
        plotargs["linewidth"] = kwargs.pop("linewidth", 2)
        plotargs["color"] = kwargs.pop("color", "black")
        plotargs["linestyle"] = kwargs.pop("linestyle", "-")

        # initialise figure
        super(SpectrumPlot, self).__init__(**kwargs)
        self._series = []

        # plot time series
        for spectrum in series:
            self._series.append(spectrum)
            self.add_spectrum(spectrum)
        if len(series) == 1:
            self.add_label_unit(series[0].frequencies.unit, axis="x")
            self.add_label_unit(series[0].unit, axis="y")
        if len(series):
            self.logx = self.logy = True
            self.axes.autoscale_view()


class SpectrumAxes(Axes):
    """Extension of the basic matplotlib :class:`~matplotlib.axes.Axes`
    specialising in frequency-series display
    """
    name = 'spectrum'

    # -------------------------------------------
    # GWpy class plotting methods

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
        :meth:`~matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        if len(args) == 1 and isinstance(args[0], Spectrum):
            return self.plot_spectrum(*args, **kwargs)
        elif len(args) == 1 and isinstance(args[0], SpectralVariance):
            return self.plot_variance(*args, **kwargs)
        else:
            return super(SpectrumAxes, self).plot(*args, **kwargs)

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
        :meth:`~matplotlib.axes.Axes.plot`
            for a full description of acceptable ``*args` and ``**kwargs``
        """
        kwargs.setdefault('label', spectrum.name)
        line = self.plot(spectrum.frequencies, spectrum.data, **kwargs)
        if len(self.lines) == 1:
            self.set_xlim(spectrum.frequencies[0],
                          spectrum.frequencies[-1] + spectrum.df.value)
        return line

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
        :meth:`~matplotlib.axes.Axes.pcolormesh`
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
