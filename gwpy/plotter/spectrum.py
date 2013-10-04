# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""This module defines plotting classes for the data series defined in
`~gwpy.data`
"""

import re
import warnings

from matplotlib.projections import register_projection

from .utils import *
from .core import Plot
from .axes import Axes
from ..spectrum import Spectrum

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

register_projection(SpectrumAxes)
