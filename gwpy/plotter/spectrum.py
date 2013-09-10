# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Docstring
"""

from matplotlib import pyplot
from . import (tex, ticks)
from .core import BasicPlot
from .decorators import auto_refresh

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version


class SpectrumPlot(BasicPlot):
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
            f = spectrum.get_frequencies()
            self.add_line(f, spectrum.data,
                          label=tex.label_to_latex(spectrum.name))
        if len(series) == 1:
            self.add_label_unit(f.unit, axis="x")
            self.add_label_unit(series[0].unit, axis="y")
        self.logx = self.logy = True
        self.axes.autoscale_view()

    def add_label_unit(self, unit, axis="x"):
        attr = "%slabel" % axis
        label = getattr(self, attr)
        if not label and unit.physical_type != u'unknown':
            label = unit.physical_type.title()
        elif not label and hasattr(unit, "name"):
            label = unit.name
        if pyplot.rcParams.get("text.usetex", False):
            unitstr = tex.unit_to_latex(unit)
        else:
            unitstr = unit.to_string()
        if label:
            setattr(self, attr, "%s (%s)" % (label, unitstr))
        else:
            setattr(self, attr, unitstr)
