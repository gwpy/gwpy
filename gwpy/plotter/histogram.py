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

import numpy

from glue import iterutils

from .core import Plot

__author__ = "Duncan M. Macleod <duncan.macleod@ligo.org>"
__version__ = ""
__date__ = ""

__all__ = ["LineHistogram", "BarHistogram", "StepHistogram"]


class Histogram(Plot):
    """A plot showing a histogram of data
    """
    def __init__(self, *data, **kwargs):
        bins = kwargs.pop("bins", 30)
        logspace = kwargs.pop("logspace", False)
        orientation = kwargs.pop("orientation", "vertical")
        super(Histogram, self).__init__(**kwargs)
        self._datasets = []
        self._kwargsets = []
        self._logspace = logspace

        for ds in data:
            self.add_dataset(ds)

        if len(data):
            self.make(bins=bins, logspace=logspace, orientation=orientation)
        if orientation == "vertical":
            self.logx = logspace
        elif orientation == "horizontal":
            self.logy = logspace

    def add_dataset(self, data, **kwargs):
        self._datasets.append(data)
        self._kwargsets.append(kwargs)

    def _make(self, style, bins=30, logspace=False, **kwargs):
        # set bins
        if isinstance(bins, int):
            binlow, binhigh = common_limits(self._datasets)
            bins = self.bins(binlow, binhigh, bins, logspace=logspace)
        width = numpy.diff(bins)
        orientation = kwargs.pop("orientation", "vertical")
        # histogram each data set, adding some bars
        for (dataset, datakwargs) in zip(self._datasets, self._kwargsets):
            y,x = numpy.histogram(dataset, bins=bins)
            allargs = dict(list(kwargs.items()) + list(datakwargs.items()))
            if style == "line":
                if orientation == "vertical":
                    self.add_line(x[:-1]+ width/2., y, **allargs)
                else:
                    self.add_line(y, x[:-1]+ width/2., **allargs)
            elif style == "step":
                step_x = numpy.vstack((x[:-1],
                                       x[:-1]+width)).reshape((-1), order="F")
                step_y = numpy.vstack((y, y)).reshape((-1), order="F")
                if orientation == "vertical":
                    self.add_line(step_x, step_y, **allargs)
                else:
                    self.add_line(step_y, step_x, **allargs)
            elif style == "bar":
                self.add_bars(x[:-1], y, width=width, orientation=orientation,
                              **allargs)

    def bins(self, lower, upper, number, logspace=False):
        if logspace:
            bins = numpy.logspace(numpy.log10(lower), numpy.log10(upper),
                                  number+1, endpoint=True)
        else:
            bins = numpy.linspace(lower, upper, number+1, endpoint=True)
        return bins


class LineHistogram(Histogram):
    """A plot showing a line histogram of data
    """
    def make(self, bins=30, logspace=False, **kwargs):
        self._make("line", bins=bins, logspace=logspace, **kwargs)

class BarHistogram(Histogram):
    """A plot showing a bar histogram of data
    """
    def make(self, bins=30, logspace=False, **kwargs):
        self._make("bar", bins=bins, logspace=logspace, **kwargs)


class StepHistogram(Histogram):
    """A plot showing a bar histogram of data
    """
    def make(self, bins=30, logspace=False, **kwargs):
        self._make("step", bins=bins, logspace=logspace, **kwargs)


def common_limits(data_sets, default_min=0, default_max=0):
    """Find the global maxima and minima of a list of datasets
    """
    max_stat = max(list(iterutils.flatten(data_sets)) + [-numpy.inf])
    min_stat = min(list(iterutils.flatten(data_sets)) + [numpy.inf])
    if numpy.isinf(-max_stat):
        max_stat = default_max
    if numpy.isinf(min_stat):
        min_stat = default_min
    return min_stat, max_stat
