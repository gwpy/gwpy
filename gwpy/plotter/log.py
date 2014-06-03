# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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

"""This module redefines the matplotlib log scale to give a more
helpful set of major and minor ticks.
"""

from __future__ import division

from math import (ceil, floor, modf)

import numpy

from matplotlib import rcParams
from matplotlib.axis import XAxis
from matplotlib.scale import (LogScale, register_scale)
from matplotlib.ticker import (is_decade, LogFormatterMathtext, LogLocator)

from .tex import float_to_latex
from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class GWpyLogFormatterMathtext(LogFormatterMathtext):
    """Format values for log axis.

    This `Formatter` extends the standard
    :class:`~matplotlib.ticker.LogFormatterMathtext` to print numbers
    in the range [0.01, 1000) normally, and all others via the
    `LogFormatterMathtext` output.
    """
    def __call__(self, x, pos=None):
        usetex = rcParams['text.usetex']
        if float(x).is_integer():
            x = int(x)
        if 0.01 <= x < 1000:
            if usetex:
                return '$%s$' % x
            else:
                return '$\mathdefault{%s}$' % x
        elif usetex:
            return '$%s$' % float_to_latex(x, '%.2g')
        else:
            return super(GWpyLogFormatterMathtext, self).__call__(x, pos=pos)


class MinorLogFormatterMathtext(GWpyLogFormatterMathtext):
    """Format minor tick labels on a log scale.

    This `Formatter` conditionally formats minor tick labels based on the
    number of major ticks visible when the formatter is called, either

    - no minor tick labels if two or more major ticks are visible
    - half-decade tick labels (0.5, 5, 50, ...) if only one major tick
      is visible
    - otherwise all minor ticks
    """

    def __init__(self, *args, **kwargs):
        kwargs.setdefault('labelOnlyBase', False)
        super(MinorLogFormatterMathtext, self).__init__(*args, **kwargs)

    def __call__(self, x, pos=None):
        """Format the minor tick label if less than 2 visible major ticks.
        """
        viewlim = self.axis.get_view_interval()
        loglim = numpy.log10(viewlim)
        majticks = numpy.arange(ceil(loglim[0]), floor(ceil(loglim[1])), dtype=int)
        nticks = majticks.size
        halfdecade = nticks == 1 and modf(loglim[0])[0] < 0.7
        # if already two major ticks, don't need minor labels
        if nticks >= 2 or (halfdecade and not is_decade(x * 2, self._base)):
            return ''
        else:
            return super(MinorLogFormatterMathtext, self).__call__(x, pos=pos)


class CombinedLogFormatterMathtext(MinorLogFormatterMathtext):
    """Format major and minor ticks with a single `Formatter`.

    This is just a swap between the `MinorLogFormatterMathtext` and
    the GWpyLogFormatterMathtext` depending on whether the tick in
    question is a decade (0.1, 1, 10, 100, ...) or not.
    """
    def __call__(self, x, pos=None):
        if is_decade(x, self._base):
            return super(MinorLogFormatterMathtext, self).__call__(x, pos=pos)
        else:
            return super(CombinedLogFormatterMathtext, self).__call__(x,
                                                                      pos=pos)


class GWpyLogScale(LogScale):
    """GWpy version of the matplotlib `LogScale`.

    This scale overrides the default to use the new GWpy formatters
    for major and minor ticks.
    """
    def set_default_locators_and_formatters(self, axis):
        if isinstance(axis, XAxis):
            axis.set_tick_params(which='both', pad=7)
            axis.labelpad = 8
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(GWpyLogFormatterMathtext(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(MinorLogFormatterMathtext(self.base))

register_scale(GWpyLogScale)
