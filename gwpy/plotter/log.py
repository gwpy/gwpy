# coding=utf-8
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

from math import (ceil, floor, modf)

import numpy

from matplotlib import rcParams
from matplotlib.scale import (LogScale, register_scale)
from matplotlib.ticker import (is_decade, LogFormatterMathtext, LogLocator)

from .. import version

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version


class GWpyLogFormatterMathtext(LogFormatterMathtext):
    """Minor edit to standard LogFormatterMathtext.
    """
    def __call__(self, x, pos=None):
        if 0.01 <= x <= 100:
            if numpy.isclose(x, int(x)):
                x = int(x)
            usetex = rcParams['text.usetex']
            if usetex:
                return '$%s$' % x
            else:
                return '$\mathdefault{%s}$' % x
        else:
            return super(GWpyLogFormatterMathtext, self).__call__(x, pos=pos)


class MinorLogFormatterMathtext(GWpyLogFormatterMathtext):
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
    def __call__(self, x, pos=None):
        if is_decade(x, self._base):
            return super(MinorLogFormatterMathtext, self).__call__(x, pos=pos)
        else:
            return super(CombinedLogFormatterMathtext, self).__call__(x,
                                                                      pos=pos)


class GWpyLogScale(LogScale):
    """Enhanced version of the matplotlib `LogScale`.
    """
    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(LogLocator(self.base))
        axis.set_major_formatter(GWpyLogFormatterMathtext(self.base))
        axis.set_minor_locator(LogLocator(self.base, self.subs))
        axis.set_minor_formatter(MinorLogFormatterMathtext(self.base))

register_scale(GWpyLogScale)
