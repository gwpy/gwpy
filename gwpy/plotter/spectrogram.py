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

"""An extension of the Plot class for handling Spectrograms
"""

import re
import numpy
import math

from ..time import Time
from ..segments import SegmentList
from .timeseries import TimeSeriesPlot

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['SpectrogramPlot']


class SpectrogramPlot(TimeSeriesPlot):
    """`Figure` for displaying a :class:`~gwpy.spectrogram.Spectrogram`.
    """
    def __init__(self, *spectrograms, **kwargs):
        """Generate a new `SpectrogramPlot`
        """
        # extract plotting keyword arguments
        plotargs = dict()
        plotargs['vmin'] = kwargs.pop('vmin', None)
        plotargs['vmax'] = kwargs.pop('vmax', None)
        plotargs['norm'] = kwargs.pop('norm', None)
        plotargs['cmap'] = kwargs.pop('cmap', None)
        sep = kwargs.pop('sep', True)

        # initialise figure
        super(SpectrogramPlot, self).__init__(**kwargs)

        # plot data
        for i,spectrogram in enumerate(spectrograms):
            self.add_spectrogram(spectrogram, newax=sep, **plotargs)
            self.axes[-1].fmt_ydata = lambda f: ('%s %s'
                                                 % (f, spectrogram.yunit))
            self.axes[-1].set_ylabel('Frequency [%s]' % spectrogram.yunit)

        # set matching epoch for each set of axes
        if len(spectrograms) and not sep:
            span = SegmentList([spec.span for spec in spectrograms]).extent()
            for ax in self.axes:
                ax.set_xlim(*span)
            for ax in self.axes[:-1]:
                ax.set_xlabel("")
