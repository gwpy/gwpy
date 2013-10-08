# Licensed under a 3-clause BSD style license - see LICENSE.rst

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
    """Plot data from a `~gwpy.data.Spectrogram` object
    """
    def __init__(self, *spectrograms, **kwargs):
        self._logy = False

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

        # set matching epoch for each set of axes
        if len(spectrograms) and not sep:
            span = SegmentList([spec.span for spec in spectrograms]).extent()
            for ax in self.axes:
                ax.set_xlim(*span)
            for ax in self.axes[:-1]:
                ax.set_xlabel("")
