#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) Joseph Areeda (2015)
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
#

"""Coherence spectrogram
"""

try:
    from matplotlib.cm import plasma
except ImportError:
    DEFAULT_CMAP = None
else:
    DEFAULT_CMAP = plasma.name

from .spectrogram import Spectrogram

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Coherencegram(Spectrogram):
    """Plot the coherence-spectrogram comparing two time series
    """
    MIN_DATASETS = 2
    MAX_DATASETS = 2
    action = 'coherencegram'

    def _finalize_arguments(self, args):
        if args.color_scale is None:
            args.color_scale = 'linear'
        if args.color_scale == 'linear':
            if args.imin is None:
                args.imin = 0.
            if args.imax is None:
                args.imax = 1.
        if args.cmap is None:
            args.cmap = DEFAULT_CMAP
        return super(Coherencegram, self)._finalize_arguments(args)

    def get_ylabel(self):
        """Text for y-axis label
        """
        return 'Frequency (Hz)'

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return "Coherence spectrogram: {0} vs {1}".format(*self.chan_list)

    def get_color_label(self):
        if self.args.norm:
            return 'Normalized to {}'.format(self.args.norm)
        return 'Coherence'

    def get_stride(self):
        fftlength = float(self.args.secpfft)
        overlap = self.args.overlap  # fractional overlap
        return max(self.duration / (self.width * 0.8),
                   fftlength * (1 + (1-overlap)*32),
                   fftlength * 2)

    def get_spectrogram(self):
        args = self.args
        fftlength = float(args.secpfft)
        overlap = args.overlap  # fractional overlap
        stride = self.get_stride()
        self.log(2, "Calculating coherence spectrogram, "
                    "secpfft: %s, overlap: %s" % (fftlength, overlap))

        if overlap is not None:  # overlap in seconds
            overlap *= fftlength

        return self.timeseries[0].coherence_spectrogram(
            self.timeseries[1], stride, fftlength=fftlength, overlap=overlap,
            window=args.window)
