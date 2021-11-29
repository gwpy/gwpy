# -*- coding: utf-8 -*-
# Copyright (C) Joseph Areeda (2015-2020)
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

"""Coherence spectrogram
"""

from .spectrogram import Spectrogram

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Coherencegram(Spectrogram):
    """Plot the coherence-spectrogram comparing two time series
    """
    DEFAULT_CMAP = "plasma"
    MIN_DATASETS = 2
    MAX_DATASETS = 2
    action = 'coherencegram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type (e.g. m-trend)
        if ',' in self.ref_chan:
            self.ref_chan = self.ref_chan.split(',')[0]

    @classmethod
    def arg_channels(cls, parser):
        group = super().arg_channels(parser)
        group.add_argument('--ref', help='Reference channel against which '
                                         'others will be compared')
        return group

    def _finalize_arguments(self, args):
        if args.color_scale is None:
            args.color_scale = 'linear'
        if args.color_scale == 'linear':
            if args.imin is None:
                args.imin = 0.
            if args.imax is None:
                args.imax = 1.
        return super()._finalize_arguments(args)

    def get_ylabel(self):
        """Text for y-axis label
        """
        return 'Frequency (Hz)'

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        a, b = self.chan_list
        return f"Coherence spectrogram: {a} vs {b}"

    def get_color_label(self):
        if self.args.norm:
            return f'Normalized to {self.args.norm}'
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
                    f"secpfft: {fftlength}, overlap: {overlap}")

        if overlap is not None:  # overlap in seconds
            overlap *= fftlength

        if self.timeseries[0].name == self.ref_chan:
            ref = 0
            other = 1
        else:
            ref = 1
            other = 0
        return self.timeseries[ref].coherence_spectrogram(
            self.timeseries[other], stride, fftlength=fftlength,
            overlap=overlap, window=args.window)
