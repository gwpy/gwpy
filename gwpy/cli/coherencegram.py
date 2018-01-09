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

"""Coherence plots
"""

from numpy import percentile

from .cliproduct import CliProduct

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Coherencegram(CliProduct):
    """Plot the coherence-spectrogram comparing two time series
    """

    def get_action(self):
        """Return the string used as "action" on command line.
        """
        return 'coherencegram'

    def init_cli(self, parser):
        """Set up the argument list for this product
        """
        self.arg_chan2(parser)
        self.arg_freq2(parser)
        self.arg_ax_linx(parser)
        self.arg_ax_ylf(parser)
        self.arg_ax_intlin(parser)
        self.arg_imag(parser)
        self.arg_plot(parser)

    def get_max_datasets(self):
        """Coherencegram only handles 1 set of 2 at a time
        """
        return 2

    def get_min_datasets(self):
        """Coherence requires 2 datasets to calculate
        """
        return 2

    def is_image(self):
        """This plot is image type
        """
        return True

    def freq_is_y(self):
        """This plot puts frequency on the y-axis of the image
        """
        return True

    def get_ylabel(self, args):
        """Text for y-axis label
        """
        return 'Frequency (Hz)'

    def get_title(self):
        """Start of default super title, first channel is appended to it
        """
        return "Coherence spectrogram: "

    def get_color_label(self):
        return self.scale_text

    def get_sup_title(self):
        """We want both channels in the title
        """
        sup = self.get_title() + self.timeseries[0].channel.name
        sup += " vs. " + self.timeseries[1].channel.name
        return sup

    def gen_plot(self, arg_list):
        """Generate the plot from time series and arguments
        """
        self.is_freq_plot = True

        secpfft = 0.5
        if arg_list.secpfft:
            secpfft = float(arg_list.secpfft)
        ovlp_frac = 0.9
        if arg_list.overlap:
            ovlp_frac = float(arg_list.overlap)
        self.secpfft = secpfft
        self.overlap = ovlp_frac

        ovlap_sec = secpfft*ovlp_frac
        stride = int(self.dur/(self.width * 0.8))

        stride = max(stride, secpfft+(1-ovlp_frac)*32)
        stride = max(stride, secpfft*2)

        coh = self.timeseries[0].coherence_spectrogram(
            self.timeseries[1], stride, fftlength=secpfft, overlap=ovlap_sec)
        norm = False
        if arg_list.norm:
            coh = coh.ratio('mean')
            norm = True

        # set default frequency limits
        self.fmax = coh.band[1]
        self.fmin = 1 / secpfft

        # default time axis
        self.xmin = self.timeseries[0].times.value.min()
        self.xmax = self.timeseries[0].times.value.max()

        # set intensity (color) limits
        if arg_list.imin:
            lo = float(arg_list.imin)
        elif norm:
            lo = 0.5
        else:
            lo = 0.01
        if norm or arg_list.nopct:
            imin = lo
        else:
            imin = percentile(coh, lo*100)

        if arg_list.imax:
            up = float(arg_list.imax)
        elif norm:
            up = 2
        else:
            up = 100
        if norm or arg_list.nopct:
            imax = up
        else:
            imax = percentile(coh, up)

        pltargs = dict()
        if arg_list.cmap:
            pltargs['cmap'] = arg_list.cmap

        pltargs['vmin'] = imin
        pltargs['vmax'] = imax

        # plot the thing
        if norm:
            self.scale_text = 'Normalized to mean'
        elif arg_list.logcolors:
            pltargs['norm'] = 'log'
            self.scale_text = r'log_10 Coherence'
        else:
            self.scale_text = r'Coherence'

        self.plot = coh.plot(**pltargs)
        self.result = coh

        # pass the scaling to the annotater
        self.imin = imin
        self.imax = imax
