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

""" Coherence plots
"""
from CliProduct import CliProduct

class Spectrogram(CliProduct):

    def get_action(self):
        """Return the string used as "action" on command line."""
        return 'Spectrogram'

    def init_cli(self, parser):
        """Set up the argument list for this product"""
        self.arg_chan1(parser)
        self.arg_freq(parser)
        self.arg_time(parser)
        self.arg_imag(parser)
        self.arg_plot(parser)
        return

    def get_ylabel(self, args):
        """Default text for y-axis label"""
        return r'$\mathrm{log_{10}  ASD}$ $\left( \frac{\mathrm{Counts}}{\sqrt{\mathrm{Hz}}}\right)$'

    def get_max_datasets(self):
        """Spectrogram only handles 1 at a time"""
        return 1

    def is_image(self):
        """This plot is image type"""
        return True

    def freq_is_y(self):
        """This plot puts frequency on the y-axis of the image"""
        return True

    def get_title(self, args):
        """Start of default super title, first channel is appended to it"""
        return 'Spectrogram: '

    def gen_plot(self, arg_list):
        """Generate the plot from time series and arguments"""
        self.is_freq_plot = True

        from numpy import min,max,percentile

        secpfft = 1
        if arg_list.secpfft:
            secpfft = float(arg_list.secpfft)
        ovlp = 0.5
        if arg_list.overlap:
            ovlp = float(arg_list.overlap)
        self.secpfft = secpfft
        self.overlap = ovlp

        ovlap = secpfft*ovlp
        nfft = self.dur/(secpfft - ovlap)
        stride = int(nfft/(self.width * 0.8) )
        if stride < 1:
            stride = 1
        stride = stride * secpfft
        fs = self.timeseries[0].sample_rate

        specgram = self.timeseries[0].spectrogram(stride, fftlength=secpfft, overlap=ovlap) ** (1/2.)


        # set default frequency limits
        self.fmax = fs.value / 2
        self.fmin = 1 / secpfft

        # default time axis
        self.xmin = self.timeseries[0].times.data.min()
        self.xmax = self.timeseries[0].times.data.max()

        # set intensity (color) limits
        imin = min(specgram).value
        imax = max(specgram).value
        if arg_list.lo:
            lo = float(arg_list.lo) * 100
            imin = percentile(specgram,lo)
        if arg_list.up:
            up = float(arg_list.up) * 100
            imax = percentile(specgram,up)

        if arg_list.lincolors:
            self.plot = specgram.plot(vmin=imin, vmax=imax)
            self.scaleText = r'ASD $\left( \frac{\mathrm{Counts}}{\sqrt{\mathrm{Hz}}}\right)$'
        else:
            self.plot = specgram.plot(norm='log',vmin=imin, vmax=imax)
            self.scaleText = r'log_10 ASD $\left(\frac{\mathrm{Counts}}{\sqrt{\mathrm{Hz}}}\right)$'
        return
