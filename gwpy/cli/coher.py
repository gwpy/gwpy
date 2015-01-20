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

class Coher(CliProduct):

    min_timeseries = 2
    plot = 0    # this will be a matplotlib plot after derived class makes it

    def get_action(self):
        """Return the string used as "action" on command line."""
        return 'coher'

    def init_cli(self, parser):
        """Set up the argument list for this product"""
        self.arg_chan2(parser)
        self.arg_freq(parser)
        self.arg_plot(parser)
        return

    def get_ylabel(self, args):
        """Text for y-axis label"""
        return 'Coherence'

    def get_title(self, args):
        """Start of default super title, first channel is appended to it"""
        return 'Coherence with: '

    def get_min_datasets(self):
        "Coherence requires 2 datasets for the calculation"
        return 2

    def get_xlabel(self):
        return 'Frequency (Hz)'


    def gen_plot(self, arg_list):
        """Generate the coherence plot from all time series"""
        from numpy import min, max
        self.is_freq_plot = True
        fftlen = 1
        if arg_list.secpfft:
            fftlen = float(arg_list.secpfft)
        ovlap = 0.5
        if arg_list.overlap:
            ovlap = float(arg_list.overlap)
        self.secpfft = fftlen
        self.overlap = ovlap

        # calculate and plot the first pair, note the first channel is the reference channel
        coh = self.timeseries[0].coherence(self.timeseries[1], fftlength=fftlen, overlap=ovlap*fftlen)
        #coh2 = 1 / (1-coh)
        self.fmin = min(coh.frequencies.data)
        self.fmax = max(coh.frequencies.data)
        self.ymin = 0
        self.ymax = 1

        coh.name = self.timeseries[1].channel.name
        if len(self.start_list) > 1:
            coh.name += ", %s" % self.timeseries[1].times.epoch.gps

        self.plot = coh.plot()

        # if we have more time series calculate and add to the first plot
        if len(self.timeseries) > 2:
            for idx in range(2, len(self.timeseries)):
                cohb = self.timeseries[0].coherence(self.timeseries[idx], fftlength=fftlen, overlap=ovlap*fftlen)
                cohb.name = self.timeseries[idx].channel.name
                if len(self.start_list) > 1:
                    cohb.name += ", %s" % self.timeseries[idx].times.epoch.gps
                self.plot.add_spectrum(cohb)

