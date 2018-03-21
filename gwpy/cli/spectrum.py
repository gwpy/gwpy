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

"""Spectrum plots
"""

from .cliproduct import CliProduct

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Spectrum(CliProduct):
    """
    Plot the ASD spectrum of one or more time series
    """

    def get_action(self):
        """Return the string used as "action" on command line.
        """
        return 'spectrum'

    def init_cli(self, parser):
        """Set up the argument list for this product
        """
        self.arg_chan1(parser)
        self.arg_freq(parser)
        self.arg_ax_xlf(parser)
        self.arg_ax_logy(parser)
        self.arg_plot(parser)
        self.xaxis_is_freq = True

    def get_ylabel(self, args):
        """Text for y-axis label
        """
        if args.nology:
            ylabel = (r'$\mathrm{log_{10}  ASD}$ '
                      r'$\left( \frac{\mathrm{%s}}'
                      r'{\sqrt{\mathrm{Hz}}}\right)$' % self.units)
        else:
            ylabel = (r'$\mathrm{ASD}$ $\left( \frac{\mathrm{%s}}'
                      r'{\sqrt{\mathrm{Hz}}}\right)$' % self.units)
        return ylabel

    def get_title(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Spectrum: '

    def get_xlabel(self):
        xlabel = 'Frequency (Hz)'
        return xlabel

    def freq_is_y(self):
        """This plot puts frequency on the y-axis of the graph
        """
        return False

    def gen_plot(self, arg_list):
        """Generate the plot from time series and arguments
        """
        self.is_freq_plot = True

        fftlen = 1.0
        if arg_list.secpfft:
            fftlen = float(arg_list.secpfft)
        self.secpfft = fftlen
        ovlap = 0.5
        if arg_list.overlap:
            ovlap = float(arg_list.overlap)
        self.overlap = ovlap

        self.log(2, "Calculating spectrum secpfft: %.2f, overlap: %.2f" %
                 (fftlen, ovlap))
        spectra = []

        # calculate and plot the first spectrum
        spectrum = self.timeseries[0].asd(fftlen, fftlen*ovlap)
        spectra.append(spectrum)

        fs = self.timeseries[0].sample_rate.value
        self.fmin = 1/self.secpfft
        self.fmax = fs/2
        self.ymin = spectrum.value.min()
        self.ymax = spectrum.value.max()

        label = self.timeseries[0].channel.name
        if len(self.start_list) > 1:
            label += ", %s" % self.timeseries[0].epoch.gps
        spectrum.name = label
        self.plot = spectrum.plot()

        # if we have more time series calculate and add to the first plot
        if len(self.timeseries) > 1:
            for idx in range(1, len(self.timeseries)):
                specb = self.timeseries[idx].asd(fftlen, ovlap*fftlen)
                spectra.append(specb)
                fsb = self.timeseries[idx].sample_rate.value
                self.fmax = max(self.fmax, fsb/2)
                self.ymin = min(self.ymin, specb.value.min())
                self.ymax = max(self.ymax, specb.value.max())

                label = self.timeseries[idx].channel.name
                if len(self.start_list) > 1:
                    label += ", %s" % self.timeseries[idx].epoch.gps
                specb.name = label
                self.plot.add_frequencyseries(specb)
        self.log(2, ('Frequency range: [%f, %f]' % (self.fmin, self.fmax)))
        # if the specified frequency limits adjust our ymin and ymax values
        # at this point self.ymin and self.ymax represent the full spectra
        if arg_list.fmin or arg_list.fmax:
            import numpy
            mymin = self.ymax   # guaranteed to be >= anything we look at
            mymax = self.ymin   # guaranteed to be <= anything we look at
            myfmin = self.fmin
            myfmax = self.fmax
            if arg_list.fmin:
                myfmin = float(arg_list.fmin)
            if arg_list.fmax:
                myfmax = float(arg_list.fmax)

            for idx in range(0, len(spectra)):
                t = numpy.where(spectra[idx].frequencies.value >= myfmin)
                if t[0].size:
                    strt = t[0][0]
                    t = numpy.where(spectra[idx].frequencies.value >= myfmax)
                    if t[0].size:
                        stop = t[0][0]
                    else:
                        stop = spectra[idx].frequencies.size - 1
                    mymin = min(mymin,
                                numpy.min(spectra[idx].value[strt:stop]))
                    mymax = max(mymax,
                                numpy.max(spectra[idx].value[strt:stop]))

            self.ymin = mymin
            self.ymax = mymax
