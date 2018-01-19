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

from .cliproduct import CliProduct

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Coherence(CliProduct):
    """Plot coherence between a reference time series and one
    or more other time series
    """

    min_timeseries = 2
    plot = 0    # this will be a matplotlib plot after derived class makes it

    def get_action(self):
        """Return the string used as "action" on command line.
        """
        return 'coherence'

    def init_cli(self, parser):
        """Set up the argument list for this product
        """
        self.arg_chan2(parser)
        self.arg_freq(parser)
        self.arg_ax_xlf(parser)
        self.arg_ax_liny(parser)
        self.arg_plot(parser)
        self.xaxis_is_freq = True

    def get_ylabel(self, args):
        """Text for y-axis label
        """
        return 'Coherence'

    def get_title(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Coherence with: '

    def get_min_datasets(self):
        """Coherence requires 2 datasets for the calculation
        """
        return 2

    def get_xlabel(self):
        return 'Frequency (Hz)'

    def get_sup_title(self):
        """Override if default lacks critical info
        """
        return self.get_title() + self.ref_chan_name

    def gen_plot(self, arg_list):
        """Generate the coherence plot from all time series
        """
        import numpy

        self.is_freq_plot = True
        fftlen = 1
        if arg_list.secpfft:
            fftlen = float(arg_list.secpfft)
        ovlap_frac = 0.5
        if arg_list.overlap:
            ovlap_frac = float(arg_list.overlap)
        self.secpfft = fftlen
        self.overlap = ovlap_frac
        maxfs = 0

        if arg_list.ref:
            ref_name = arg_list.ref
        else:
            ref_name = self.timeseries[0].channel.name
        self.ref_chan_name = ref_name

        self.log(3, 'Reference channel: ' + ref_name)
        # we don't want to compare the reference channel to itself
        # at a different time this section checks that we have
        # something to do
        next_ts = -1
        for idx in range(0, len(self.timeseries)):
            legend_text = self.timeseries[idx].channel.name
            if (not legend_text.startswith(ref_name) and
                    self.timeseries[idx].min() !=
                    self.timeseries[idx].value.max()):
                next_ts = idx
                break
        if next_ts == -1:
            raise ValueError('No appropriate channels for '
                             'Coherence calculation')

        cohs = []
        for time_group in self.time_groups:
            ref_idx = time_group[0]
            if len(time_group) >= 2:
                # find the reference channel in this group
                for idx in range(0, len(time_group)):
                    idxp = time_group[idx]
                    if self.timeseries[idxp].channel.name.startswith(ref_name):
                        ref_idx = idxp

                maxfs = max(maxfs, self.timeseries[ref_idx].sample_rate)
                if numpy.min(self.timeseries[ref_idx]) == \
                        numpy.max(self.timeseries[ref_idx]):
                    print('Channel %s at %d has min=max,it cannot be used '
                          'as a reference channel'
                          % (self.timeseries[ref_idx].channel.name,
                             self.timeseries[ref_idx].epoch.gps))
                else:
                    for idxp in range(0, len(time_group)):
                        next_ts = time_group[idxp]
                        if next_ts != ref_idx:
                            if numpy.min(self.timeseries[next_ts]) == \
                                    numpy.max(self.timeseries[next_ts]):
                                print('Channel %s at %d has min=max, '
                                      'coherence with this channel will not '
                                      'be calculated'
                                      % (self.timeseries[next_ts].channel.name,
                                         self.timeseries[next_ts].epoch.gps))
                            else:
                                maxfs = max(maxfs,
                                            self.timeseries[next_ts].
                                            sample_rate)
                                # calculate and plot the first pair,
                                # note the first channel is the reference chan
                                snd_ts = self.timeseries[next_ts]
                                coh = self.timeseries[ref_idx].\
                                    coherence(snd_ts, fftlength=fftlen,
                                              overlap=ovlap_frac*fftlen)

                                legend_text = self.timeseries[next_ts].\
                                    channel.name
                                if len(self.start_list) > 1:
                                    legend_text += ', %s' % \
                                                   snd_ts.times[0]
                                coh.name = legend_text

                                # coh2 = 1 / (1-coh) : how to do alt scaler

                                if not cohs:
                                    self.plot = coh.plot()
                                else:
                                    self.plot.add_frequencyseries(coh)

                                cohs.append(coh)

        if not cohs:
            raise ValueError('No coherence was calculated due to data'
                             ' problems (avaiability or constant values)')
        # if the specified frequency limits adjust our ymin and ymax values
        # at this point self.ymin and self.ymax represent the full spectra
        mymin = cohs[0].value.min()
        mymax = cohs[0].value.max()
        myfmin = 1/self.secpfft * cohs[0].frequencies.unit
        myfmax = maxfs/2
        if arg_list.fmin:
            myfmin = float(arg_list.fmin) * cohs[0].frequencies.unit
        if arg_list.fmax:
            myfmax = float(arg_list.fmax) * cohs[0].frequencies.unit

        for idx in range(0, len(cohs)):
            t = numpy.where(cohs[idx].frequencies >= myfmin)
            if t[0].size:
                strt = t[0][0]
                t = numpy.where(cohs[idx].frequencies >= myfmax)
                if t[0].size:
                    stop = t[0][t[0].size-1]
                else:
                    stop = cohs[idx].frequencies.size - 1
                mymin = min(mymin, numpy.min(cohs[idx].value[strt:stop]))
                mymax = max(mymax, numpy.max(cohs[idx].value[strt:stop]))

        self.ymin = mymin
        self.ymax = mymax
        self.fmin = max(myfmin.value, 1/self.secpfft)
        self.fmax = min(myfmax.value, maxfs.value/2)
