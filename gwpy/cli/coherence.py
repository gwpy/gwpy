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

from collections import OrderedDict

from ..plotter import FrequencySeriesPlot
from ..plotter.tex import label_to_latex
from .spectrum import Spectrum

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Coherence(Spectrum):
    """Plot coherence between a reference time series and one
    or more other time series
    """
    action = 'coherence'

    @classmethod
    def arg_channels(cls, parser):
        group = super(Coherence, cls).arg_channels(parser)
        group.add_argument('--ref', help='Reference channel against which '
                                         'others will be compared')
        return group

    def _finalize_arguments(self, args):
        if args.yscale is None:
            args.yscale = 'linear'
        if args.yscale == 'linear':
            if not args.ymin:
                args.ymin = 0
            if not args.ymax:
                args.ymax = 1.05
        return super(Coherence, self)._finalize_arguments(args)

    def get_ylabel(self):
        """Text for y-axis label
        """
        return 'Coherence'

    def get_min_datasets(self):
        """Coherence requires 2 datasets for the calculation
        """
        return 2

    def get_sup_title(self):
        """Override if default lacks critical info
        """
        return self.get_title() + self.ref_chan_name

    def make_plot(self):
        """Generate the coherence plot from all time series
        """
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        self.log(2, "Calculating spectrum secpfft: %s, overlap: %s" %
                 (fftlength, overlap))
        if overlap is not None:
            overlap *= fftlength

        ref = args.ref or self.timeseries[0].channel.name
        self.ref_chan_name = ref

        self.log(3, 'Reference channel: ' + ref)

        # group data by segment
        groups = OrderedDict()
        for ts in self.timeseries:
            seg = ts.span
            try:
                groups[seg][ts.channel.name] = ts
            except KeyError:
                groups[seg] = OrderedDict()
                groups[seg][ts.channel.name] = ts

        # -- plot

        plot = FrequencySeriesPlot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca()
        self.spectra = []

        # calculate coherence
        for seg in groups:
            refts = groups[seg].pop(ref)
            for name in groups[seg]:
                ts = groups[seg][name]
                coh = ts.coherence(refts, fftlength=fftlength,
                                   overlap=overlap, window=args.window)

                label = name
                if len(self.start_list) > 1:
                    label += ', {0}'.format(ts.epoch.gps)
                if self.usetex:
                    label = label_to_latex(label)

                ax.plot(coh, label=label)
                self.spectra.append(coh)

        if args.xscale == 'log' and not args.xmin:
            args.xmin = 1/fftlength

        return plot

    def add_legend(self):
        leg = super(Coherence, self).add_legend()
        if leg is not None:
            leg.set_title('Coherence with:')
        return leg
