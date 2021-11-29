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

"""Coherence plots
"""

from collections import OrderedDict

from ..plot import Plot
from ..plot.tex import label_to_latex
from .spectrum import Spectrum

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Coherence(Spectrum):
    """Plot coherence between a reference time series and one
    or more other time series
    """
    action = 'coherence'

    MIN_DATASETS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type appendages
        if ',' in self.ref_chan:
            self.ref_chan = self.ref_chan.split(',')[0]

    @classmethod
    def arg_channels(cls, parser):
        group = super().arg_channels(parser)
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
        return super()._finalize_arguments(args)

    def get_ylabel(self):
        """Text for y-axis label
        """
        return 'Coherence'

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return f'Coherence: {self.ref_chan}'

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

        self.log(3, 'Reference channel: ' + self.ref_chan)

        # group data by segment
        groups = OrderedDict()
        for series in self.timeseries:
            seg = series.span
            try:
                groups[seg][series.channel.name] = series
            except KeyError:
                groups[seg] = OrderedDict()
                groups[seg][series.channel.name] = series

        # -- plot

        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca()
        self.spectra = []

        # calculate coherence
        for seg in groups:
            refts = groups[seg].pop(self.ref_chan)
            for name in groups[seg]:
                series = groups[seg][name]
                coh = series.coherence(refts, fftlength=fftlength,
                                       overlap=overlap, window=args.window)

                label = name
                if len(self.start_list) > 1:
                    label += f', {series.epoch.gps}'
                if self.usetex:
                    label = label_to_latex(label)

                ax.plot(coh, label=label)
                self.spectra.append(coh)

        if args.xscale == 'log' and not args.xmin:
            args.xmin = 1/fftlength

        return plot

    def set_legend(self):
        """Create a legend for this product
        """
        leg = super().set_legend()
        if leg is not None:
            leg.set_title('Coherence with:')
        return leg
