# -*- coding: utf-8 -*-
# Copyright (C) Evan Goetz (2021)
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

"""Transfer function plots
"""

from collections import OrderedDict

from ..plot.bode import BodePlot
from ..plot.tex import label_to_latex
from .spectrum import Spectrum

__author__ = 'Evan Goetz <evan.goetz@ligo.org>'


class TransferFunction(Spectrum):
    """Plot transfer function between a reference time series and one
    or more other time series
    """
    action = 'transferfunction'

    MIN_DATASETS = 2

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type appendages
        if ',' in self.ref_chan:
            self.ref_chan = self.ref_chan.split(',')[0]
        self.plot_dB = self.args.plot_dB

    @classmethod
    def arg_channels(cls, parser):
        group = super().arg_channels(parser)
        group.add_argument('--ref', help='Reference channel against which '
                                         'others will be compared')
        return group

    @classmethod
    def arg_yaxis(cls, parser):
        group = super().arg_yaxis(parser)
        group.add_argument('--plot-dB', action='store_true',
                           help='Plot transfer function in dB')
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
        return 'Transfer function'

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return f"Transfer function: {self.ref_chan}"

    def make_plot(self):
        """Generate the transfer function plot from the time series
        """
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        self.log(2, "Calculating transfer function secpfft: "
                 f"{fftlength}, overlap: {overlap}")
        if overlap is not None:
            overlap *= fftlength

        self.log(3, f"Reference channel: {self.ref_chan}")

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

        plot = BodePlot(figsize=self.figsize, dpi=self.dpi,
                        dB=self.plot_dB)
        ax = plot.gca()
        self.spectra = []

        # calculate transfer function
        for seg in groups:
            refts = groups[seg].pop(self.ref_chan)
            for name in groups[seg]:
                series = groups[seg][name]
                tf = series.transfer_function(refts, fftlength=fftlength,
                                              overlap=overlap,
                                              window=args.window)

                label = name
                if len(self.start_list) > 1:
                    label += f', {series.epoch.gps}'
                if self.usetex:
                    label = label_to_latex(label)

                ax.plot(tf, label=label)
                self.spectra.append(tf)

        if args.xscale == 'log' and not args.xmin:
            args.xmin = 1/fftlength

        return plot

    def set_legend(self):
        """Create a legend for this product
        """
        leg = super().set_legend()
        if leg is not None:
            leg.set_title('Transfer function with:')
        return leg
