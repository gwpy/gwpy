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

"""The timeseries CLI product
"""

from .cliproduct import TimeDomainProduct
from ..plot import Plot
from ..plot.tex import label_to_latex
import re
import warnings

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class TimeSeries(TimeDomainProduct):
    """Plot one or more time series
    """
    action = 'timeseries'

    def get_ylabel(self):
        """Text for y-axis label,  check if channel defines it
        """
        units = self.units
        if len(units) == 1 and str(units[0]) == '':  # dimensionless
            return ''
        if len(units) == 1 and self.usetex:
            return units[0].to_string('latex')
        elif len(units) == 1:
            return units[0].to_string()
        elif len(units) > 1:
            return 'Multiple units'
        return super().get_ylabel()

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Time series: {0}'.format(self.chan_list[0])

    def get_title(self):
        suffix = super().get_title()
        # limit significant digits for minute trends
        rates = {ts.sample_rate.round(3) for ts in self.timeseries}
        fss = '({0})'.format('), ('.join(map(str, rates)))
        return ', '.join([
            'Fs: {0}'.format(fss),
            'duration: {0}'.format(self.duration),
            suffix,
        ])

    def _finalize_arguments(self, args):
        """if we are overlaying different times set x-axis appropriately"""
        super()._finalize_arguments(args)
        if args.overlay_times:
            starts = [float(gps) for gpsl in args.start for gps in gpsl]
            xmax = min(starts) + args.duration
            if args.xmax > xmax:
                args.xmax = xmax

    def make_plot(self):
        """Generate the plot from time series and arguments
        """
        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca(xscale='auto-gps')

        # handle user specified plot labels
        if self.args.legend:
            nlegargs = len(self.args.legend[0])
        elif self.args.overlay_times:
            # add times to legend
            legend_text = list()
            for ts in self.timeseries:
                ltext = '{} - {:.0f}'.format(ts.channel.name, ts.t0.value)
                legend_text.append(ltext)
            self.args.legend = [legend_text]
            nlegargs = len(legend_text)
        else:
            nlegargs = 0
        if nlegargs > 0 and nlegargs != self.n_datasets:
            warnings.warn('The number of legends specified must match '
                          'the number of time series'
                          ' (channels * start times).  '
                          'There are {:d} series and {:d} legends'.format(
                            len(self.timeseries), len(self.args.legend)))
            nlegargs = 0    # don't use  them

        if self.args.overlay_times:
            starts = [ts.t0.value for ts in self.timeseries]
            min_t0 = min(starts)
            for ts in self.timeseries:
                ts.t0 = min_t0

        for i in range(0, self.n_datasets):
            series = self.timeseries[i]
            if nlegargs:
                label = self.args.legend[0][i]
            else:
                label = series.channel.name
            if self.usetex:
                label = label_to_latex(label)
            ax.plot(series, label=label)

        return plot

    def scale_axes_from_data(self):
        """Restrict data limits for Y-axis based on what you can see
        """
        # get tight limits for X-axis
        if self.args.xmin is None:
            self.args.xmin = min(ts.xspan[0] for ts in self.timeseries)
        if self.args.xmax is None:
            self.args.xmax = max(ts.xspan[1] for ts in self.timeseries)

        # autoscale view for Y-axis
        cropped = [ts.crop(self.args.xmin, self.args.xmax) for
                   ts in self.timeseries]
        ymin = min(ts.value.min() for ts in cropped)
        ymax = max(ts.value.max() for ts in cropped)
        self.plot.gca().yaxis.set_data_interval(ymin, ymax, ignore=True)
        self.plot.gca().autoscale_view(scalex=False)

    def set_plot_properties(self):
        """If we're overlaying different times default time axis label can
        be confusing so remove the first start time"""
        super().set_plot_properties()
        if self.args.overlay_times and len(self.args.start[0]) > 1:
            lbl = self.ax.get_xlabel()
            m = re.match('^(Time \\(.+\\) from )', lbl)
            new_lbl = m.group(1) + 'start of each time series'
            self.ax.set_xlabel(new_lbl)

    @classmethod
    def arg_channels(self, parser):
        """Add an option to overlay different times rather than on the
        same time scale"""
        super().arg_channels(parser)
        group = parser.add_argument_group(
                'Data options time series', 'Allow overlay')
        group.add_argument('--overlay-times', action='store_true',
                           help='Display multiple start times overlayed '
                                'rather than sequential on time axis')
