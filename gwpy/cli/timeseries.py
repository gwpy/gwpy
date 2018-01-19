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

"""Time Series plots
"""

from .cliproduct import CliProduct

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class TimeSeries(CliProduct):
    """
    Plot one or more time series
    """

    def get_action(self):
        """Return the string used as "action" on command line.
        """
        return 'timeseries'

    def init_cli(self, parser):
        """Set up the argument list for this product
        """
        self.arg_chan1(parser)
        self.arg_ax_linx(parser)
        self.arg_ax_liny(parser)
        self.arg_plot(parser)

    def get_ylabel(self, args):
        """Text for y-axis label,  check if channel defines it
        """
        ret = self.units

        return ret

    def get_title(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Time series: '

    def gen_plot(self, args):
        """Generate the plot from time series and arguments
        """
        self.max_size = 16384. * 6400.  # that works on my mac
        self.yscale_factor = 1.0

        from gwpy.plotter.tex import label_to_latex
        from numpy import min as npmin
        from numpy import max as npmax

        if self.timeseries[0].size <= self.max_size:
            self.plot = self.timeseries[0].plot()
        else:
            self.plot = self.timeseries[0].plot(linestyle='None', marker='.')
        self.ymin = self.timeseries[0].min().value
        self.ymax = self.timeseries[0].max().value
        self.xmin = self.timeseries[0].times.value.min()
        self.xmax = self.timeseries[0].times.value.max()

        if len(self.timeseries) > 1:
            for idx in range(1, len(self.timeseries)):
                chname = self.timeseries[idx].channel.name
                lbl = label_to_latex(chname)
                if self.timeseries[idx].size <= self.max_size:
                    self.plot.add_timeseries(self.timeseries[idx], label=lbl)
                else:
                    self.plot.add_timeseries(self.timeseries[idx], label=lbl,
                                             linestyle='None', marker='.')
                self.ymin = min(self.ymin, self.timeseries[idx].min().value)
                self.ymax = max(self.ymax, self.timeseries[idx].max().value)
                self.xmin = min(self.xmin,
                                self.timeseries[idx].times.value.min())
                self.xmax = max(self.xmax,
                                self.timeseries[idx].times.value.max())
        # if they chose to set the range of the x-axis find the range of y
        strt = self.xmin
        stop = self.xmax
        # a bit weird but global ymax will be >= any value in
        # the range same for ymin
        new_ymin = self.ymax
        new_ymax = self.ymin

        if args.xmin:
            strt = float(args.xmin)
        if args.xmax:
            stop = float(args.xmax)
        if strt != self.xmin or stop != self.xmax:
            for idx in range(0, len(self.timeseries)):
                x0 = self.timeseries[idx].x0.value
                dt = self.timeseries[idx].dt.value
                if strt < 1e8:
                    strt += x0
                if stop < 1e8:
                    stop += x0
                b = int(max(0, (strt - x0) / dt))

                e = int(min(self.xmax, (stop - x0) / dt))

                if e >= self.timeseries[idx].size:
                    e = self.timeseries[idx].size - 1
                new_ymin = min(new_ymin,
                               npmin(self.timeseries[idx].value[b:e]))
                new_ymax = max(new_ymax,
                               npmax(self.timeseries[idx].value[b:e]))
            self.ymin = new_ymin
            self.ymax = new_ymax
        if self.yscale_factor > 1:
            self.log(2, ('Scaling y-limits, original: %f, %f)' %
                         (self.ymin, self.ymax)))
            yrange = self.ymax - self.ymin
            mid = (self.ymax + self.ymin) / 2.
            self.ymax = mid + yrange / (2 * self.yscale_factor)
            self.ymin = mid - yrange / (2 * self.yscale_factor)
            self.log(2, ('Scaling y-limits, new: %f, %f)' %
                         (self.ymin, self.ymax)))
