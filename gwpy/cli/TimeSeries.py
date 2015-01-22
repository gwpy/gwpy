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

""" Time Series plots
"""
from CliProduct import CliProduct

class TimeSeries(CliProduct):

    def get_action(self):
        """Return the string used as "action" on command line."""
        return 'TimeSeries'

    def init_cli(self, parser):
        """Set up the argument list for this product"""
        self.arg_chan1(parser)
        self.arg_time(parser)
        self.arg_plot(parser)
        return

    def get_ylabel(self, args):
        """Text for y-axis label"""
        return 'Counts'

    def get_title(self):
        """Start of default super title, first channel is appended to it"""
        return 'Time series: '

    def resample_if_needed(self, timeseries):
        """
        Matplotlib has a maximum number of points it will plot without throwing a too many blocks
        exception.  This hack is a way to get around that.
        """
        max_size = 16384. * 32.  # that works on my mac
        current_size = timeseries.data.size
        if current_size <= max_size:
            return timeseries
        else:
            fs = timeseries.sample_rate.value
            newfs = max_size/current_size * fs
            new_ts = timeseries.resample(newfs)
        return new_ts

    def gen_plot(self, args):
        """Generate the plot from time series and arguments"""
        from gwpy.plotter.tex import label_to_latex
        from numpy import min as npmin
        from numpy import max as npmax

        plotable_timeseries = self.resample_if_needed(self.timeseries[0])
        self.plot = plotable_timeseries.plot()
        self.ymin = plotable_timeseries.min().value
        self.ymax = plotable_timeseries.max().value
        self.xmin = plotable_timeseries.times.data.min()
        self.xmax = plotable_timeseries.times.data.max()

        if len(self.timeseries) > 1:
            for idx in range(1, len(self.timeseries)):
                plotable_timeseries = self.resample_if_needed(self.timeseries[idx])
                chname = plotable_timeseries.channel.name
                lbl = label_to_latex(chname)
                self.plot.add_timeseries(plotable_timeseries,label=lbl)
                self.ymin = min(self.ymin, plotable_timeseries.min().value)
                self.ymax = max(self.ymax, plotable_timeseries.max().value)
                self.xmin = min(self.xmin, plotable_timeseries.times.data.min())
                self.xmax = max(self.xmax, plotable_timeseries.times.data.max())
        # if they chose to set the tange of the x-axis find the range of y
        strt = self.xmin
        stop = self.xmax
        # a bit weird but global ymax will be >= any value in the range same for ymin
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
                b = max(0, (strt - x0) / dt)

                e = min(self.xmax, (stop - x0) / dt)
                if e >= self.timeseries[idx].size:
                    e = self.timeseries[idx].size - 1
                new_ymin = min(new_ymin, npmin(self.timeseries[idx][b:e]))
                new_ymax = max(new_ymax, npmax(self.timeseries[idx][b:e]))
            self.ymin = new_ymin
            self.ymax = new_ymax
        return
