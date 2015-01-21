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

    def gen_plot(self, args):
        """Generate the plot from time series and arguments"""
        from numpy import min, max

        self.plot = self.timeseries[0].plot()
        self.ymin = self.timeseries[0].min()
        self.ymax = self.timeseries[0].max()
        self.xmin = self.timeseries[0].times.data.min()
        self.xmax = self.timeseries[0].times.data.max()

        if len(self.timeseries) > 10:
            for idx in range(1, len(self.timeseries)):
                chname = self.timeseries[idx].channel.name
                self.plot.add_timeseries(self.timeseries[idx],label=chname)
                self.ymin = min(self.ymin, self.timeseries[idx].min())
                self.ymax = max(self.ymax, self.timeseries[idx].max())
                self.xmin = min(self.xmin, self.timeseries[idx].times.data.min())
                self.xmax = max(self.xmax, self.timeseries[idx].times.data.max())
        return
