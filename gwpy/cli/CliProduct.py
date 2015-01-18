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

"""Base class for cli plot products, common code for arguments, data transfer,
plot annotation:

Abstract methods (must be overridden by actual products):
    get_action()     - return the string used as "action" in command line
    init_cli(parser) - set up the argument list for this product

Implemented methods (used by multiple products):
    arg_chan(parser) - add definition of a list TimeSeries objects
    arg_freq(parser) - add parameters for fft based plots
    arg_time(parser) - add parameters for time based plots
    arg_plot(parser) - add plot annotations (labels, legends..)
"""

import abc
from argparse import ArgumentError
from gwpy.timeseries import TimeSeries


class CliProduct(object):
    """Base class for all cli plot products"""

    __metaclass__ = abc.ABCMeta

    min_timeseries = 1      # how many datasets do we need for this product
    chan_list = []
    start_list = []
    timeSeries = []
    minMax = []
    VERBOSE = 1

    @abc.abstractmethod
    def get_action(self):
        """Return the string used as "action" on command line."""
        return

    @abc.abstractmethod
    def init_cli(self, parser):
        """Set up the argument list for this product"""
        return

    def arg_chan(self, parser):
        """Allow user to specify list of channel names, list of start gps times and single duration"""
        parser.add_argument('--start', nargs='+', action='append', required=True,
                    help='Starting GPS times(required)')
        parser.add_argument('--duration', default=10, help='Duration (seconds) [10]')
        parser.add_argument('-c', '--framecache',
                            help='use .gwf files in cache not NDS2, default use NDS2')
        return

    def arg_chan1(self,parser):
        # list of channel names when only 1 is required
        parser.add_argument('--chan', nargs='+', action='append', required=True,
                    help='One or more channel names.')
        self.arg_chan(parser)
        return

    def arg_chan2(self,parser):
        """list of channel names when at least 2 are required"""
        parser.add_argument('--chan', nargs='+', action='append', required=True,
                    help='Two or more channel names, first one is compared to all the others')
        self.arg_chan(parser)
        return

    def arg_freq(self, parser):
        """Parameters for FFT based plots"""
        parser.add_argument('--secpfft', help='length of fft in seconds for coh calculation [duration]')
        parser.add_argument('--overlap', help='Overlap as fraction [0-1)')
        parser.add_argument('--logf', action='store_true',
                            help='make frequency axis logarithmic')
        parser.add_argument('--fmin', help='min value for frequency axis')
        parser.add_argument('--fmax', help='max value for frequency axis')

        return

    def arg_time(self, parser):
        """Add arguments for time based plots"""
        parser.add_argument('--logx', action='store_true')
        parser.add_argument('--xmin', help='min value for X-axis')
        parser.add_argument('--xmax', help='max value for X-axis')

        return

    def arg_plot(self, parser):
        """Add arguments common to all plots"""
        parser.add_argument('-g', '--geometry', default='1200x600',
                        help='size of resulting image WxH, default: %(default)s')
        parser.add_argument('--interactive', action='store_true',
                            help='when running from ipython allows experimentation')
        parser.add_argument('--logy', action='store_true',
                            help='make y-axis logarithmic')
        parser.add_argument('--title', action='append', help='One or more title lines')
        parser.add_argument('--suptitle',
                            help='1st title line (larger than the others)')
        parser.add_argument('--xlabel', help='x axis text')
        parser.add_argument('--ylabel', help='y axis text')
        parser.add_argument('--ymin', help='fix min value for yaxis defaults to min of data')
        parser.add_argument('--ymax', help='max value for y-axis default to max of data')
        parser.add_argument('--out', help='output filename, type=ext (png, pdf, jpg)')
        # legends match input files in position are displayed if any are specified.
        parser.add_argument('--legend', nargs='*', action='append',
                      help='strings to match data files')
        parser.add_argument('--nolegend', action='store_true',
                            help='do not display legend')
        parser.add_argument('--nogrid', action='store_true',
                            help='do not display grid lines')

        return

    def arg_imag(self,parser):
        """Add arguments for image based plots like spectrograms"""
        parser.add_argument('--lo', help='min value in resulting image')
        parser.add_argument('--up', help='max value in resulting image')
        parser.add_argument('--nopct', action='store_true',
                            help='up and lo are pixel values, default=percentile')
        parser.add_argument('--nocolorbar', action='store_true',
                            help='hide the color bar')
        parser.add_argument('--lincolors', action='store_true',
                            help='set intensity scale of image to linear, default=logarithmic')

    def getTimeSeries(self,arg_list):
        """Verify and interpret arguments and get all TimeSeries objects defined"""

        # retrieve channel data from NDS as a TimeSeries
        if len(arg_list.chan) >= self.min_timeseries:
            for chan in arg_list.chan:
                self.chan_list.append(chan[0])
        else:
            raise ArgumentError('A minimum of %d channels must be specified for this product'  % self.min_timeseries)

        if len(arg_list.start) > 0:
            for startArg in arg_list.start:
                while type(startArg) is list:
                    self.start_list.append(int(startArg[0]))
        else:
            raise ArgumentError('No start times specified')

        if arg_list.duration:
            dur = int(arg_list.duration)
        else:
            raise ArgumentError('No duration specified')

        verb = self.VERBOSE > 1

        # determine how we're supposed get our data
        source = 'NDS2'
        frame_cache = False

        if arg_list.framecache:
            source='frames'
            frame_cache = arg_list.framecache

        # Get the data from NDS or Frames
        for chan in self.chan_list:
            for start in self.start_list:
                if verb:
                    print 'Fetching %s %d, %d using %s' % (chan, start, dur, source)
                if frame_cache:
                    data = TimeSeries.read(frame_cache, chan, start=start, end=start+dur)
                else:
                    data = TimeSeries.fetch(chan, start, start+dur, verbose=verb)

                if data.min() == data.max():
                    print 'Data from {0:s} has a constant value of {1:f}.',  \
                        'Coherence can not be calculated.' \
                        .format(chan, data.min())
                else:
                    self.timeSeries.append(data)
        return

    def makePlot(self, plotObj, args):
        """Make the plot, all actions are generally the same at this level"""
        if args.silent:
            self.VERBOSE = 0
        else:
            self.VERBOSE = args.verbose
        self.getTimeSeries(args)

