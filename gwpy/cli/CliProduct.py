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

    def __init__(self):
        self.min_timeseries = 1      # how many datasets do we need for this product
        self.chan_list = []
        self.start_list = []
        self.timeseries = []
        self.minMax = []
        self.VERBOSE = 1
        self.fmin = 0
        self.fmax = 0
        self.ymin = 0
        self.ymax = 0
        self.xmin =0
        self.xmax = 0
        self.xinch = 12
        self.yinch = 7.68
        self.dpi = 100
        self.is_freq_plot = False

    @abc.abstractmethod
    def get_action(self):
        """Return the string used as "action" on command line."""
        return

    @abc.abstractmethod
    def init_cli(self, parser):
        """Set up the argument list for this product"""
        return

    @abc.abstractmethod
    def gen_plot(self, args):
        """Generate the plot from time series and arguments"""
        return

    @abc.abstractmethod
    def get_ylabel(self, args):
        """Text for y-axis label"""
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
        self.is_freq_plot = True
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
                if type(startArg) is list:
                    self.start_list.append(int(startArg[0]))
                else:
                    self.start_list.append(int(startArg))
        else:
            raise ArgumentError('No start times specified')

        if arg_list.duration:
            self.dur = int(arg_list.duration)
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
                    print 'Fetching %s %d, %d using %s' % (chan, start, self.dur, source)
                if frame_cache:
                    data = TimeSeries.read(frame_cache, chan, start=start, end=start+self.dur)
                else:
                    data = TimeSeries.fetch(chan, start, start+self.dur, verbose=verb)

                if data.min() == data.max():
                    print 'Data from {0:s} has a constant value of {1:f}.',  \
                        'Coherence can not be calculated.' \
                        .format(chan, data.min())
                else:
                    self.timeseries.append(data)
        # report what we have if they asked for it
        if self.VERBOSE > 2:
            print 'Channels: %s' % self.chan_list
            print 'Start times: %s, duration' % self.start_list, self.dur
            print 'Number of time series: %d' % len(self.timeseries)
        return

    def show_plot_info(self):
        print 'X-scale: %s, Y-scale: %s' % (self.plot.get_xscale(), self.plot.get_yscale())
        print 'X-limits %s, Y-limits %s' % (self.plot.xlim, self.plot.ylim)

    def config_plot(self,arg_list):
        """Configure global plot parameters"""
        import matplotlib
        from matplotlib import rcParams

        # set rcParams
        rcParams.update({
            'figure.dpi': 100.,
            'font.family': 'sans-serif',
            'font.size': 16.,
            'font.weight': 'book',
            'legend.loc': 'best',
            'lines.linewidth': 1.5,
        })

        # determine image dimensions (geometry)
        width = 1200
        height = 768
        if arg_list.geometry:
            try:
                width, height = map(float, arg_list.geometry.split('x', 1))
                height = max(height, 500)
            except (TypeError, ValueError) as e:
                e.args = ('Cannot parse --geometry as WxH, e.g. 1200x600',)
                raise

        self.dpi = rcParams['figure.dpi']
        self.xinch = width / self.dpi
        self.yinch = height / self.dpi
        rcParams['figure.figsize'] = (self.xinch, self.yinch)

    def annotate_save_plot(self,arg_list):
        """After the derived class generated a plot object finish the process"""
        from astropy.time import Time
        from gwpy.plotter.tex import label_to_latex

        ax = self.plot.gca()

        if arg_list.logy:
            ax.set_yscale('log')
        else:
            ax.set_yscale('linear')

        if self.is_freq_plot:
            if arg_list.logf:
                ax.set_xscale('log')
            else:
                ax.set_xscale('linear')
        else:
            if arg_list.logx:
                ax.set_xscale('log')
            else:
                ax.set_xscale('linear')

        # scale the axes
        ymin = self.ymin
        ymax = self.ymax

        if arg_list.ymin:
            ymin = arg_list.ymin
        if arg_list.ymax:
            ymax = arg_list.ymax

        if self.VERBOSE > 2:
            print 'Y-axis limits are [ %f, %f]' % (ymin, ymax)

        ax.set_ylim(ymin, ymax)

        if self.is_freq_plot:
            if arg_list.fmin:
                self.fmin = float(arg_list.fmin)
            if arg_list.fmax:
                self.fmax = float(arg_list.fmax)
            if self.VERBOSE > 2:
                print 'Freq-axis limits are [ %f, %f]' % (fmin, fmax)
            ax.set_xlim(self.fmin, self.fmax)

        #todo if time domain set to x limits

        ax.legend(prop={'size':10})

        # add titles
        title = ''
        if arg_list.title:
            for t in arg_list.title:
                if len(title) > 0:
                    title += "\n"
                title += t
        # info on the processing
        fs = self.timeseries[0].sample_rate
        start = self.start_list[0]
        startGPS = Time(start, format='gps')
        timeStr = "%s - %10d (%ds)" % (startGPS.iso, start, self.dur)

        if self.is_freq_plot:
            secpfft = float(arg_list.secpfft)
            ovlap = float(arg_list.overlap)
            spec = r'%s, Fs=%s, secpfft=%.1f, overlap=%.2f' % (timeStr, fs, secpfft, ovlap)

        if len(title) > 0:
            title += "\n"
            title += spec
            title = label_to_latex(title)
            self.plot.set_title(title, fontsize=12)

        if arg_list.xlabel:
            xlabel = label_to_latex(arg_list.xlabel)
        else:
            xlabel = 'Frequency (Hz)'
        self.plot.xlabel = xlabel

        if arg_list.ylabel:
            ylabel = label_to_latex(arg_list.ylabel)
        else:
            ylabel = self.get_ylabel(arg_list)
            if arg_list.logy:
                ylabel += ' log$_{10}$'

        self.plot.ylabel = ylabel

        if not arg_list.nogrid:
            ax.grid(b=True, which='major', color='k', linestyle='solid')
            ax.grid(b=True, which='minor', color='0.06', linestyle='dotted')

        # info on the channel
        if arg_list.suptitle:
            sup_title = arg_list.suptitle
        else:
            sup_title = "Coherence with " + self.timeSeries[0].channel.name
        sup_title = label_to_latex(sup_title)
        self.plot.suptitle(sup_title, fontsize=14)

        if self.VERBOSE > 2:
            self.show_plot_info()

        # if they specified an output file write it
        # save the figure. Note type depends on extension of output filename (png, jpg, pdf)
        if arg_list.out:
            if self.VERBOSE > 2:
                print 'xinch: %.2f, yinch: %.2f, dpi: %d' % (self.xinch, self.yinch, self.dpi)

            self.plot.savefig(arg_list.out, edgecolor='white', figsize=[self.xinch, self.yinch], dpi=self.dpi)
            if self.VERBOSE > 0:
                print 'wrote %s' % arg_list.out
        return

    def makePlot(self, plotObj, args):
        """Make the plot, all actions are generally the same at this level"""
        if args.silent:
            self.VERBOSE = 0
        else:
            self.VERBOSE = args.verbose

        if self.VERBOSE > 1:
            print 'Verbosity level: %d' % self.VERBOSE

        if self.VERBOSE > 2:
            print 'Arguments:'
            for key, value in args.__dict__.iteritems():
                print '%s = %s' % (key, value)
        self.getTimeSeries(args)

        #this one is in the derived class
        self.gen_plot(args)

        self.annotate_save_plot(args)

        if args.interactive:
            if self.VERBOSE > 2:
                print 'Interactive manipulation of image should be available.'
            self.plot.show()

