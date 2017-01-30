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
import sys
import re

from argparse import ArgumentError

from gwpy.timeseries import TimeSeries


class CliProduct(object):
    """Base class for all cli plot products"""

    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.min_timeseries = 1     # datasets needed for this product
        self.xaxis_type = 'uk'      # scaling hints, set by individual actions
        self.xaxis_is_freq = False  # x is going to be frequency or time
        self.yaxis_type = 'uk'      # x and y axis types must be set
        self.iaxis_type = None      # intensity axis (colors) may be missing
        self.chan_list = []         # list of channel names
        self.start_list = []        # list of start times as GPS seconds
        self.dur = 0                # one duration in secs for all start times
        self.timeseries = []        # time series objects after transfer
        self.time_groups = []       # lists of indexes into time series,
        self.minMax = []
        self.verbose = 1
        self.secpfft = 1
        self.overlap = 0.5
        # describe the actual plotted limits
        self.result = 0          # spectrum, or coherence what is plotted
        self.fmin = 0
        self.fmax = 1
        self.ymin = 0
        self.ymax = 0
        self.xmin = 0
        self.xmax = 0
        self.imin = 0
        self.imax = 0
        # image geometry
        self.width = 0
        self.height = 0
        self.xinch = 12
        self.yinch = 7.68
        self.dpi = 100

        self.is_freq_plot = False
        self.n_datasets = 0
        self.filter = ''        # string for annotation if we filtered data
        self.plot = 0           # plot object
        self.ax = 0             # current axis object from plot

# ------Abstract metods------------
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

    @abc.abstractmethod
    def get_title(self):
        """Start of default super title, first channel is appended to it"""
        return

# --------Defaults for overridable methods if product differs from default
    def get_min_datasets(self):
        """Override if plot requires more than 1 dataset.
        eg: coherence requires 2"""
        return 1

    def get_max_datasets(self):
        """Override if plot has a maximum number of datasets.
        eg: spectrogram only handles 1"""
        return 16  # arbitrary max

    def is_image(self):
        """Override if plot is image type, eg: spectrogram"""
        return False

    def freq_is_y(self):
        """Override if frequency is on y-axis like spectrogram"""
        return False

    def get_xlabel(self):
        """Override if you have a better label.
        default is to usw gwpy's default label"""
        return ''

    def get_color_label(self):
        """All products with a color bar should override this"""
        return 'action does not label color bar (it should)'

    def get_sup_title(self):
        """Override if default lacks critical info"""
        return self.get_title() + self.timeseries[0].channel.name

# ------Helper functions
    def log(self, level, msg):
        """print log message if verbosity is set high enough"""
        if self.verbose >= level:
            print msg
        return


# ------Argparse methods

    def arg_chan(self, parser):
        """Allow user to specify list of channel names,
        list of start gps times and single duration"""
        parser.add_argument('--start', nargs='+',
                            help='Starting GPS times(required)')
        parser.add_argument('--duration', default=10,
                            help='Duration (seconds) [10]')
        parser.add_argument('-c', '--framecache',
                            help='use .gwf files in cache not NDS2,' +
                                 ' default use NDS2')
        parser.add_argument('--highpass',
                            help='frequency for high pass filter,' +
                                 ' default no filter')
        parser.add_argument('--lowpass',
                            help='frequency for low pass filter,' +
                                 ' default no filter')
        parser.add_argument('--fshift',
                            help='frequency to shift spectrum,' + 
                                 ' default no shift')
        parser.add_argument('-w','--whiten',
                            help='whiten data using the inverse ASD,' + 
                                 ' default no whitening')
        return

    def arg_chan1(self, parser):
        # list of channel names when only 1 is required
        parser.add_argument('--chan', nargs='+', action='append',
                            required=True, help='One or more channel names.')
        self.arg_chan(parser)
        return

    def arg_chan2(self, parser):
        """list of channel names when at least 2 are required"""
        parser.add_argument('--chan', nargs='+', action='append',
                            required=True,
                            help='Two or more channels or times, first' +
                                 ' one is compared to all the others')
        parser.add_argument('--ref',
                            help='Reference channel against which ' +
                                 'others will be compared')

        self.arg_chan(parser)

        return

    def arg_freq(self, parser):
        """Parameters for FFT based plots, with Spectral defaults"""
        self.is_freq_plot = True
        parser.add_argument('--secpfft', default='1.0',
                            help='length of fft in seconds ' +
                                 'for each calculation, default = 1.0')
        parser.add_argument('--overlap', default='0.5',
                            help='Overlap as fraction [0-1), default=0.5')
        return

    def arg_freq2(self, parser):
        """Parameters for FFT based plots, with Coherencegram defaults"""
        self.is_freq_plot = True
        parser.add_argument('--secpfft', default='0.5',
                            help='length of fft in seconds ' +
                                 'for each calculation, default=0.5')
        parser.add_argument('--overlap', default='0.9',
                            help='Overlap as fraction [0-1), default=0.9')
        return

    def arg_plot(self, parser):
        """Add arguments common to all plots"""
        parser.add_argument('-g', '--geometry', default='1200x600',
                            help='size of resulting image WxH, ' +
                                 'default: %(default)s')
        parser.add_argument('--interactive', action='store_true',
                            help='when running from ipython ' +
                                 'allows experimentation')
        parser.add_argument('--title', action='append',
                            help='One or more title lines')
        parser.add_argument('--suptitle',
                            help='1st title line (larger than the others)')
        parser.add_argument('--xlabel', help='x axis text')
        parser.add_argument('--ylabel', help='y axis text')
        parser.add_argument('--out',
                            help='output filename, type=ext (png, pdf, ' +
                                 'jpg), default=gwpy.png')
        # legends match input files in position are displayed if specified.
        parser.add_argument('--legend', nargs='*', action='append',
                            help='strings to match data files')
        parser.add_argument('--nolegend', action='store_true',
                            help='do not display legend')
        parser.add_argument('--nogrid', action='store_true',
                            help='do not display grid lines')

        return

    def arg_ax_x(self, parser):
        """X-axis is called X. Do not call this
        one call arg_ax_linx or arg_ax_logx"""
        parser.add_argument('--xmin', help='min value for X-axis')
        parser.add_argument('--xmax', help='max value for X-axis')
        return

    def arg_ax_linx(self, parser):
        """X-axis is called X and defaults to linear"""
        self.xaxis_type = 'linx'
        parser.add_argument('--logx', action='store_true',
                            help='make X-axis logarithmic, default=linear')
        parser.add_argument('--epoch',
                            help='center X axis on this GPS time. ' +
                                 'Incompatible with logx')
        self.arg_ax_x(parser)
        return

    def arg_ax_logx(self, parser):
        """X-axis is called X and defaults to logarithmic"""
        self.xaxis_type = 'logx'
        parser.add_argument('--nologx', action='store_true',
                            help='make X-axis linear, default=logarithmic')
        self.arg_ax_x(parser)
        return

    def arg_ax_lf(self, parser):
        """One of this  axis is frequency and logarthmic"""
        parser.add_argument('--nologf', action='store_true',
                            help='make frequency axis linear, ' +
                                 'default=logarithmic')
        parser.add_argument('--fmin', help='min value for frequency axis')
        parser.add_argument('--fmax', help='max value for frequency axis')
        return

    def arg_ax_int(self, parser):
        """Images have an intensity axis"""
        parser.add_argument('--imin',
                            help='min pixel value in resulting image')
        parser.add_argument('--imax',
                            help='max pixek value in resulting image')
        return

    def arg_ax_intlin(self, parser):
        """Intensity (colors) default to linear"""
        self.iaxis = 'lini'
        parser.add_argument('--logcolors', action='store_true',
                            help='set intensity scale of image ' +
                                 'to logarithmic, default=linear')
        self.arg_ax_int(parser)
        return

    def arg_ax_intlog(self, parser):
        """Intensity (colors) default to log"""
        self.iaxis = "logi"
        parser.add_argument('--lincolors', action='store_true',
                            help='set intensity scale of image to linear, '
                                 'default=logarithmic')
        self.arg_ax_int(parser)
        return

    def arg_ax_xlf(self, parser):
        """X-axis is called F and defaults to log"""
        self.xaxis_type = 'logf'
        self.arg_ax_lf(parser)
        return

    def arg_ax_ylf(self, parser):
        """Y-axis is called Frequency and defaults to log"""
        self.yaxis_type = 'logf'
        self.arg_ax_lf(parser)
        return

    def arg_ax_y(self, parser):
        """Y-axis limits.  Do not call this one
        use arg_ax_liny or arg_ax_logy"""
        parser.add_argument('--ymin', help='fix min value for yaxis' +
                                           ' defaults to min of data')
        parser.add_argument('--ymax', help='max value for y-axis ' +
                                           'default to max of data')
        return

    def arg_ax_liny(self, parser):
        """Y-axis is called Y and defaults to linear"""
        self.yaxis_type = 'liny'
        parser.add_argument('--logy', action='store_true',
                            help='make Y-axis logarithmic, default=linear')
        self.arg_ax_y(parser)
        return

    def arg_ax_logy(self, parser):
        """Y-axis is called Y and defaults to log"""
        self.yaxis_type = 'logy'
        parser.add_argument('--nology', action='store_true',
                            help='make Y-axis linear, default=logarthmic')
        self.arg_ax_y(parser)
        return

    def arg_imag(self, parser):
        """Add arguments for image based plots like spectrograms"""
        parser.add_argument('--nopct', action='store_true',
                            help='up and lo are pixel values, ' +
                                 'default=percentile if not normalized')
        parser.add_argument('--nocolorbar', action='store_true',
                            help='hide the color bar')
        parser.add_argument('--norm', action='store_true',
                            help='Display the ratio of each fequency ' +
                                 'bin to the mean of that frequency')
        return

# -------Data transfer methods

    def getTimeSeries(self, arg_list):
        """Verify and interpret arguments to get all
        TimeSeries objects defined"""

        # retrieve channel data from NDS as a TimeSeries
        for chans in arg_list.chan:
            for chan in chans:
                if chan not in self.chan_list:
                    self.chan_list.append(chan)

        if len(self.chan_list) < self.min_timeseries:
            raise ArgumentError(
                'A minimum of %d channels must be specified for this product'
                % self.min_timeseries)

        if len(arg_list.start) > 0:
            self.start_list = list(set(map(int, arg_list.start)))
        else:
            raise ArgumentError('No start times specified')

        # Verify the number of datasets specified is valid for this plot
        self.n_datasets = len(self.chan_list) * len(self.start_list)
        if self.n_datasets < self.get_min_datasets():
            raise ArgumentError(
                '%d datasets are required for this plot but only %d are '
                'supplied' % (self.get_min_datasets(), self.n_datasets))

        if self.n_datasets > self.get_max_datasets():
            raise ArgumentError(
                'A maximum of %d datasets allowed for this plot but %d '
                'specified' % (self.get_max_datasets(), self.n_datasets))

        if arg_list.duration:
            self.dur = int(arg_list.duration)
        else:
            self.dur = 10

        verb = self.verbose > 1

        # determine how we're supposed get our data
        source = 'NDS2'
        frame_cache = False

        if arg_list.framecache:
            source = 'frames'
            frame_cache = arg_list.framecache

        # set up filter parameters for all channels
        highpass = 0
        if arg_list.highpass:
            highpass = float(arg_list.highpass)

        lowpass = 0
        if arg_list.lowpass:
            lowpass = float(arg_list.lowpass)

       whiten = False
       if arg_list.whiten:
            whiten = arg_list.whiten
            self.filter += "whitening "            

       fshift = 0
       if arg_list.fshift:
           fshift = float(arg_list.fshift)
           self.filter += "fshift(%.1f) " % fshift

        # Get the data from NDS or Frames
        # time_groups is a list of timeseries index grouped by
        # start time for coherence like plots
        self.time_groups = []
        for start in self.start_list:
            time_group = []
            for chan in self.chan_list:
                if verb:
                    print 'Fetching %s %d, %d using %s' % \
                          (chan, start, self.dur, source)
                if frame_cache:
                    data = TimeSeries.read(frame_cache, chan, start=start,
                                           end=start+self.dur)
                else:
                    data = TimeSeries.fetch(chan, start, start+self.dur,
                                            verbose=verb)

                if highpass > 0 and lowpass == 0:
                    data = data.highpass(highpass)
                    self.filter += "high pass (%.1f) " % highpass
                elif lowpass > 0 and highpass == 0:
                    data = data.lowpass(lowpass)
                    self.filter += "low pass (%.1f) " % lowpass
                elif lowpass > 0 and highpass > 0:
                    data = data.bandpass(highpass, lowpass)
                    self.filter = "band pass (%.1f-%.1f)" % (highpass, lowpass)
                if whiten:
                    if self.dur < 8:
                        data = data.whiten(self.dur/2.0,self.dur/4.0)
                    else:
                        data = data.whiten(4,2)
                if fshift != 0:
                    data = data.shift(fshift)

                self.timeseries.append(data)
                time_group.append(len(self.timeseries)-1)
            self.time_groups.append(time_group)

        # report what we have if they asked for it
        self.log(3, ('Channels: %s' % self.chan_list))
        self.log(3, ('Start times: %s, duration' % self.start_list, self.dur))
        self.log(3, ('Number of time series: %d' % len(self.timeseries)))

        if len(self.timeseries) != self.n_datasets:
            self.log(0, ('%d datasets requested but only %d transfered' %
                         (self.n_datasets, len(self.timeseries))))
            if len(self.timeseries) > self.get_min_datasets():
                self.log(0, 'Proceeding with the data that was transferred.')
            else:
                self.log(0, 'Not enough data for requested plot.')
                sys.exit(2)
        return

# ---- Plotting methods

    def show_plot_info(self):
        self.log(3, ('X-scale: %s, Y-scale: %s' % (self.plot.get_xscale(),
                                                   self.plot.get_yscale())))
        self.log(3, ('X-limits %s, Y-limits %s' % (self.plot.xlim,
                                                   self.plot.ylim)))

    def config_plot(self, arg_list):
        """Configure global plot parameters"""
        from matplotlib import rcParams

        # set rcParams
        rcParams.update({
            'figure.dpi': 100.,
            'font.family': 'sans-serif',
            'font.size': 16.,
            'font.weight': 'book',
            'legend.loc': 'best',
            'lines.linewidth': 1.5,
            'text.usetex': 'true',
            'agg.path.chunksize': 10000,
        })

        # determine image dimensions (geometry)
        self.width = 1600
        self.height = 900
        if arg_list.geometry:
            try:
                self.width, self.height = map(float,
                                              arg_list.geometry.split('x', 1))
                self.height = max(self.height, 500)
            except (TypeError, ValueError) as e:
                e.args = ('Cannot parse --geometry as WxH, e.g. 1200x600',)
                raise

        self.dpi = rcParams['figure.dpi']
        self.xinch = self.width / self.dpi
        self.yinch = self.height / self.dpi
        rcParams['figure.figsize'] = (self.xinch, self.yinch)
        return

    def setup_xaxis(self, arg_list):
        """Handle scale and limits of X-axis by type"""

        xmin = 0        # these will be set by x min, max or f min, max
        xmax = 1
        scale = 'linear'
        if self.xaxis_type == 'linx' or self.xaxis_type == 'logx':
            # handle time on X-axis
            xmin = self.xmin
            xmax = self.xmax
            if arg_list.xmin:
                al_xmin = float(arg_list.xmin)
                if self.xaxis_is_freq:
                    xmin = al_xmin      # frequency specified
                elif al_xmin <= 1e8:
                    xmin = self.xmin + al_xmin     # time specified as
                    # seconds relative to start GPS
                else:
                    xmin = al_xmin      # time specified as absolute GPS
            if arg_list.xmax:
                al_xmax = float(arg_list.xmax)
                if self.xaxis_is_freq:
                    xmax = al_xmax
                elif al_xmax <= 9e8:
                    xmax = self.xmin + al_xmax
                else:
                    xmax = al_xmax

            if self.xaxis_type == 'linx':
                epoch = None
                if arg_list.epoch:
                    epoch = float(arg_list.epoch)
                    self.log(3, ('Epoch set to %.2f' % epoch))

                scale = False
                self.ax.set_xscale('auto-gps')

                if epoch is not None:
                    # note zero is also false
                    if epoch > 0 and epoch < 1e8:
                        epoch += self.xmin       # specified as seconds
                        self.ax.set_epoch(epoch)
                    elif epoch == 0:
                        self.ax.set_xscale('gps')
                        self.ax.set_epoch(0)
                if arg_list.logx:
                    scale = 'log'
                elif not (self.get_xlabel() or arg_list.xlabel):
                    # duplicate default label except use parens not brackets
                    xscale = self.ax.xaxis._scale
                    if self.ax.get_xscale() == 'auto-gps':
                        epoch = xscale.get_epoch()
                        if epoch is None:
                            arg_list.xlabel = 'GPS Time'
                if self.ax.get_xscale() == 'gps':
                    for l in self.ax.xaxis.get_ticklabels():
                        l.set_rotation(25)
                        l.set_ha('right')
            elif self.xaxis_type == 'logx':
                if arg_list.nologx:
                    scale = 'linear'
                else:
                    scale = 'log'
        elif self.xaxis_type == 'logf':
            # Handle frequency on the X-axis
            xmin = self.fmin
            xmax = self.fmax

            scale = 'log'
            if arg_list.nologf:
                scale = 'linear'
            if arg_list.fmin:
                xmin = float(arg_list.fmin)
            if arg_list.fmax:
                xmax = float(arg_list.fmax)
        else:
            raise AssertionError('X-axis type [%s] is unknown' %
                                 self.xaxis_type)

        if scale:
            self.ax.set_xscale(scale)
        self.ax.set_xlim(xmin, xmax)
        self.log(3, ('X-axis limits are [ %f, %f], scale: %s' %
                     (xmin, xmax, scale)))
        return

    def setup_yaxis(self, arg_list):
        "Set scale and limits of y-axis by type"
        ymin = self.ymin
        ymax = self.ymax
        scale = 'linear'

        if self.yaxis_type == 'logf':
            # Handle frequency on the Y-axis
            ymin = self.fmin
            ymax = self.fmax
            scale = 'log'
            if arg_list.nologf:
                scale = 'linear'
            if arg_list.fmin:
                ymin = float(arg_list.fmin)
            if arg_list.fmax:
                ymax = float(arg_list.fmax)
        elif self.yaxis_type == 'liny' or self.yaxis_type == 'logy':
            # Handle everything but frequency on Y-axis
            if self.yaxis_type == 'liny':
                scale = 'linear'
                if arg_list.logy:
                    scale = 'log'
            elif self.yaxis_type == 'logy':
                scale = 'log'
                if arg_list.nology:
                    scale = 'linear'
            if arg_list.ymin:
                ymin = float(arg_list.ymin)
            if arg_list.ymax:
                ymax = float(arg_list.ymax)
        else:
            raise AssertionError('Y-axis type is unknown')

        # modify axis
        self.ax.set_yscale(scale)
        self.ax.set_ylim(ymin, ymax)
        self.log(3, ('Y-axis limits are [ %f, %f], scale: %s' %
                     (ymin, ymax, scale)))
        return

    def setup_iaxis(self, arg_list):
        """ set the limits and scale of the colorbar (intensity axis)
        :param arg_list: global arguments
        :return: none
        """

        if self.iaxis_type not in [None, 'lin1', 'log1']:
            raise AssertionError('Unknown intensity axis scale')
        return

    def annotate_save_plot(self, arg_list):
        """After the derived class generated a plot
        object finish the process"""
        from astropy.time import Time
        from gwpy.plotter.tex import label_to_latex
        import matplotlib

        self.ax = self.plot.gca()
        # set up axes
        self.setup_xaxis(arg_list)
        self.setup_yaxis(arg_list)
        self.setup_iaxis(arg_list)

        if self.is_image():
            if arg_list.nocolorbar:
                self.plot.add_colorbar(visible=False)
            else:
                self.plot.add_colorbar(label=self.get_color_label())
        else:
            self.plot.add_colorbar(visible=False)

        # image plots don't have legends
        if not self.is_image():
            leg = self.ax.legend(prop={'size': 10})
            # if only one series is plotted hide legend
            if self.n_datasets == 1 and leg:
                try:
                    leg.remove()
                except NotImplementedError:
                    leg.set_visible(False)

        # add titles
        title = ''
        if arg_list.title:
            for t in arg_list.title:
                if len(title) > 0:
                    title += "\n"
                title += t
        # info on the processing
        start = self.start_list[0]
        startGPS = Time(start, format='gps', scale='utc')
        timeStr = "%s - %10d (%ds)" % (startGPS.iso, start, self.dur)

        # list the different sample rates available in all time series
        fs_set = set()

        for idx in range(0, len(self.timeseries)):
            fs = self.timeseries[idx].sample_rate
            fs_set.add(fs)

        fs_str = ''
        for fs in fs_set:
            if len(fs_str) > 0:
                fs_str += ', '
            fs_str += '(%s)' % fs

        if self.is_freq_plot:
            spec = r'%s, Fs=%s, secpfft=%.1f (bw=%.3f), overlap=%.2f' %  \
                    (timeStr, fs_str, self.secpfft, 1/self.secpfft,
                     self.overlap)
        else:
            xdur = self.xmax - self.xmin
            spec = r'Fs=%s, duration: %.1f' % (fs_str, xdur)
        spec += ", " + self.filter
        if len(title) > 0:
            title += "\n"
        title += spec

        title = label_to_latex(title)
        self.plot.set_title(title, fontsize=12)
        self.log(3, ('Title is: %s' % title))

        if arg_list.xlabel:
            xlabel = label_to_latex(arg_list.xlabel)
        else:
            xlabel = self.get_xlabel()
        if xlabel:
            self.plot.set_xlabel(xlabel)
            self.log(3, ('X-axis label is: %s' % xlabel))

        if arg_list.ylabel:
            ylabel = label_to_latex(arg_list.ylabel)
        else:
            ylabel = self.get_ylabel(arg_list)

        if ylabel:
            self.plot.set_ylabel(ylabel)
            self.log(3, ('Y-axis label is: %s' % ylabel))

        if not arg_list.nogrid:
            self.ax.grid(b=True, which='major', color='k', linestyle='solid')
            self.ax.grid(b=True, which='minor', color='0.06',
                         linestyle='dotted')

        # info on the channel
        if arg_list.suptitle:
            sup_title = arg_list.suptitle
        else:
            sup_title = self.get_sup_title()
        sup_title = label_to_latex(sup_title)
        self.plot.suptitle(sup_title, fontsize=18)

        self.log(3, ('Super title is: %s' % sup_title))
        self.show_plot_info()

        # change the label for GPS time so Josh is happy
        if self.ax.get_xscale() == 'auto-gps':
            xscale = self.ax.xaxis._scale
            epoch = xscale.get_epoch()
            unit = xscale.get_unit_name()
            utc = re.sub('\.0+', '',
                         Time(epoch, format='gps', scale='utc').iso)
            self.plot.set_xlabel('Time (%s) from %s (%s)' % (unit, utc, epoch))
            self.ax.xaxis._set_scale(unit, epoch=epoch)

        # if they specified an output file write it
        # save the figure. Note type depends on extension of
        # output filename (png, jpg, pdf)
        if arg_list.out:
            out_file = arg_list.out
        else:
            out_file = "./gwpy.png"

        self.log(3, ('xinch: %.2f, yinch: %.2f, dpi: %d' %
                     (self.xinch, self.yinch, self.dpi)))

        self.fig = matplotlib.pyplot.gcf()
        self.fig.set_size_inches(self.xinch, self.yinch)
        self.plot.savefig(out_file, edgecolor='white',
                          figsize=[self.xinch, self.yinch], dpi=self.dpi, bbox_inches='tight')
        self.log(3, ('wrote %s' % arg_list.out))

        return

# -----The one that does all the work
    def makePlot(self, args):
        """Make the plot, all actions are generally the same at this level"""


        if args.silent:
            self.verbose = 0
        else:
            self.verbose = args.verbose

        self.log(3, ('Verbosity level: %d' % self.verbose))

        if self.verbose > 2:
            print 'Arguments:'
            for key, value in args.__dict__.iteritems():
                print '%s = %s' % (key, value)

        self.getTimeSeries(args)

        self.config_plot(args)

        # this one is in the derived class
        self.gen_plot(args)

        self.annotate_save_plot(args)

        self.is_interactive = False
        if args.interactive:
            self.log(3, 'Interactive manipulation of '
                        'image should be available.')
            self.plot.show()
            self.is_interactive = True
