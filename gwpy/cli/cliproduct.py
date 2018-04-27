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

"""Base class for CLI (`gwpy-plot`) products.
"""

import abc
import os.path
import re
import time
from collections import OrderedDict
from functools import wraps

from matplotlib import (rcParams, pyplot, style)

from astropy.time import Time
from astropy.units import Quantity

from ..signal import filter_design
from ..signal.window import recommended_overlap
from ..time import to_gps
from ..timeseries import TimeSeriesDict
from ..plotter.gps import GPSTransform
from ..plotter.tex import label_to_latex

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'

try:
    from matplotlib.cm import viridis
except ImportError:  # mpl < 1.5
    DEFAULT_CMAP = 'YlOrRd'
else:
    DEFAULT_CMAP = viridis.name

BAD_UNITS = {'*', }


# -- utilities ----------------------------------------------------------------

def timer(func):
    """Time a method and print its duration after return
    """
    name = func.__name__

    @wraps(func)
    def timed_func(self, *args, **kwargs):
        t = time.time()
        out = func(self, *args, **kwargs)
        e = time.time()
        self.log(2, '{0} took {1:.1f} sec'.format(name, e - t))
        return out

    return timed_func


def to_float(unit):
    """Factory to build a converter from quantity string to float

    Examples
    --------
    >>> conv = to_float('Hz')
    >>> conv('4 mHz')
    >>> 0.004
    """
    def converter(x):
        return Quantity(x, unit).value

    return converter

to_hz = to_float('Hz')
to_s = to_float('s')


def unique(list_):
    """Returns a unique version of the input list preserving order

    Examples
    --------
    >>> unique(['b', 'c', 'a', 'a', 'd', 'e', 'd', 'a'])
    ['b', 'c', 'a', 'd', 'e']
    """
    return list(OrderedDict.fromkeys(list_).keys())


# -- base product class -------------------------------------------------------

class CliProduct(object):
    """Base class for all cli plot products

    Parameters
    ----------
    args : `argparse.Namespace`
        Command-line arguments as parsed using
        :meth:`~argparse.ArgumentParser.parse_args`

    Notes
    -----
    This object has two main entry points,

    - `CliProduct.init_cli` - which adds arguments and argument groups to
       an `~argparse.ArgumentParser`
    - `CliProduct.run` - executes the arguments to product one or more plots

    So, the basic usage is follows::

    >>> import argparse
    >>> parser = argparse.ArgumentParser()
    >>> from gwpy.cli import CliProduct
    >>> CliProduct.init_cli(parser)
    >>> product = CliProduct(parser.parse_args())
    >>> product.run()

    The key methods for subclassing are

    - `CliProduct.action` - property defines 'name' for command-line subcommand
    - `CliProduct._finalize_arguments` - this is called from `__init__`
      to set defaults arguments for products that weren't set from the command
      line
    - `CliProduct.make_plot` - post-processes timeseries data and generates
      one figure
    """
    __metaclass__ = abc.ABCMeta

    MIN_CHANNELS = 1
    MIN_DATASETS = 1
    MAX_DATASETS = 1e100

    action = None

    def __init__(self, args):
        self._finalize_arguments(args)  # post-process args

        #: input argument Namespace
        self.args = args

        #: verbosity
        self.verbose = 0 if args.silent else args.verbose

        if args.style:  # apply custom styling
            style.use(args.style)

        #: the current figure object
        self.plot = None

        #: figure number
        self.plot_num = 0

        #: start times for data sets
        self.start_list = unique(map(int, args.start))

        #: duration of each time segment
        self.duration = args.duration

        #: channels to load
        self.chan_list = unique(c for clist in args.chan for c in clist)

        # total number of datasets that _should_ be acquired
        self.n_datasets = len(self.chan_list) * len(self.start_list)

        #: list of input data series (populated by get_data())
        self.timeseries = []

        #: dots-per-inch for figure
        self.dpi = args.dpi
        #: width and height in pixels
        self.width, self.height = map(float, self.args.geometry.split('x', 1))
        #: figure size in inches
        self.figsize = (self.width / self.dpi, self.height / self.dpi)

        # please leave this last
        self._validate_arguments()

    # -- abstract methods ------------------------

    @abc.abstractmethod
    def make_plot(self):
        """Generate the plot from time series and arguments
        """
        return

    # -- properties -----------------------------

    @property
    def ax(self):
        return self.plot.gca()

    @property
    def units(self):
        return unique(ts.unit for ts in self.timeseries)

    @property
    def usetex(self):
        return rcParams['text.usetex']

    # -- utilities ------------------------------

    def log(self, level, msg):
        """print log message if verbosity is set high enough
        """
        if self.verbose >= level:
            print(msg)
        return

    # -- argument parsing -----------------------

    # each method below is a classmethod so that the command-line
    # for a product can be set up without having to create an instance
    # of the class

    @classmethod
    def init_cli(cls, parser):
        """Set up the argument list for this product
        """
        cls.init_data_options(parser)
        cls.init_plot_options(parser)

    @classmethod
    def init_data_options(cls, parser):
        """Set up data input and signal processing options
        """
        cls.arg_channels(parser)
        cls.arg_data(parser)
        cls.arg_signal(parser)

    @classmethod
    def init_plot_options(cls, parser):
        """Set up plotting options
        """
        cls.arg_plot(parser)
        cls.arg_xaxis(parser)
        cls.arg_yaxis(parser)

    # -- data options

    @classmethod
    def arg_channels(cls, parser):
        group = parser.add_argument_group(
            'Data options', 'What data to load')
        group.add_argument('--chan', type=str, nargs='+', action='append',
                           required=True, help='channels to load')
        group.add_argument('--start', type=to_gps, nargs='+',
                           help='Starting GPS times (required)')
        group.add_argument('--duration', type=to_s, default=10,
                           help='Duration (seconds) [10]')
        return group

    @classmethod
    def arg_data(cls, parser):
        group = parser.add_argument_group(
            'Data source options', 'Where to get the data')
        meg = group.add_mutually_exclusive_group()
        meg.add_argument('-c', '--framecache', type=os.path.abspath,
                         help='read data from cache')
        meg.add_argument('-n', '--nds2-server', metavar='HOSTNAME',
                         help='name of nds2 server to use, default is to '
                         'try all of them')
        meg.add_argument('--frametype', help='GWF frametype to read from')
        return group

    @classmethod
    def arg_signal(cls, parser):
        group = parser.add_argument_group(
            'Signal processing options',
            'What to do with the data before plotting'
        )
        group.add_argument('--highpass', type=to_hz,
                           help='Frequency for highpass filter')
        group.add_argument('--lowpass', type=to_hz,
                           help='Frequency for lowpass filter')
        group.add_argument('--notch', type=to_hz, nargs='*',
                           help='Frequency for notch (can give multiple)')
        return group

    @classmethod
    def arg_fft(cls, parser):
        group = parser.add_argument_group('Fourier transform options')
        group.add_argument('--secpfft', type=float, default=1.,
                           help='length of FFT in seconds')
        group.add_argument('--overlap', type=float,
                           help='overlap as fraction of FFT length [0-1)')
        group.add_argument('--window', type=str, default='hann',
                           help='window function to use when overlapping FFTs')
        return group

    # -- plot options

    @classmethod
    def arg_plot(cls, parser):
        """Add arguments common to all plots
        """
        group = parser.add_argument_group('Plot options')
        group.add_argument('-g', '--geometry', default='1200x600',
                           metavar='WxH', help='size of resulting image')
        group.add_argument('--dpi', type=int, default=rcParams['figure.dpi'],
                           help='dots-per-inch for figure')
        group.add_argument('--interactive', action='store_true',
                           help='when running from ipython '
                                'allows experimentation')
        group.add_argument('--title', action='append',
                           help='One or more title lines')
        group.add_argument('--suptitle',
                           help='1st title line (larger than the others)')
        group.add_argument('--out',
                           help='output filename, type=ext (png, pdf, '
                                'jpg), default=gwpy.png')
        # legends match input files in position are displayed if specified.
        group.add_argument('--legend', nargs='*', action='append',
                           help='strings to match data files')
        group.add_argument('--nolegend', action='store_true',
                           help='do not display legend')
        group.add_argument('--nogrid', action='store_true',
                           help='do not display grid lines')
        # allow custom styling with a style file
        group.add_argument(
            '--style', metavar='FILE',
            help='path to custom matplotlib style sheet, see '
                 'http://matplotlib.org/users/style_sheets.html#style-sheets '
                 'for details of how to write one')
        return group

    @classmethod
    def arg_xaxis(cls, parser):
        """Setup options for X-axis
        """
        return cls._arg_axis('x', parser)

    @classmethod
    def arg_yaxis(cls, parser):
        """Setup options for Y-axis
        """
        return cls._arg_axis('y', parser)

    @classmethod
    def _arg_axis(cls, axis, parser):
        name = '{}-axis'.format(axis.upper())
        group = parser.add_argument_group('{0} options'.format(name))
        group.add_argument('--{0}label'.format(axis),
                           help='{0} label'.format(name))
        group.add_argument('--{0}min'.format(axis), type=float,
                           help='min value for {0}'.format(name))
        group.add_argument('--{0}max'.format(axis), type=float,
                           help='max value for {0}'.format(name))
        group.add_argument('--{0}scale'.format(axis), type=str,
                           help='scale for {0}'.format(name))
        return group

    def _finalize_arguments(self, args):
        """Sanity-check and set defaults for arguments
        """
        # this method is called by __init__ (after command-line arguments
        # have been parsed)

        if args.out is None:
            args.out = "gwpy.png"

    def _validate_arguments(self):
        """Sanity check arguments and raise errors if required
        """
        # validate number of data sets requested
        if len(self.chan_list) < self.MIN_CHANNELS:
            raise ValueError('this product requires at least {0} '
                             'channels'.format(self.MIN_CHANNELS))
        if self.n_datasets < self.MIN_DATASETS:
            raise ValueError(
                '%d datasets are required for this plot but only %d are '
                'supplied' % (self.MIN_DATASETS, self.n_datasets)
            )
        if self.n_datasets > self.MAX_DATASETS:
            raise ValueError(
                'A maximum of %d datasets allowed for this plot but %d '
                'specified' % (self.MAX_DATASETS, self.n_datasets)
            )

    # -- data transfer --------------------------

    @timer
    def get_data(self):
        """Get all the data

        This method populates the `timeseries` list attribute
        """
        self.log(2, '---- Loading data -----')

        verb = self.verbose > 1
        args = self.args

        # determine how we're supposed get our data
        source = 'cache' if args.framecache is not None else 'nds2'

        # Get the data from NDS or Frames
        for start in self.start_list:
            end = start + self.duration
            if source == 'nds2':
                tsd = TimeSeriesDict.get(self.chan_list, start, end,
                                         verbose=verb, host=args.nds2_server,
                                         frametype=args.frametype)
            else:
                tsd = TimeSeriesDict.read(args.framecache, self.chan_list,
                                          start=start, end=end)

            for data in tsd.values():
                if str(data.unit) in BAD_UNITS:
                    data.override_unit('undef')

                data = self._filter_timeseries(
                    data, highpass=args.highpass, lowpass=args.lowpass,
                    notch=args.notch)

                if data.dtype.kind == 'f':  # cast single to double
                    data = data.astype('float64', order='A', copy=False)

                self.timeseries.append(data)

        # report what we have if they asked for it
        self.log(3, ('Channels: %s' % self.chan_list))
        self.log(3, ('Start times: %s, duration %s' % (
            self.start_list, self.duration)))
        self.log(3, ('Number of time series: %d' % len(self.timeseries)))

    @staticmethod
    def _filter_timeseries(data, highpass=None, lowpass=None, notch=None):
        """Apply highpass, lowpass, and notch filters to some data
        """
        # catch nothing to do
        if all(x is None for x in (highpass, lowpass, notch)):
            return data

        # build ZPK
        zpks = []
        if highpass is not None and lowpass is not None:
            zpks.append(filter_design.bandpass(highpass, lowpass,
                                               data.sample_rate))
        elif highpass is not None:
            zpks.append(filter_design.highpass(highpass, data.sample_rate))
        elif lowpass is not None:
            zpks.append(filter_design.lowpass(lowpass, data.sample_rate))
        for f in notch:
            zpks.append(filter_design.notch(f, data.sample_rate))
        zpk = filter_design.concatenate_zpks(*zpks)

        # apply forward-backward (zero-phase) filter
        return data.filter(*zpk, filtfilt=True)

    # -- plotting -------------------------------

    def get_xlabel(self):
        """Default X-axis label for plot
        """
        return

    def get_ylabel(self):
        """Default Y-axis label for plot
        """
        return

    def get_title(self):
        """Default title for plot
        """
        highpass = self.args.highpass
        lowpass = self.args.lowpass
        notch = self.args.notch
        filt = ''
        if highpass and lowpass:
            filt += "band pass (%.1f-%.1f)" % (highpass, lowpass)
        elif highpass:
            filt += "high pass (%.1f) " % highpass
        elif lowpass:
            filt += "low pass (%.1f) " % lowpass
        if notch:
            filt += ', notch ({0})'.format(', '.join(map(str, notch)))
        return filt

    def get_suptitle(self):
        """Default super-title for plot
        """
        return self.timeseries[0].channel.name

    def set_plot_properties(self):
        """Finalize figure object and show() or save()
        """
        self.set_axes_properties()
        self.set_legend()
        self.set_title(self.args.title)
        self.set_suptitle(self.args.suptitle)
        self.set_grid(self.args.nogrid)

    def set_axes_properties(self):
        """Set properties for each axis (scale, limits, label)
        """
        self.scale_axes_from_data()
        self.set_xaxis_properties()
        self.set_yaxis_properties()

    def scale_axes_from_data(self):
        """Auto-scale the view based on visible data
        """
        pass

    def _set_axis_properties(self, axis):
        """Generic method to set properties for X/Y axis
        """
        def _get(p):
            return getattr(self.ax, 'get_{0}{1}'.format(axis, p))()

        def _set(p, *args, **kwargs):
            return getattr(self.ax, 'set_{0}{1}'.format(axis, p))(
                *args, **kwargs)

        scale = getattr(self.args, '{}scale'.format(axis))
        label = getattr(self.args, '{}label'.format(axis))
        min_ = getattr(self.args, '{}min'.format(axis))
        max_ = getattr(self.args, '{}max'.format(axis))

        # parse limits
        if scale == 'auto-gps' and (
                min_ is not None and
                max_ is not None and
                max_ < 1e8):
            limits = (min_, min_ + max_)
        else:
            limits = (min_, max_)

        # set limits
        if limits[0] is not None or limits[1] is not None:
            _set('lim', *limits)

        # set scale
        if scale:
            _set('scale', scale)

        # reset scale with epoch if using GPSTransform
        if isinstance(getattr(self.ax, '{}axis'.format(axis)).get_transform(),
                      GPSTransform):
            _set('scale', scale, epoch=self.args.epoch)

        # set label
        if label is None:
            label = getattr(self, 'get_{}label'.format(axis))()
        if label:
            if self.usetex:
                label = label_to_latex(label)
            _set('label', label)

        # log
        limits = _get('lim')
        scale = _get('scale')
        label = _get('label')
        self.log(2, '{0}-axis parameters | scale: {1} | '
                    'limits: {2[0]!s} - {2[1]!s}'.format(
                        axis.upper(), scale, limits))
        self.log(3, ('{0}-axis label: {1}'.format(axis.upper(), label)))

    def set_xaxis_properties(self):
        """Set properties for X-axis
        """
        self._set_axis_properties('x')

    def set_yaxis_properties(self):
        """Set properties for X-axis
        """
        self._set_axis_properties('y')

    def set_legend(self):
        leg = self.ax.legend(prop={'size': 10})
        if self.n_datasets == 1 and leg:
            try:
                leg.remove()
            except NotImplementedError:
                leg.set_visible(False)
        return leg

    def set_title(self, title):
        if title is None:
            title = self.get_title().rstrip(', ')
        if self.usetex:
            title = label_to_latex(title)
        if title:
            self.ax.set_title(title, fontsize=12)
            self.log(3, ('Title is: %s' % title))

    def set_suptitle(self, suptitle):
        if not suptitle:
            suptitle = self.get_suptitle()
        if self.usetex:
            suptitle = label_to_latex(suptitle)
        self.plot.suptitle(suptitle, fontsize=18)
        self.log(3, ('Super title is: %s' % suptitle))

    def set_grid(self, b):
        self.ax.grid(b=b, which='major', color='k', linestyle='solid')
        self.ax.grid(b=b, which='minor', color='0.06', linestyle='dotted')

    def save(self, outfile):
        self.plot.savefig(outfile, edgecolor='white', bbox_inches='tight')
        self.log(3, ('wrote %s' % outfile))

    def has_more_plots(self):
        """override if product needs multiple annotate and saves"""
        if self.plot_num == 0:
            return True
        return False

    @timer
    def _make_plot(self):
        """Override when product needs multiple saves
        """
        self.plot = self.make_plot()

    # -- the one that does all the work ---------

    def run(self):
        """Make the plot
        """
        self.log(3, ('Verbosity level: %d' % self.verbose))

        self.log(3, 'Arguments:')
        argsd = vars(self.args)
        for key in sorted(argsd):
            self.log(3, '{0:>15s} = {1}'.format(key, argsd[key]))

        # grab the data
        self.get_data()

        # for each plot
        while self.has_more_plots():
            self._make_plot()
            self.set_plot_properties()
            if self.args.interactive:
                self.log(3, 'Interactive manipulation of '
                            'image should be available.')
                pyplot.show(self.plot)
            else:
                self.save(self.args.out)
            self.plot_num += 1


# -- extensions ---------------------------------------------------------------

class ImageProduct(CliProduct):
    MAX_DATASETS = 1

    @classmethod
    def init_plot_options(cls, parser):
        super(ImageProduct, cls).init_plot_options(parser)
        cls.arg_color_axis(parser)

    @classmethod
    def arg_color_axis(self, parser):
        group = parser.add_argument_group('Colour axis options')
        group.add_argument('--imin', type=float,
                           help='minimum value for colorbar')
        group.add_argument('--imax', type=float,
                           help='maximum value for colorbar')
        group.add_argument('--cmap',
                           help='Colormap. See '
                                'https://matplotlib.org/examples/color/'
                                'colormaps_reference.html for options')
        group.add_argument('--color-scale', choices=('log', 'linear'),
                           help='scale for colorbar')
        group.add_argument('--norm', nargs='?', const='median',
                           choices=('median', 'mean'), metavar='NORM',
                           help='normalise each pixel against average in '
                                'that frequency bin')
        group.add_argument('--nocolorbar', action='store_true',
                           help='hide the colour bar')

    def _finalize_arguments(self, args):
        if args.cmap is None:
            args.cmap = DEFAULT_CMAP
        return super(ImageProduct, self)._finalize_arguments(args)

    def get_color_label():
        return None

    def set_axes_properties(self):
        super(ImageProduct, self).set_axes_properties()
        self.set_colorbar()

    def set_colorbar(self):
        args = self.args
        if args.nocolorbar:
            self.plot.add_colorbar(visible=False)

        else:
            self.plot.add_colorbar(label=self.get_color_label())

    def set_legend(self):
        return  # image plots don't have legends


class FFTMixin(object):
    """Mixin for `CliProduct` class that will perform FFTs

    This just adds FFT-based command line options
    """
    @classmethod
    def init_data_options(cls, parser):
        super(FFTMixin, cls).init_data_options(parser)
        cls.arg_fft(parser)

    def _finalize_arguments(self, args):
        if args.overlap is None:
            try:
                args.overlap = recommended_overlap(args.window)
            except ValueError:
                args.overlap = .5
        return super(FFTMixin, self)._finalize_arguments(args)


class TimeDomainProduct(CliProduct):
    """`CliProduct` with time on the X-axis
    """
    @classmethod
    def arg_xaxis(cls, parser):
        group = super(TimeDomainProduct, cls).arg_xaxis(parser)
        group.add_argument('--epoch', type=to_gps,
                           help='center X axis on this GPS time')
        return group

    def _finalize_arguments(self, args):
        if args.xscale is None:  # set default x-axis scale
            args.xscale = 'auto-gps'
        if args.epoch is None and args.xmin is not None:
            args.epoch = args.xmin
        elif args.epoch is None:
            args.epoch = args.start[0]
        if args.xmin is None:
            args.xmin = min(args.start)
        if args.xmax is None:
            args.xmax = max(args.start) + args.duration
        return super(TimeDomainProduct, self)._finalize_arguments(args)

    def get_xlabel(self):
        trans = self.ax.xaxis.get_transform()
        if isinstance(trans, GPSTransform):
            epoch = trans.get_epoch()
            unit = trans.get_unit_name()
            utc = re.sub(r'\.0+', '',
                         Time(epoch, format='gps', scale='utc').iso)
            return 'Time ({unit}) from {utc} ({gps})'.format(
                unit=unit, gps=epoch, utc=utc)


class FrequencyDomainProduct(CliProduct):
    """`CliProduct` with frequency on the X-axis
    """
    def _finalize_arguments(self, args):
        if args.xscale is None:  # default frequency scale
            args.xscale = 'log'
        super(FrequencyDomainProduct, self)._finalize_arguments(args)

    def get_xlabel(self):
        return 'Frequency (Hz)'
