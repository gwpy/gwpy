# -*- coding: utf-8 -*-
# Copyright (C) Joseph Areeda (2015-2019)
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

"""Base class for CLI (`gwpy-plot`) products.
"""

from __future__ import print_function
import abc
import os.path
import re
import time
import warnings
import sys
from functools import wraps

from six import add_metaclass

from matplotlib import rcParams
try:
    from matplotlib.cm import viridis as DEFAULT_CMAP
except ImportError:
    from matplotlib.cm import YlOrRd as DEFAULT_CMAP

from astropy.time import Time
from astropy.units import Quantity

from ..utils import unique
from ..signal import filter_design
from ..signal.window import recommended_overlap
from ..time import to_gps
from ..timeseries import TimeSeriesDict
from ..plot.gps import (GPS_SCALES, GPSTransform)
from ..plot.tex import label_to_latex
from ..segments import DataQualityFlag


__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'

BAD_UNITS = {'*', 'Counts.', }


# -- utilities ----------------------------------------------------------------

def timer(func):
    """Time a method and print its duration after return
    """
    name = func.__name__

    @wraps(func)
    def timed_func(self, *args, **kwargs):  # pylint: disable=missing-docstring
        _start = time.time()
        out = func(self, *args, **kwargs)
        self.log(2, '{0} took {1:.1f} sec'.format(name, time.time() - _start))
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
        """Convert the input to a `float` in %s
        """
        return Quantity(x, unit).value

    converter.__doc__ %= str(unit)  # pylint: disable=no-member
    return converter


to_hz = to_float('Hz')  # pylint: disable=invalid-name
to_s = to_float('s')  # pylint: disable=invalid-name


# -- base product class -------------------------------------------------------

@add_metaclass(abc.ABCMeta)
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

    MIN_CHANNELS = 1
    MIN_DATASETS = 1
    MAX_DATASETS = 1e100

    action = None

    def __init__(self, args):

        #: input argument Namespace
        self.args = args

        #: verbosity
        self.verbose = 0 if args.silent else args.verbose

        # NB: finalizing may want to log if we're being verbose
        self._finalize_arguments(args)  # post-process args

        if args.style:  # apply custom styling
            try:
                from matplotlib import style
            except ImportError:
                from matplotlib import __version__ as mpl_version
                warnings.warn('--style can only be used with matplotlib >= '
                              '1.4.0, you have {0}'.format(mpl_version))
            else:
                style.use(args.style)

        #: the current figure object
        self.plot = None

        #: figure number
        self.plot_num = 0

        #: start times for data sets
        self.start_list = unique(
            map(int, (gps for gpsl in args.start for gps in gpsl)))

        #: duration of each time segment
        self.duration = args.duration

        #: channels to load
        self.chan_list = unique(c for clist in args.chan for c in clist)

        # use reduced as an alias for rds in channel name
        for idx in range(len(self.chan_list)):
            self.chan_list[idx] = self.chan_list[idx].replace('reduced', 'rds')

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
        #: Flag for data validation (like all zeroes)
        self.got_error = False

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
    def ax(self):  # pylint: disable=invalid-name
        """The current `~matplotlib.axes.Axes` of this product's plot
        """
        return self.plot.gca()

    @property
    def units(self):
        """The (unique) list of data units for this product
        """
        return unique(ts.unit for ts in self.timeseries)

    @property
    def usetex(self):
        """Switch denoting whether LaTeX will be used or not
        """
        return rcParams['text.usetex']

    # -- utilities ------------------------------

    def log(self, level, msg):
        """print log message if verbosity is set high enough
        :rtype: object
        """
        if self.verbose >= level:
            if level == 0:
                # level zero is important if not fatal error
                print(msg, file=sys.stderr)
            else:
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
        """Add an `~argparse.ArgumentGroup` for channel options
        """
        group = parser.add_argument_group(
            'Data options', 'What data to load')
        group.add_argument('--chan', type=str, nargs='+', action='append',
                           required=True, help='channels to load')
        group.add_argument('--start', type=to_gps, nargs='+', action='append',
                           help='Starting GPS times (required)')
        group.add_argument('--duration', type=to_s, default=10,
                           help='Duration (seconds) [10]')
        return group

    @classmethod
    def arg_data(cls, parser):
        """Add an `~argparse.ArgumentGroup` for data options
        """
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
        """Add an `~argparse.ArgumentGroup` for signal-processing options
        """
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

    # -- plot options

    @classmethod
    def arg_plot(cls, parser):
        """Add an `~argparse.ArgumentGroup` for basic plot options
        """
        group = parser.add_argument_group('Plot options')
        group.add_argument('-g', '--geometry', default='1200x600',
                           metavar='WxH', help='size of resulting image')
        group.add_argument('--dpi', type=int, default=rcParams['figure.dpi'],
                           help='dots-per-inch for figure')
        group.add_argument('--interactive', action='store_true',
                           help='when running from ipython '
                                'allows experimentation')
        group.add_argument('--title', action='store',
                           help='Set title (below suptitle, defaults to '
                                'parameter summary')
        group.add_argument('--suptitle',
                           help='1st title line (larger than the others)')
        group.add_argument('--out', default='gwpy.png',
                           help='output filename')

        # legends match input files in position are displayed if specified.
        group.add_argument('--legend', nargs='+', action='append', default=[],
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
        """Add an `~argparse.ArgumentGroup` for X-axis options.
        """
        return cls._arg_axis('x', parser)

    @classmethod
    def arg_yaxis(cls, parser):
        """Add an `~argparse.ArgumentGroup` for Y-axis options.
        """
        return cls._arg_axis('y', parser)

    @classmethod
    def _arg_axis(cls, axis, parser, **defaults):
        name = '{} axis'.format(axis.title())
        group = parser.add_argument_group('{0} options'.format(name))

        # label
        group.add_argument('--{0}label'.format(axis),
                           default=defaults.get('label'),
                           dest='{0}label'.format(axis),
                           help='{0} label'.format(name))

        # min and max
        for extrema in ('min', 'max'):
            opt = axis + extrema
            group.add_argument('--{0}'.format(opt), type=float,
                               default=defaults.get(extrema), dest=opt,
                               help='{0} value for {1}'.format(extrema, name))

        # scale
        scaleg = group.add_mutually_exclusive_group()
        scaleg.add_argument('--{0}scale'.format(axis), type=str,
                            default=defaults.get('scale'),
                            dest='{0}scale'.format(axis),
                            help='scale for {0}'.format(name))
        if defaults.get('scale') == 'log':
            scaleg.add_argument('--nolog{0}'.format(axis),
                                action='store_const',
                                dest='{0}scale'.format(axis),
                                const=None, default='log',
                                help='use logarithmic {0}'.format(name))
        else:
            scaleg.add_argument('--log{0}'.format(axis), action='store_const',
                                dest='{0}scale'.format(axis),
                                const='log', default=None,
                                help='use logarithmic {0}'.format(name))
        return group

    def _finalize_arguments(self, args):
        # pylint: disable=no-self-use
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
        for freq in notch or []:
            zpks.append(filter_design.notch(freq, data.sample_rate))
        zpk = filter_design.concatenate_zpks(*zpks)

        # apply forward-backward (zero-phase) filter
        return data.filter(*zpk, filtfilt=True)

    # -- plotting -------------------------------

    @staticmethod
    def get_xlabel():
        """Default X-axis label for plot
        """
        return

    @staticmethod
    def get_ylabel():
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
        def _get(param):
            return getattr(self.ax, 'get_{0}{1}'.format(axis, param))()

        def _set(param, *args, **kwargs):
            return getattr(self.ax, 'set_{0}{1}'.format(axis, param))(
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

        # reset scale with epoch if using GPS scale
        if _get('scale') in GPS_SCALES:
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
        """Create a legend for this product (if applicable)
        """
        leg = self.ax.legend(prop={'size': 10})
        if leg and self.n_datasets == 1:
            try:
                leg.remove()
            except NotImplementedError:
                leg.set_visible(False)
        return leg

    def set_title(self, title):
        """Set the title(s) for this plot.

        The `Axes.title` actually serves at the sub-title for the plot,
        typically giving processing parameters and information.
        """

        if title is None:
            title_line = self.get_title().rstrip(', ')
        else:
            title_line = title

        if self.usetex:
            title_line = label_to_latex(title_line)
        if title_line:
            self.ax.set_title(title_line, fontsize=12)
            self.log(3, ('Title is: %s' % title_line))

    def set_suptitle(self, suptitle):
        """Set the super title for this plot.
        """
        if not suptitle:
            suptitle = self.get_suptitle()
        if self.usetex:
            suptitle = label_to_latex(suptitle)
        self.plot.suptitle(suptitle, fontsize=18)
        self.log(3, ('Super title is: %s' % suptitle))

    def set_grid(self, enable):
        """Set the grid parameters for this plot.
        """
        self.ax.grid(b=enable, which='major', color='k', linestyle='solid')
        self.ax.grid(b=enable, which='minor', color='0.06', linestyle='dotted')

    def save(self, outfile):
        """Save this product to the target `outfile`.
        """
        self.plot.savefig(outfile, edgecolor='white', bbox_inches='tight')
        self.log(3, ('wrote %s' % outfile))

    def has_more_plots(self):
        """Determine whether this product has more plots to be created.
        """
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
        """Make the plot.
        """
        self.log(3, ('Verbosity level: %d' % self.verbose))

        self.log(3, 'Arguments:')
        argsd = vars(self.args)
        for key in sorted(argsd):
            self.log(3, '{0:>15s} = {1}'.format(key, argsd[key]))

        # grab the data
        self.get_data()

        # for each plot
        show_error = True       # control ours separate from product's
        while self.has_more_plots():
            self._make_plot()
            if self.plot:
                self.set_plot_properties()
                self.add_segs(self.args)
                if self.args.interactive:
                    self.log(3, 'Interactive manipulation of '
                                'image should be available.')
                    self.plot.show()
                else:
                    self.save(self.args.out)
            elif show_error:
                # Some plots reject inpput data for reasons like all zeroes
                self.log(0, 'No plot produced because of data '
                            'validation error.')
                self.got_error = True
                show_error = False
            self.plot_num += 1

    def add_segs(self, args):
        """ If requested add DQ segments
        """
        std_segments = [
            '{ifo}:DMT-GRD_ISC_LOCK_NOMINAL:1',
            '{ifo}:DMT-DC_READOUT_LOCKED:1',
            '{ifo}:DMT-CALIBRATED:1',
            '{ifo}:DMT-ANALYSIS_READY:1'
        ]
        segments = list()
        if hasattr(args, 'std_seg'):
            if args.std_seg:
                segments = std_segments
            if args.seg:
                for seg in args.seg:
                    # NB: args.seg may be list of lists
                    segments += seg

            chan = args.chan[0][0]
            m = re.match('^([A-Za-z][0-9]):', chan)
            ifo = m.group(1) if m else '?:'

            start = None
            end = 0
            for ts in self.timeseries:
                if start:
                    start = min(ts.t0, start)
                    end = max(ts.t0+ts.duration, end)
                else:
                    start = ts.t0
                    end = start + ts.duration

            for segment in segments:
                seg_name = segment.replace('{ifo}', ifo)
                seg_data = DataQualityFlag.query_dqsegdb(
                        seg_name, start, end)

                self.plot.add_segments_bar(seg_data, label=seg_name)


# -- extensions ---------------------------------------------------------------

@add_metaclass(abc.ABCMeta)
class ImageProduct(CliProduct):
    """Base class for all x/y/color plots
    """
    MAX_DATASETS = 1

    @classmethod
    def init_plot_options(cls, parser):
        super(ImageProduct, cls).init_plot_options(parser)
        cls.arg_color_axis(parser)

    @classmethod
    def arg_color_axis(cls, parser):
        """Add an `~argparse.ArgumentGroup` for colour-axis options.
        """
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
            args.cmap = DEFAULT_CMAP.name
        return super(ImageProduct, self)._finalize_arguments(args)

    @staticmethod
    def get_color_label():
        """Returns the default colorbar label
        """
        return None

    def set_axes_properties(self):
        """Set properties for each axis (scale, limits, label) and create
        a colorbar.
        """
        super(ImageProduct, self).set_axes_properties()
        if not self.args.nocolorbar:
            self.set_colorbar()

    def set_colorbar(self):
        """Create a colorbar for this product
        """
        self.ax.colorbar(label=self.get_color_label())

    def set_legend(self):
        """This method does nothing, since image plots don't have legends
        """
        return  # image plots don't have legends


@add_metaclass(abc.ABCMeta)
class FFTMixin(object):
    """Mixin for `CliProduct` class that will perform FFTs

    This just adds FFT-based command line options
    """
    @classmethod
    def init_data_options(cls, parser):
        """Set up data input and signal processing options including FFTs
        """
        super(FFTMixin, cls).init_data_options(parser)
        cls.arg_fft(parser)

    @classmethod
    def arg_fft(cls, parser):
        """Add an `~argparse.ArgumentGroup` for FFT options
        """
        group = parser.add_argument_group('Fourier transform options')
        group.add_argument('--secpfft', type=float, default=1.,
                           help='length of FFT in seconds')
        group.add_argument('--overlap', type=float,
                           help='overlap as fraction of FFT length [0-1)')
        group.add_argument('--window', type=str, default='hann',
                           help='window function to use when overlapping FFTs')
        return group

    @classmethod
    def _arg_faxis(cls, axis, parser, **defaults):
        defaults.setdefault('scale', 'log')
        axis = axis.lower()
        name = '{0} axis'.format(axis.title())
        group = parser.add_argument_group('{0} options'.format(name))

        # label
        group.add_argument('--{0}label'.format(axis),
                           default=defaults.get('label'),
                           dest='{0}label'.format(axis),
                           help='{0} label'.format(name))

        # min and max
        for extrema in ('min', 'max'):
            meg = group.add_mutually_exclusive_group()
            for ax_ in (axis, 'f'):
                meg.add_argument(
                    '--{0}{1}'.format(ax_, extrema), type=float,
                    default=defaults.get(extrema),
                    dest='{0}{1}'.format(axis, extrema),
                    help='{0} value for {1}'.format(extrema, name))

        # scale
        scaleg = group.add_mutually_exclusive_group()
        scaleg.add_argument('--{0}scale'.format(axis), type=str,
                            default=defaults.get('scale'),
                            dest='{0}scale'.format(axis),
                            help='scale for {0}'.format(name))
        for ax_ in (axis, 'f'):
            if defaults.get('scale') == 'log':
                scaleg.add_argument(
                    '--nolog{0}'.format(ax_), action='store_const',
                    dest='{0}scale'.format(axis), const=None, default='log',
                    help='use linear {0}'.format(name))
            else:
                scaleg.add_argument(
                    '--log{0}'.format(ax_), action='store_const',
                    dest='{0}scale'.format(axis), const='log', default=None,
                    help='use logarithmic {0}'.format(name))

        return group

    def _finalize_arguments(self, args):
        if args.overlap is None:
            try:
                args.overlap = recommended_overlap(args.window)
            except ValueError:
                args.overlap = .5
        return super(FFTMixin, self)._finalize_arguments(args)


@add_metaclass(abc.ABCMeta)
class TimeDomainProduct(CliProduct):
    """`CliProduct` with time on the X-axis
    """
    @classmethod
    def arg_xaxis(cls, parser):
        """Add an `~argparse.ArgumentGroup` for X-axis options.

        This method includes the standard X-axis options, as well as a new
        ``--epoch`` option for the time axis.
        """
        group = super(TimeDomainProduct, cls).arg_xaxis(parser)
        group.add_argument('--epoch', type=to_gps,
                           help='center X axis on this GPS time, may be'
                                'absolute date/time or delta')

        group.add_argument('--std-seg', action='store_true',
                           help='add DQ segment describing IFO state')
        group.add_argument('--seg', type=str, nargs='+', action='append',
                           help='specify one or more DQ segment names')
        return group

    def _finalize_arguments(self, args):
        starts = [float(gps) for gpsl in args.start for gps in gpsl]
        if args.xscale is None:  # set default x-axis scale
            args.xscale = 'auto-gps'
        if args.xmin is None:
            args.xmin = min(starts)
        if args.epoch is None:
            args.epoch = args.xmin
        elif args.epoch < 1e8:
            args.epoch += min(starts)

        if args.xmax is None:
            args.xmax = max(starts) + args.duration
        return super(TimeDomainProduct, self)._finalize_arguments(args)

    def get_xlabel(self):
        """Default X-axis label for plot
        """
        trans = self.ax.xaxis.get_transform()
        if isinstance(trans, GPSTransform):
            epoch = trans.get_epoch()
            unit = trans.get_unit_name()
            utc = re.sub(r'\.0+', '',
                         Time(epoch, format='gps', scale='utc').iso)
            return 'Time ({unit}) from {utc} ({gps})'.format(
                unit=unit, gps=epoch, utc=utc)
        return ''


@add_metaclass(abc.ABCMeta)
class FrequencyDomainProduct(CliProduct):
    """`CliProduct` with frequency on the X-axis
    """
    def get_xlabel(self):
        """Default X-axis label for plot
        """
        return 'Frequency (Hz)'
