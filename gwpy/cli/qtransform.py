#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (C) Joseph Areeda (2017)
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

""" Q-transform plots
"""

import re
from .cliproduct import CliProduct
import types
import math
from pprint import pprint


class Qtransform(CliProduct):
    """Derived class to calculate Q-transform"""

    def get_action(self):
        """Return the string used as "action" on command line."""
        return 'qtransform'

    def init_cli(self, parser):
        """Set up the argument list for this product"""

        self.arg_qxform(parser)
        self.arg_ax_linx(parser)
        self.arg_ax_ylf(parser)
        self.arg_ax_intlin(parser)
        self.arg_imag(parser)
        self.arg_plot(parser)
        return

    def post_arg(self, args):
        """Derive standard args from our weird ones
        :type args: Namespace with command line arguments
        """
        event = float(args.gps)
        search = int(args.search)

        start = int(event - search / 2)
        epoch = event
        args.start = [str(start)]
        args.epoch = ('%.3f' % epoch)
        args.duration = search
        args.chan = [[args.chan]]
        args.highpass = None
        args.lowpass = None
        self.verbose = args.verbose

    def get_ylabel(self, args):
        """Default text for y-axis label"""
        return 'Frequency (Hz)'

    def get_color_label(self):
        return 'Normalized energy'

    def get_max_datasets(self):
        """Q-transform only handles 1 at a time"""
        return 1

    def is_image(self):
        """This plot is image type"""
        return True

    def freq_is_y(self):
        """This plot puts frequency on the y-axis of the image"""
        return True

    def get_title(self):
        """Start of default super title, first channel is appended to it"""
        return 'Q-transform: '

    def gen_plot(self, args):
        """Generate the plot from time series and arguments"""
        self.is_freq_plot = False   # not fft based

        my_ts = self.timeseries[0]
        self.title2 = ''
        kwargs = dict()     # optional args passed to qxfrm
        kwargs['search'] = my_ts.dt.value * len(my_ts)
        if args.qrange:
            kwargs['qrange'] = (float(args.qrange[0]), float(args.qrange[1]))
            self.title2 += (' q-range [%.1f, %.1f], ' %
                            (kwargs['qrange'][0], kwargs['qrange'][1]))
        if args.frange:
            kwargs['frange'] = (float(args.frange[0]), float(args.frange[1]))

        if args.nowhiten:
            kwargs['whiten'] = False
            self.title2 += 'not whitened, '
        else:
            self.title2 += 'whitened, '

        kwargs['gps'] = float(args.gps)
        kwargs['fres'] = 0.5
        kwargs['tres'] = 0.002

        new_fs = float(args.sample_freq)
        cur_fs = my_ts.sample_rate.value

        if cur_fs > new_fs:
            my_ts = my_ts.resample(new_fs)
            self.title2 = (' %.0f resampled to %.0f Hz, ' %
                           (cur_fs, new_fs)) + self.title2
            self.log(3, 'Resampled input to %d Hz' % new_fs)
        else:
            new_fs = cur_fs
            self.title2 = (' %.0f Hz, ' % cur_fs) + self.title2

        # if 'frange' in kwargs:
        #     nyq = new_fs / 2
        #     if kwargs['frange'][1] > nyq:
        #         kwargs['frange'] = (kwargs['frange'][0], nyq-24)

        if self.verbose >= 3:
            print ('Q-transform args:')
            pprint(kwargs)

        self.result = my_ts.q_transform(**kwargs)
        self.log(2, 'Result shape: %dx%d' %
                 (self.result.shape[0], self.result.shape[1]))

        self.pltargs = dict()
        if args.cmap:
            self.pltargs['cmap'] = args.cmap
        # weirdness because we allow 2 ways to specify intensity range
        imin = None
        imax = None
        if args.imin:
            imin = float(args.imin)
        if args.imax:
            imax = float(args.imax)
        if args.erange:
            imin = float(args.erange[0])
            imax = float(args.erange[1])

        if imin:
            self.pltargs['vmin'] = imin
        if imax:
            self.pltargs['vmax'] = imax

        if self.verbose >= 3:
            print('Plot args:')
            pprint(self.pltargs)

        self.plot = self.result.plot(**self.pltargs)
        self.scaleText = 'Normalized energy'

        fmax = self.result.frequencies.max().value
        fmin = self.result.frequencies.min().value
        self.log(2, 'Frequency range of result: %.2f - %.2f' %
                 (fmin, fmax))
        self.title2 += (' calc f-range [%.1f, %.1f], ' %
                        (fmin, fmax))
        self.fmin = fmin
        self.fmax = fmax

        emin = self.result.min()
        emax = self.result.max()
        self.title2 += (' calc e-range [%.1f, %.1f] ' %
                        (emin, emax))

        self.plot_num = 0
        self.qx_plot_setup(args)

    def qx_plot_setup(self, args):
        prng = args.plot
        if isinstance(prng, types.ListType):
            prange = float(prng[self.plot_num])
        elif prng:
            prange = float(prng)
        else:
            prange = 0.5
        epoch = float(args.epoch)
        args.xmin = ('%.3f' % (epoch - prange / 2))
        args.xmax = ('%.3f' % (epoch + prange / 2))

        args.out = ('%s/%s-%.3f-%05.2f.png' % (args.outdir,
                    re.sub(':', '-', self.timeseries[0].channel.name),
                    float(args.gps), prange))

    def has_more_plots(self, args):
        """any ranges left to plot?"""
        self.plot_num += 1
        ret = self.plot_num < len(args.plot)
        return ret

    def prep_next_plot(self, args):
        """Override when product needs multiple saves
        """
        self.qx_plot_setup(args)
        self.plot = self.result.plot(**self.pltargs)
