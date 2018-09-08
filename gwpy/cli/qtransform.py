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

import os.path
import re
import warnings

from astropy.units import Quantity

from ..segments import Segment
from ..time import to_gps
from .spectrogram import (FFTMixin, Spectrogram)


class Qtransform(Spectrogram):
    """Plot the Q-transform (Omega)"""
    MAX_DATASETS = 1
    action = 'qtransform'

    def __init__(self, *args, **kwargs):
        super(Qtransform, self).__init__(*args, **kwargs)

        args = self.args
        self.qxfrm_args = {
            'gps': float(args.gps),
            'search': args.search / 2.,
            'fres': 0.5,
            'tres': 0.002,
            'whiten': not args.nowhiten,
        }
        if args.qrange is not None:
            self.qxfrm_args['qrange'] = args.qrange
        if args.frange is not None:
            self.qxfrm_args['frange'] = args.frange

    @classmethod
    def init_data_options(cls, parser):
        # call super of FFTMixin to skip setting FFT arguments
        super(FFTMixin, cls).init_data_options(parser)
        cls.arg_qxform(parser)

    @classmethod
    def arg_channels(cls, parser):
        group = parser.add_argument_group('Data options', 'What data to load')
        group.add_argument('--chan', required=True, help='Channel name.')
        group.add_argument('--gps', type=to_gps, required=True,
                           help='Central time of transform')
        group.add_argument('--search', type=float, default=64,
                           help='Time window around GPS to search')
        return group

    @classmethod
    def arg_signal(cls, parser):
        group = super(Qtransform, cls).arg_signal(parser)
        group.add_argument('--sample-freq', type=float, default=2048,
                           help='Downsample freq')

    @classmethod
    def arg_plot(cls, parser):
        group = super(Qtransform, cls).arg_plot(parser)

        # remove --out option
        outopt = [act for act in group._actions if act.dest == 'out'][0]
        group._remove_action(outopt)

        # and replace with --outdir
        group.add_argument('--outdir', default=os.path.curdir, dest='out',
                           type=os.path.abspath,
                           help='Directory for output images')

        return group

    @classmethod
    def arg_qxform(cls, parser):
        """Add an `~argparse.ArgumentGroup` for Q-transform options
        """
        group = parser.add_argument_group('Q-transform options')
        group.add_argument('--plot', nargs='+', type=float, default=[.5],
                           help='One or more times to plot')
        group.add_argument('--frange', nargs=2, type=float,
                           help='Frequency range to plot')
        group.add_argument('--qrange', nargs=2, type=float,
                           help='Search Q range')

        group.add_argument('--nowhiten', action='store_true',
                           help='do not whiten input before transform')

    def _finalize_arguments(self, args):
        """Derive standard args from our weird ones
        :type args: Namespace with command line arguments
        """
        gps = args.gps
        search = args.search

        args.start = [[int(gps - search/2)]]
        if args.epoch is None:
            args.epoch = args.gps
        args.duration = search

        args.chan = [[args.chan]]

        if args.color_scale is None:
            args.color_scale = 'linear'

        args.overlap = 0  # so that FFTMixin._finalize_arguments doesn't fail

        xmin = args.xmin
        xmax = args.xmax

        super(Qtransform, self)._finalize_arguments(args)

        # unset defaults from `TimeDomainProduct`
        args.xmin = xmin
        args.xmax = xmax

    def get_ylabel(self):
        """Default text for y-axis label"""
        return 'Frequency (Hz)'

    def get_color_label(self):
        return 'Normalized energy'

    def get_suptitle(self):
        return 'Q-transform: {0}'.format(self.chan_list[0])

    def get_title(self):
        """Default title for plot
        """
        def fformat(x):  # float format
            if isinstance(x, (list, tuple)):
                return '[{0}]'.format(', '.join(map(fformat, x)))
            if isinstance(x, Quantity):
                x = x.value
            elif isinstance(x, str):
                warnings.warn('WARNING: fformat called with a' +
                            ' string. This has ' +
                            'been depricated and may disappear ' +
                            'in a future release.')
                x = float(x)
            return '{0:.2f}'.format(x)

        bits = [('Q', fformat(self.result.q))]
        if self.qxfrm_args.get('qrange'):
            bits.append(('q-range', fformat(self.qxfrm_args['qrange'])))
        if self.qxfrm_args['whiten']:
            bits.append(('whitened',))
        bits.extend([
            ('calc f-range', fformat(self.result.yspan)),
            ('calc e-range', fformat((self.result.min(), self.result.max()))),
        ])
        return ', '.join(['='.join(bit) for bit in bits])

    def get_spectrogram(self):
        args = self.args

        asd = self.timeseries[0].asd().value
        if (asd.min() == 0):
            self.log(0, 'Input data has a zero in ASD. '
                     'Q-transform not possible.')
            self.got_error = True
            qtrans = None
        else:
            gps = self.qxfrm_args['gps']
            outseg = Segment(gps, gps).protract(args.plot[self.plot_num])
            qtrans = self.timeseries[0].q_transform(
                outseg=outseg,
                **self.qxfrm_args)

            if args.ymin is None:  # set before Spectrogram.make_plot
                args.ymin = qtrans.yspan[0]

        return qtrans

    def scale_axes_from_data(self):
        self.args.xmin, self.args.xmax = self.result.xspan
        return super(Qtransform, self).scale_axes_from_data()

    def has_more_plots(self):
        """any ranges left to plot?
        """
        return self.plot_num < len(self.args.plot)

    def save(self, outdir):  # pylint: disable=arguments-differ
        cname = re.sub('[-_:]', '_', self.timeseries[0].channel.name).replace(
            '_', '-', 1)
        png = '{0}-{1}-{2}.png'.format(cname, float(self.args.gps),
                                       self.args.plot[self.plot_num])
        outfile = os.path.join(outdir, png)
        return super(Qtransform, self).save(outfile)
