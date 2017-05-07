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
from .cliproduct import CliProduct


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
        self.arg_ax_intlog(parser)
        self.arg_imag(parser)
        self.arg_plot(parser)
        return

    def post_arg(self,args):
        """Derive standard args from our weird ones
        :type args: Namespace with command line arguments
        """
        event = float(args.gps)
        search = int(args.search)
        start = int(event - search / 2)
        epoch = event - start
        args.start = [str(start)]
        args.epoch = str(epoch)
        args.duration = search
        args.chan = [[args.chan]]
        args.highpass = None
        args.lowpass = None

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
        self.is_freq_plot = True

        kwargs = {}     # optional args passed to qxfrm
        kwargs['search'] = self.timeseries[0].dt * len(self.timeseries[0])
        if args.qrange:
            kwargs['qrange'] = args.qrange
        if args.frange:
            kwargs['frange'] = args.frange
        if args.nowhiten:
            kwargs['whiten'] = False
        kwargs['gps'] = float(args.event)

        new_fs = args.sample_freq