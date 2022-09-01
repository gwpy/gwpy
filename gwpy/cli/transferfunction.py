# -*- coding: utf-8 -*-
# Copyright (C) Evan Goetz (2021)
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

"""Transfer function plots
"""

from astropy.time import Time
from collections import OrderedDict
import numpy as np

from ..plot.bode import BodePlot
from ..plot.tex import label_to_latex
from .cliproduct import (TransferFunctionProduct, FFTMixin)
from ..plot.gps import GPS_SCALES

__author__ = 'Evan Goetz <evan.goetz@ligo.org>'


class TransferFunction(FFTMixin, TransferFunctionProduct):
    """Plot transfer function between a reference time series and one
    or more other time series
    """
    action = 'transferfunction'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ref_chan = self.args.ref or self.chan_list[0]
        # deal with channel type appendages
        if ',' in self.ref_chan:
            self.ref_chan = self.ref_chan.split(',')[0]
        self.plot_dB = self.args.plot_dB
        self.subplot = None
        self.test_chan = self.chan_list[1]
        self.tfs = []

    @property
    def ax(self):  # pylint: disable=invalid-name
        """The current `~matplotlib.axes.Axes` of this product's plot
        """
        return self.plot.axes[self.subplot]

    @classmethod
    def arg_channels(cls, parser):
        group = super().arg_channels(parser)
        group.add_argument('--ref', help='Reference channel against which '
                                         'others will be compared')
        return group

    @classmethod
    def arg_yaxis(cls, parser):
        group = cls._arg_axis('ymag', parser, scale='log')
        group.add_argument('--plot-dB', action='store_true',
                           help='Plot transfer function in dB')
        group = cls._arg_axis('yphase', parser, scale='linear')

        return group

    def get_title(self):
        gps = self.start_list[0]
        utc = Time(gps, format='gps', scale='utc').iso
        tstr = f'{utc} | {gps} ({self.duration})'

        fftstr = f'fftlength={self.args.secpfft}, overlap={self.args.overlap}'

        return ', '.join([tstr, fftstr])

    def _finalize_arguments(self, args):
        if args.ymagscale is None:
            args.ymagscale = 'log'
        if args.plot_dB:
            args.ymagscale = 'linear'

        return super()._finalize_arguments(args)

    def get_ylabel(self):
        """Text for y-axis label
        """
        ylabelstr = ''
        if self.subplot == 0:
            ylabelstr = 'Magnitude'
            if self.plot_dB:
                ylabelstr += ' [dB]'
        if self.subplot == 1:
            ylabelstr = 'Phase [deg.]'

        return ylabelstr

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return f"Transfer function: {self.test_chan}/{self.ref_chan}"

    def set_axes_properties(self):

        for subplot in [0, 1]:
            self.subplot = subplot
            self.scale_axes_from_data()
            self.set_xaxis_properties()
            self.set_yaxis_properties()

    def _set_axis_properties(self, axis):
        """Generic method to set properties for X/Y axis
        on a specific subplot
        """
        def _get(param):
            ret = getattr(self.plot.axes[self.subplot],
                          f'get_{axis[0].lower()}{param}')()
            return ret

        def _set(param, *args, **kwargs):
            if axis.lower().startswith('y'):
                ret = getattr(self.plot.axes[self.subplot], f'set_y{param}')(
                    *args, **kwargs)
            else:
                ret = getattr(self.plot.axes[self.subplot], f'set_x{param}')(
                    *args, **kwargs)
            return ret

        scale = getattr(self.args, f'{axis}scale')
        label = getattr(self.args, f'{axis}label')
        min_ = getattr(self.args, f'{axis}min')
        max_ = getattr(self.args, f'{axis}max')

        # parse limits
        if (
            scale == 'auto-gps'
            and min_ is not None
            and max_ is not None
            and max_ < 1e8
        ):
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
            if axis.lower().startswith('y'):
                label = getattr(self, 'get_ylabel')()
            else:
                label = getattr(self, 'get_xlabel')()
        if self.subplot == 0 and axis == 'x':
            label = None
        if label:
            if self.usetex:
                label = label_to_latex(label)
            _set('label', label)

        # log
        limits = _get('lim')
        scale = _get('scale')
        label = _get('label')
        self.log(
            2,
            f'{axis.upper()}-axis parameters | '
            f'scale: {scale} | '
            f'limits: {limits[0]!s} - {limits[1]!s}'
        )
        self.log(3, (f'{axis.upper()}-axis label: {label}'))

    def set_yaxis_properties(self):
        """Set properties for Y-axis
        """
        if self.subplot == 0:
            self._set_axis_properties('ymag')
        else:
            self._set_axis_properties('yphase')

    def scale_axes_from_data(self):
        """Restrict data limits for Y-axis based on what you can see
        """
        # get tight limits for X-axis
        if self.args.xmin is None:
            self.args.xmin = min(tf.xspan[0] for tf in self.tfs)
            # this is then typically zero, so if the xscale is log or None
            # we'll need to set to be the next bin higher (one step of df)
            if (self.args.xmin == 0
                    and (self.args.xscale == 'log'
                         or self.args.xscale is None)):
                self.args.xmin = min(tf.df.value for tf in self.tfs)
        if self.args.xmax is None:
            self.args.xmax = max(tf.xspan[1] for tf in self.tfs)

        # autoscale view for Y-axis
        cropped = [tf.crop(self.args.xmin, self.args.xmax) for
                   tf in self.tfs]
        ymin = None
        ymax = None
        for tf in cropped:
            if self.subplot == 0:
                if self.plot_dB:
                    vals = 20 * np.log10(abs(tf.value))
                else:
                    vals = abs(tf.value)
            else:
                vals = np.angle(tf.value, deg=True)
            minval = min(vals)
            maxval = max(vals)
            if ymin is None or minval < ymin:
                ymin = minval
            if ymax is None or maxval > ymax:
                ymax = maxval
        self.ax.yaxis.set_data_interval(ymin, ymax, ignore=True)
        self.ax.autoscale_view(scalex=False)

    def set_plot_properties(self):
        """Finalize figure object and show() or save()
        """
        self.set_axes_properties()
        self.subplot = 0
        self.set_title(self.args.title)
        self.set_suptitle(self.args.suptitle)
        self.set_grid(not self.args.nogrid)
        self.subplot = 1
        self.set_grid(not self.args.nogrid)

    def make_plot(self):
        """Generate the transfer function plot from the time series
        """
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        self.log(2, "Calculating transfer function secpfft: "
                 f"{fftlength}, overlap: {overlap}")
        if overlap is not None:
            overlap *= fftlength

        self.log(3, f"Reference channel: {self.ref_chan}")

        # group data by segment
        groups = OrderedDict()
        for series in self.timeseries:
            seg = series.span
            try:
                groups[seg][series.channel.name] = series
            except KeyError:
                groups[seg] = OrderedDict()
                groups[seg][series.channel.name] = series

        # -- plot

        plot = BodePlot(figsize=self.figsize, dpi=self.dpi,
                        dB=self.plot_dB)
        # ax = plot.gca()
        self.tfs = []

        # calculate transfer function
        for seg in groups:
            refts = groups[seg].pop(self.ref_chan)
            for name in groups[seg]:
                series = groups[seg][name]
                self.test_chan = name
                tf = series.transfer_function(refts, fftlength=fftlength,
                                              overlap=overlap,
                                              window=args.window)

                label = name
                if len(self.start_list) > 1:
                    label += f', {series.epoch.gps}'
                if self.usetex:
                    label = label_to_latex(label)

                plot.add_frequencyseries(tf, dB=self.plot_dB, label=label)
                self.tfs.append(tf)

        if args.xscale == 'log' and not args.xmin:
            args.xmin = 1/fftlength

        return plot
