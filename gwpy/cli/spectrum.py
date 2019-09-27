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

"""Spectrum plots
"""

from astropy.time import Time
import warnings

from .cliproduct import (FrequencyDomainProduct, FFTMixin)
from ..utils import unique
from ..plot import Plot
from ..plot.tex import label_to_latex

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Spectrum(FFTMixin, FrequencyDomainProduct):
    """Plot the ASD spectrum of one or more time series
    """
    action = 'spectrum'

    def __init__(self, *args, **kwargs):
        super(Spectrum, self).__init__(*args, **kwargs)
        self.spectra = []

    @property
    def units(self):
        return unique(fs.unit for fs in self.spectra)

    @classmethod
    def arg_xaxis(cls, parser):
        # use frequency axis on X
        return cls._arg_faxis('x', parser)

    @classmethod
    def arg_yaxis(cls, parser):
        # default log Y-axis
        return cls._arg_axis('y', parser, scale='log')

    def _finalize_arguments(self, args):
        if args.yscale is None:
            args.yscale = 'log'
        super(Spectrum, self)._finalize_arguments(args)

    def get_ylabel(self):
        """Text for y-axis label
        """
        if len(self.units) == 1:
            return r'ASD $\left({0}\right)$'.format(
                self.units[0].to_string('latex').strip('$'))
        return 'ASD'

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Spectrum: {0}'.format(self.chan_list[0])

    def get_title(self):
        gps = self.start_list[0]
        utc = Time(gps, format='gps', scale='utc').iso
        tstr = '{0} | {1} ({2})'.format(utc, gps, self.duration)

        fftstr = 'fftlength={0}, overlap={1}'.format(self.args.secpfft,
                                                     self.args.overlap)

        return ', '.join([tstr, fftstr])

    def make_plot(self):
        """Generate the plot from time series and arguments
        """
        args = self.args

        fftlength = float(args.secpfft)
        overlap = args.overlap
        self.log(2, "Calculating spectrum secpfft: {0}, overlap: {1}".format(
            fftlength, overlap))
        overlap *= fftlength

        # create plot
        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca()

        # handle user specified plot labels
        if self.args.legend:
            nlegargs = len(self.args.legend[0])
        else:
            nlegargs = 0
        if nlegargs > 0 and nlegargs != self.n_datasets:
            warnings.warn('The number of legends specified must match '
                          'the number of time series'
                          ' (channels * start times).  '
                          'There are {:d} series and {:d} legends'.format(
                            len(self.timeseries), len(self.args.legend)))
            nlegargs = 0  # don't use  themm

        for i in range(0, self.n_datasets):
            series = self.timeseries[i]
            if nlegargs:
                label = self.args.legend[0][i]
            else:
                label = series.channel.name
                if len(self.start_list) > 1:
                    label += ', {0}'.format(series.epoch.gps)

            asd = series.asd(fftlength=fftlength, overlap=overlap)
            self.spectra.append(asd)

            if self.usetex:
                label = label_to_latex(label)

            ax.plot(asd, label=label)

        if args.xscale == 'log' and not args.xmin:
            args.xmin = 1/fftlength

        return plot

    def scale_axes_from_data(self):
        """Restrict data limits for Y-axis based on what you can see
        """
        # get tight limits for X-axis
        if self.args.xmin is None:
            self.args.xmin = min(fs.xspan[0] for fs in self.spectra)
        if self.args.xmax is None:
            self.args.xmax = max(fs.xspan[1] for fs in self.spectra)

        # autoscale view for Y-axis
        cropped = [fs.crop(self.args.xmin, self.args.xmax) for
                   fs in self.spectra]
        ymin = min(fs.value.min() for fs in cropped)
        ymax = max(fs.value.max() for fs in cropped)
        self.plot.gca().yaxis.set_data_interval(ymin, ymax, ignore=True)
        self.plot.gca().autoscale_view(scalex=False)
