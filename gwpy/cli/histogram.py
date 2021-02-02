# -*- coding: utf-8 -*-
# vim: nu:ai:ts=4:sw=4

#
#  Copyright (C) 2021 Joseph Areeda <joseph.areeda@ligo.org>
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#


"""The histogram CLI product
"""


__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'
__email__ = 'joseph.areeda@ligo.org'
__version__ = '0.0.1'
__myname__ = 'histogram.py'

import math
import sys
from scipy.stats import kurtosis, norm, skew
from astropy.units import Quantity

from .cliproduct import TimeDomainProduct
from ..plot import Plot
from ..plot.tex import label_to_latex
import warnings


class Histogram(TimeDomainProduct):
    """Plot a histogram of values for one or more time series
    """
    action = 'histogram'

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.mean = Quantity(0, '')    # value needed for pytests
        self.std = Quantity(0, '')
        self.skew = Quantity(0, '')
        self.kurtosis = 0
        self.nbins = 0
        self.ymin = sys.maxsize
        self.ymax = 0

    @classmethod
    def init_cli(self, parser):
        """We add an option to add a fit ot gaussian PDF"""
        super().init_cli(parser)    # add all timeseries options
        group = parser.add_argument_group('Curve fit options')
        group.add_argument('--gauss', action='store_true',
                           help='Add gaussian fit')

    @classmethod
    def get_xlabel(self):
        """Text for x-axis label
        """
        units = self.units
        ret = 'Timeseries amplitude'
        if isinstance(units, list):
            if len(units) == 1 and str(units[0]) == '':  # dimensionless
                pass
            elif len(units) == 1 and self.usetex:
                ret += ' [{}]'.format(units[0].to_string('latex'))
            elif len(units) == 1:
                return ' ({})'.format(units[0].to_string())
            elif len(units) > 1:
                ret += ' (multiple units)'
        return ret

    @classmethod
    def get_ylabel(self):
        """Text for y-axis label
        """
        return 'Count'

    def get_suptitle(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Histogram: {0}'.format(self.chan_list[0])

    def get_title(self):
        suffix = super().get_title()
        # limit significant digits for minute trends
        rates = {ts.sample_rate.round(3) for ts in self.timeseries}
        fss = '({0})'.format('), ('.join(map(str, rates)))
        ret = ', '.join([
            'Fs: {0}'.format(fss),
            'dur: {0}'.format(self.duration),
            suffix,
        ])
        ret = '{} bins: {}, mean: {:.3g}, sd: {:.3g}  ' \
              'skew: {:.3g}, kurtosis: {:.3g}'.\
            format(ret, self.nbins, self.mean.value, self.std.value,
                   self.skew, self.kurtosis)
        return ret

    def make_plot(self):
        """Generate the plot from time series and arguments
        """
        plot = Plot(figsize=self.figsize, dpi=self.dpi)
        ax = plot.gca(xscale='linear')

        # handle user specified plot labels
        nlegargs = len(self.args.legend[0]) if self.args.legend else 0

        if nlegargs > 0 and nlegargs != self.n_datasets:
            warnings.warn('The number of legends specified must match '
                          'the number of time series'
                          ' (channels * start times).  '
                          'There are {:d} series and {:d} legends'.format(
                            len(self.timeseries), len(self.args.legend)))
            nlegargs = 0    # don't use  them

        for i in range(0, self.n_datasets):
            series = self.timeseries[i]
            label = self.args.legend[0][i] if nlegargs else \
                series.channel.name
            if self.usetex:
                label = label_to_latex(label)
            nbins = int(max(10., min(math.sqrt(len(series)), 2000)))
            n, bins, patches = ax.hist(series, bins=nbins, label=label,
                                       density=True, alpha=0.75)
            ymax = max(n)
            self.ymax = max(self.ymax, ymax)
            self.ymin = min(self.ymin, min(n[n > 0]))

            if not self.nbins:
                self.nbins = len(n)
                self.skew = skew(series)
                self.kurtosis = kurtosis(series)
                self.mean = series.mean()
                self.std = series.std()

            if ymax > 0 and self.args.gauss:
                mu, sigma = norm.fit(series)
                fit = norm.pdf(bins, mu, sigma)
                ax.plot(bins, fit)

        if not self.args.ymin:
            self.args.ymin = self.ymin
        if not self.args.ymax:
            self.args.ymax = self.ymax

        if not self.args.xmin:
            self.args.xmin = min(ts.min() for ts in self.timeseries).value
        if not self.args.xmax:
            self.args.xmax = max(ts.max() for ts in self.timeseries).value

        return plot

    def _finalize_arguments(self, args):
        """ We cannot guess at default plot params
        until histogram is calculated"""
        if args.xscale is None:  # set default x-axis scale
            args.xscale = 'linear'

    def scale_axes_from_data(self):
        """Restrict data limits for Y-axis based on what you can see
        """
        # get tight limits for X-axis if not specified by user
        if not self.args.xmin:
            self.args.xmin = min(ts.min().value for ts in self.timeseries)
        if not self.args.xmax:
            self.args.xmax = max(ts.max().value for ts in self.timeseries)

        # autoscale view for Y-axis
        ymin = self.args.ymin if self.args.ymin else self.ymin
        ymax = self.args.ymax if self.args.ymax else self.ymax
        self.plot.gca().yaxis.set_data_interval(ymin, ymax, ignore=True)
        self.plot.gca().autoscale_view(scalex=False)
