# -*- coding: utf-8 -*-
# Copyright (C) Joseph Areeda (2015-2020)
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

""" Spectrogram plots
"""

from numpy import percentile

from .cliproduct import (FFTMixin, TimeDomainProduct, ImageProduct)
from ..utils import unique

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Spectrogram(FFTMixin, TimeDomainProduct, ImageProduct):
    """Plot the spectrogram of a time series
    """
    action = 'spectrogram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        #: attribute to hold calculated Spectrogram data array
        self.result = None

    @classmethod
    def arg_yaxis(cls, parser):
        return cls._arg_faxis('y', parser)

    def _finalize_arguments(self, args):
        if args.color_scale is None:
            args.color_scale = 'log'
        super()._finalize_arguments(args)

    @property
    def units(self):
        return unique([self.result.unit])

    def get_ylabel(self):
        """Default text for y-axis label
        """
        return 'Frequency (Hz)'

    def get_title(self):
        return f'fftlength={self.args.secpfft}, overlap={self.args.overlap}'

    def get_suptitle(self):
        return f'Spectrogram: {self.chan_list[0]}'

    def get_color_label(self):
        """Text for colorbar label
        """
        if self.args.norm:
            return f'Normalized to {self.args.norm}'
        if len(self.units) == 1 and self.usetex:
            u = self.units[0].to_string('latex').strip('$')
            return fr'ASD $\left({u}\right)$'
        if len(self.units) == 1:
            u = self.units[0].to_string('generic')
            return f'ASD ({u})'
        return super().get_color_label()

    def get_stride(self):
        """Calculate the stride for the spectrogram

        This method returns the stride as a `float`, or `None` to indicate
        selected usage of `TimeSeries.spectrogram2`.
        """
        fftlength = float(self.args.secpfft)
        overlap = fftlength * self.args.overlap
        stride = fftlength - overlap
        nfft = self.duration / stride  # number of FFTs
        ffps = int(nfft / (self.width * 0.8))  # FFTs per second
        if ffps > 3:
            return max(2 * fftlength, ffps * stride + fftlength - 1)
        return None  # do not use strided spectrogram

    def get_spectrogram(self):
        """Calculate the spectrogram to be plotted

        This exists as a separate method to allow subclasses to override
        this and not the entire `get_plot` method, e.g. `Coherencegram`.

        This method should not apply the normalisation from `args.norm`.
        """
        args = self.args

        fftlength = float(args.secpfft)
        overlap = fftlength * args.overlap
        self.log(2, "Calculating spectrogram secpfft: %s, overlap: %s" %
                 (fftlength, overlap))

        stride = self.get_stride()

        if stride:
            specgram = self.timeseries[0].spectrogram(
                stride,
                fftlength=fftlength,
                overlap=overlap,
                method=args.method,
                window=args.window,
            )
            nfft = stride * (stride // (fftlength - overlap))
            self.log(
                3,
                f'Spectrogram calc, stride: {stride}, fftlength: {fftlength}, '
                f'overlap: {overlap}, #fft: {nfft}'
            )
        else:
            specgram = self.timeseries[0].spectrogram2(
                fftlength=fftlength, overlap=overlap, window=args.window)
            nfft = specgram.shape[0]
            self.log(
                3,
                f'HR-Spectrogram calc, fftlength: {fftlength}, '
                f'overlap: {overlap}, #fft: {nfft}'
            )

        return specgram ** (1/2.)   # ASD

    def make_plot(self):
        """Generate the plot from time series and arguments
        """
        args = self.args

        # constant input causes unhelpful (weird) error messages
        # translate them to English
        inmin = self.timeseries[0].min().value
        if inmin == self.timeseries[0].max().value:
            if not self.got_error:
                self.log(0, f'ERROR: Input has constant values [{inmin:g}]. '
                            'Spectrogram-like products cannot process them.')
            self.got_error = True
        else:
            # create 'raw' spectrogram
            specgram = self.get_spectrogram()

            # there may be data that can't be processed
            if specgram is not None:  # <-DMM: why is this here?
                # apply normalisation
                if args.norm:
                    specgram = specgram.ratio(args.norm)

                self.result = specgram

                # -- update plot defaults

                if not args.ymin:
                    args.ymin = 1/args.secpfft if args.yscale == 'log' else 0

                norm = 'log' if args.color_scale == 'log' else None
                # vmin/vmax set in scale_axes_from_data()
                return specgram.plot(figsize=self.figsize, dpi=self.dpi,
                                     norm=norm, cmap=args.cmap)

    def scale_axes_from_data(self):
        args = self.args

        # get tight axes limits from time and frequency Axes
        if args.xmin is None:
            args.xmin = self.result.xspan[0]
        if args.xmax is None:
            args.xmax = self.result.xspan[1]
        if args.ymin is None:
            args.ymin = self.result.yspan[0]
        if args.ymax is None:
            args.ymax = self.result.yspan[1]

        specgram = self.result.crop(
            args.xmin, min(args.xmax, self.result.xspan[1])
        )

        # y axis cannot be cropped if non-linear
        ax = self.plot.gca()
        ax.set_ylim(args.ymin, args.ymax)

        # auto scale colours
        if args.norm:
            imin = specgram.value.min()
            imax = specgram.value.max()
        else:
            imin = percentile(specgram.value, 1)
            imax = percentile(specgram.value, 100)
        imin = args.imin if args.imin is not None else imin
        imax = args.imax if args.imax is not None else imax

        if imin == 0 and args.color_scale == 'log':
            imin = specgram.value[specgram.value > 0].min()

        self.log(3, f'Colorbar limits set to {imin:f} - {imax:f}')

        try:
            image = self.ax.images[0]
        except IndexError:
            image = self.ax.collections[0]
        image.set_clim(imin, imax)
