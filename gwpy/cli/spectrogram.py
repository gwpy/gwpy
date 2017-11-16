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

""" Spectrogram plots
"""

from .cliproduct import CliProduct

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'


class Spectrogram(CliProduct):
    """Plot the spectrogram of a time series
    """

    def get_action(self):
        """Return the string used as "action" on command line.
        """
        return 'spectrogram'

    def init_cli(self, parser):
        """Set up the argument list for this product
        """
        self.arg_chan1(parser)
        self.arg_freq(parser)
        self.arg_ax_linx(parser)
        self.arg_ax_ylf(parser)
        self.arg_ax_intlog(parser)
        self.arg_imag(parser)
        self.arg_plot(parser)

    def get_ylabel(self, args):
        """Default text for y-axis label
        """
        return 'Frequency (Hz)'

    def get_color_label(self):
        return self.scaleText

    def get_max_datasets(self):
        """Spectrogram only handles 1 at a time
        """
        return 1

    def is_image(self):
        """This plot is image type
        """
        return True

    def freq_is_y(self):
        """This plot puts frequency on the y-axis of the image
        """
        return True

    def get_title(self):
        """Start of default super title, first channel is appended to it
        """
        return 'Spectrogram: '

    def gen_plot(self, args):
        """Generate the plot from time series and arguments
        """
        self.is_freq_plot = True

        from numpy import percentile

        secpfft = 1
        if args.secpfft:
            secpfft = float(args.secpfft)
        ovlp_frac = 0.5
        if args.overlap:
            ovlp_frac = float(args.overlap)
        self.secpfft = secpfft
        self.overlap = ovlp_frac

        ovlp_sec = secpfft*ovlp_frac
        nfft = self.dur/(secpfft - ovlp_sec)
        fft_per_stride = int(nfft/(self.width * 0.8))
        stride_sec = fft_per_stride * (secpfft - ovlp_sec) + secpfft - 1
        stride_sec = max(2*secpfft, stride_sec)
        fs = self.timeseries[0].sample_rate.value

        # based on the number of FFT calculations per pixel
        # in output image choose between
        # high time resolution (spectrogram2) and high SNR (spectrogram)
        if fft_per_stride > 3:
            specgram = self.timeseries[0].spectrogram(stride_sec,
                                                      fftlength=secpfft,
                                                      overlap=ovlp_sec)
            self.log(3, ('Spectrogram calc, stride: %.2fs, fftlength: %.2f, '
                         'overlap: %.2f, #fft: %d' %
                         (stride_sec, secpfft, ovlp_sec, nfft)))
        else:
            specgram = self.timeseries[0].spectrogram2(fftlength=secpfft,
                                                       overlap=ovlp_sec)
            self.log(3, ('HR-Spectrogram calc, stride: %.2fs, fftlength: %.2f,'
                         ' overlap: %.2f, #fft: %d' %
                         (stride_sec, secpfft, ovlp_sec, nfft)))
        specgram = specgram ** (1/2.)   # ASD

        norm = False
        if args.norm:
            specgram = specgram.ratio('median')
            norm = True
        # save if we're interactive
        self.result = specgram

        # set default frequency limits
        self.fmax = fs / 2.
        self.fmin = 1 / secpfft

        # default time axis
        self.xmin = self.timeseries[0].times.value.min()
        self.xmax = self.timeseries[0].times.value.max()

        # set intensity (color) limits
        if args.imin:
            lo = float(args.imin)
        elif args.nopct:
            lo = specgram.min()
            lo = lo.value
        else:
            lo = .01

        if norm or args.nopct:
            imin = lo
        else:
            imin = percentile(specgram, lo)

        if args.imax:
            up = float(args.imax)
        elif args.nopct:
            up = specgram.max()
            up = up.value
        else:
            up = 100
        if args.nopct:
            imax = up
        else:
            imax = percentile(specgram, up)

        self.log(3, 'Intensity (colorbar) limits %.3g - %.3g' %
                 (imin, imax))

        pltargs = dict()
        if args.cmap:
            pltargs['cmap'] = args.cmap

        pltargs['vmin'] = imin
        pltargs['vmax'] = imax

        if norm:
            pltargs['norm'] = 'log'
            self.scaleText = 'Normalized to median'
        elif args.lincolors:
            self.scaleText = r'ASD $\left( \frac{\mathrm{Counts}}' \
                             r'{\sqrt{\mathrm{Hz}}}\right)$'
        else:
            pltargs['norm'] = 'log'
            self.scaleText = r'$ASD \left(\frac{\mathrm{Counts}}' \
                             r'{\sqrt{\mathrm{Hz}}}\right)$'

        self.plot = specgram.plot(**pltargs)
        # pass the image limits back to the annotater
        self.imin = imin
        self.imax = imax
