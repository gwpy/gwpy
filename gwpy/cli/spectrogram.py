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

""" Coherence plots
"""
from cliproduct import CliProduct


class Spectrogram(CliProduct):
    """Derived class to calculate Spectrograms"""

    def get_action(self):
        """Return the string used as "action" on command line."""
        return 'spectrogram'

    def init_cli(self, parser):
        """Set up the argument list for this product"""
        self.arg_chan1(parser)
        self.arg_freq(parser)
        self.arg_ax_linx(parser)
        self.arg_ax_ylf(parser)
        self.arg_ax_intlog(parser)
        self.arg_imag(parser)
        self.arg_plot(parser)
        return

    def get_ylabel(self, args):
        """Default text for y-axis label"""
        return 'Frequency (Hz)'

    def get_color_label(self):
        return self.scaleText

    def get_max_datasets(self):
        """Spectrogram only handles 1 at a time"""
        return 1

    def is_image(self):
        """This plot is image type"""
        return True

    def freq_is_y(self):
        """This plot puts frequency on the y-axis of the image"""
        return True

    def get_title(self):
        """Start of default super title, first channel is appended to it"""
        return 'Spectrogram: '

    def gen_plot(self, arg_list):
        """Generate the plot from time series and arguments"""
        self.is_freq_plot = True

        from numpy import percentile

        secpfft = 1
        if arg_list.secpfft:
            secpfft = float(arg_list.secpfft)
        ovlp_frac = 0.5
        if arg_list.overlap:
            ovlp_frac = float(arg_list.overlap)
        self.secpfft = secpfft
        self.overlap = ovlp_frac

        ovlp_sec = secpfft*ovlp_frac
        nfft = self.dur/(secpfft - ovlp_sec)
        fft_per_stride = int(nfft/(self.width * 0.8))
        stride_sec = fft_per_stride * (secpfft - ovlp_sec) + secpfft - 1
        stride_sec = max(2*secpfft, stride_sec)
        fs = self.timeseries[0].sample_rate.value

        # based on the number of FFT calculation choose between
        # high time resolution and high SNR
        snr_nfft = self.dur / stride_sec
        if snr_nfft > 512:
            specgram = self.timeseries[0].spectrogram(stride_sec,
                                                      fftlength=secpfft,
                                                      overlap=ovlp_sec)
            self.log(3, ('Spectrogram calc, stride: %.2fs, fftlength: %.2f, '
                         'overlap: %.2f, #fft: %d' %
                         (stride_sec, secpfft, ovlp_sec, snr_nfft)))
        else:
            specgram = self.timeseries[0].spectrogram2(fftlength=secpfft,
                                                       overlap=ovlp_sec)
            self.log(3, ('HR-Spectrogram calc, stride: %.2fs, fftlength: %.2f,'
                         ' overlap: %.2f, #fft: %d' %
                         (stride_sec, secpfft, ovlp_sec, snr_nfft)))
        specgram = specgram ** (1/2.)   # ASD

        norm = False
        if arg_list.norm:
            specgram = specgram.ratio('median')
            norm = True

        # set default frequency limits
        self.fmax = fs / 2.
        self.fmin = 1 / secpfft

        # default time axis
        self.xmin = self.timeseries[0].times.value.min()
        self.xmax = self.timeseries[0].times.value.max()

        # set intensity (color) limits
        if arg_list.imin:
            lo = float(arg_list.imin)
        else:
            lo = .01
        if norm or arg_list.nopct:
            imin = lo
        else:
            imin = percentile(specgram, lo*100)

        if arg_list.imax:
            up = float(arg_list.imax)
        else:
            up = 100
        if arg_list.nopct:
            imax = up
        else:
            imax = percentile(specgram, up)

        if norm:
            self.plot = specgram.plot(norm='log', vmin=imin, vmax=imax)
            self.scaleText = 'Normalized to median'
        elif arg_list.lincolors:
            self.plot = specgram.plot(vmin=imin, vmax=imax)
            self.scaleText = r'ASD $\left( \frac{\mathrm{Counts}}' \
                             r'{\sqrt{\mathrm{Hz}}}\right)$'
        else:
            self.plot = specgram.plot(norm='log', vmin=imin, vmax=imax)
            self.scaleText = r'$ASD \left(\frac{\mathrm{Counts}}' \
                             r'{\sqrt{\mathrm{Hz}}}\right)$'
        # pass the image limits back to the annotater
        self.imin = imin
        self.imax = imax
        return
