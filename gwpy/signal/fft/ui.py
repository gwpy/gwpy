# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017)
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

"""User-interface to FFT routines for GWpy

This module provides the methods that eventually get called by TimeSeries.xxx,
so isn't really for direct user interaction.
"""

from __future__ import absolute_import

from functools import wraps

import numpy

from scipy.signal import (get_window, periodogram)

from astropy.units import Quantity

from . import utils as fft_utils
from ...utils import mp as mp_utils

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def seconds_to_samples(x, rate):
    return int((Quantity(x, 's') * rate).decompose().value)


def _fft_params_in_samples(f):
    @wraps(f)
    def wrapped_func(series, *args, **kwargs):
        # extract parameters in seconds
        fftlength = kwargs.pop('fftlength', None) or series.duration
        overlap = kwargs.pop('overlap', None) or 0

        # convert to samples
        kwargs['nfft'] = seconds_to_samples(fftlength, series.sample_rate)
        kwargs['noverlap'] = seconds_to_samples(overlap, series.sample_rate)

        return f(series, *args, **kwargs)

    return wrapped_func


@_fft_params_in_samples
def psd(timeseries, method_func, *args, **kwargs):
    """Generate a PSD using a method function
    """
    if len(args) and isinstance(args[0], type(timeseries)):
        other = args[0]
        args = args[1:]
        return method_func(timeseries, other, kwargs.pop('nfft'),
                           *args, **kwargs)
    else:
        return method_func(timeseries, kwargs.pop('nfft'), *args, **kwargs)


def average_spectrogram(timeseries, method_func, stride, *args, **kwargs):
    """Generate an average spectrogram using a method function

    Each time bin of the resulting spectrogram is a PSD generated using
    the method_func
    """
    from ...spectrogram import Spectrogram

    nproc = kwargs.pop('nproc', 1)

    # get params
    epoch = timeseries.t0.value
    fftlength = kwargs['fftlength'] = kwargs.pop('fftlength', None) or stride
    overlap = kwargs['overlap'] = kwargs.pop('overlap', None) or 0

    # sanity check parameters
    if stride > abs(timeseries.span):
        raise ValueError("stride cannot be greater than the duration of "
                         "this TimeSeries")
    if fftlength > stride:
        raise ValueError("fftlength cannot be greater than stride")
    if overlap >= fftlength:
        raise ValueError("overlap must be less than fftlength")

    # get params in samples
    nstride = seconds_to_samples(stride, timeseries.sample_rate)
    nfft = seconds_to_samples(fftlength, timeseries.sample_rate)
    noverlap = seconds_to_samples(overlap, timeseries.sample_rate)
    halfoverlap = int(noverlap // 2.)

    # generate windows and FFT plans up-front
    if method_func.__module__.endswith('.lal'):
        from .lal import (generate_fft_plan, generate_window)
        if kwargs.get('window', None) is None:
            kwargs['window'] = generate_window(nfft, dtype=timeseries.dtype)
        if kwargs.get('plan', None) is None:
            kwargs['plan'] = generate_fft_plan(nfft, dtype=timeseries.dtype)
    else:
        window = kwargs.pop('window', None)
        if isinstance(window, str) or type(window) is tuple:
            window = get_window(window, nfft)
        # don't operate on None, let the method_func work out its own defaults
        if window is not None:
            kwargs['window'] = window

    # set up single process Spectrogram method
    def _psd(ts):
        """Calculate a single PSD for a spectrogram
        """
        try:
            psd_ = psd(ts, method_func, *args, **kwargs)
            del psd_.epoch  # fixes Segmentation fault (no idea why it faults)
            return psd_
        except Exception as e:
            if nproc == 1:
                raise
            return e

    # define chunks
    chunks = []
    x = y = 0
    dx = nstride - halfoverlap
    while x + nstride <= timeseries.size:
        y = min(timeseries.size, x + nstride + noverlap)
        chunks.append((x, y))
        x += dx
        dx = nstride

    tschunks = (timeseries[i:j] for i, j in chunks)

    # calculate PSDs
    psds = mp_utils.multiprocess_with_queues(nproc, _psd, tschunks,
                                             raise_exceptions=True)

    # recombobulate PSDs into a spectrogram
    return Spectrogram.from_spectra(*psds, epoch=epoch, dt=stride,
                                    channel=timeseries.channel)


@_fft_params_in_samples
def spectrogram(timeseries, *args, **kwargs):
    """Generate a spectrogram using a method function

    Each time bin of the resulting spectrogram is a PSD estimate using
    a single FFT
    """
    from ...spectrogram import Spectrogram

    # get params
    sampling = timeseries.sample_rate.to('Hz').value
    nproc = kwargs.pop('nproc', 1)
    nfft = kwargs.pop('nfft')
    noverlap = kwargs.pop('noverlap')
    nstride = nfft - noverlap

    # sanity check parameters
    if noverlap >= nfft:
        raise ValueError("overlap must be less than fftlength")

    # get window once (if given)
    window = kwargs.pop('window', None) or 'hann'
    if isinstance(window, (str, tuple)):
        window = get_window(window, nfft)

    # set up single process Spectrogram method
    def _psd(ts):
        """Calculate a single PSD for a spectrogram
        """
        try:
            return periodogram(ts, fs=sampling, nfft=nfft, window=window,
                               **kwargs)[1]
        except Exception as e:
            if nproc == 1:
                raise
            return e

    # define chunks
    chunks = []
    x = y = 0
    while x + nfft <= timeseries.size:
        y = min(timeseries.size, x + nfft)
        chunks.append((x, y))
        x += nstride

    tschunks = (timeseries.value[i:j] for i, j in chunks)

    # calculate PSDs with multiprocessing
    psds = mp_utils.multiprocess_with_queues(nproc, _psd, tschunks,
                                             raise_exceptions=True)

    # convert PSDs to array with spacing for averages
    nt = 1 + int((timeseries.size - nstride) / nstride)
    nf = int(nfft / 2 + 1)
    data = numpy.zeros((nt, nf), dtype=timeseries.dtype)
    data[:len(psds)] = psds

    # create output spectrogram
    unit = fft_utils.scale_timeseries_unit(
        timeseries.unit, scaling=kwargs.get('scaling', 'density'))
    out = Spectrogram(numpy.empty((nt, nf), dtype=timeseries.dtype),
                      copy=False, dt=nstride * timeseries.dt, t0=timeseries.t0,
                      channel=timeseries.channel, unit=unit, f0=0,
                      df=sampling/nfft)

    # normalize over-dense grid
    density = nfft // nstride
    weights = get_window('triangle', density)
    for i in range(nt):
        # get indices of overlapping columns
        x0 = max(0, i+1-density)
        x1 = min(i+1, nt-density+1)
        if x0 == 0:
            w = weights[-x1:]
        elif x1 == nt - density + 1:
            w = weights[:x1-x0]
        else:
            w = weights
        # calculate weighted average
        out.value[i, :] = numpy.average(data[x0:x1], axis=0, weights=w)

    return out
