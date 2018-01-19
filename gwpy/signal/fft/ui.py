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

from scipy.signal import get_window

from astropy.units import Quantity

from . import utils as fft_utils
from ...utils import mp as mp_utils
from ..window import (canonical_name, recommended_overlap)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def seconds_to_samples(x, rate):
    """Convert a value in seconds to a number of samples for a given rate

    Parameters
    ----------
    x : `float`, `~astropy.units.Quantity`
        number of seconds (or `Quantity` in any time units)

    rate : `float`
        the rate of the relevant data, in Hertz (or `Quantity` in
        any frequency units)

    Returns
    -------
    nsamp : `int`
        the number of samples corresponding to the relevant quantities

    Examples
    --------
    >>> from astropy import units
    >>> from gwpy.signal.fft.ui import seconds_to_samples
    >>> seconds_to_samples(4, 256)
    1024
    >>> seconds_to_samples(1 * units.minute, 16)
    960
    >>> seconds_to_samples(4 * units.second, 16.384 * units.kiloHertz)
    65536
    """
    return int((Quantity(x, 's') * rate).decompose().value)


def normalize_fft_params(series, kwargs=None):
    """Normalize a set of FFT parameters for processing

    This method reads the ``fftlength`` and ``overlap`` keyword arguments
    (presumed to be values in seconds), works out sensible defaults,
    then updates ``kwargs`` in place to include ``nfft`` and ``noverlap``
    as values in sample counts.

    If a ``window`` is given, the ``noverlap`` parameter will be set to the
    recommended overlap for that window type, if ``overlap`` is not given.

    Parameters
    ----------
    series : `gwpy.timeseries.TimeSeries`
        the data that will be processed using an FFT-based method

    kwargs : `dict`
        the dict of keyword arguments passed by the user

    Examples
    --------
    >>> from numpy.random import normal
    >>> from gwpy.timeseries import TimeSeries
    >>> from gwpy.signal.fft.ui import normalize_fft_params
    >>> normalize_fft_params(TimeSeries(normal(size=1024), sample_rate=256))
    {'nfft': 1024, 'noverlap': 0}
    >>> normalize_fft_params(TimeSeries(normal(size=1024), sample_rate=256), {'window': 'hann'})
    {'nfft': 1024, 'noverlap': 0, 'window': 'hann'}
    """
    if kwargs is None:
        kwargs = dict()
    samp = series.sample_rate
    fftlength = kwargs.pop('fftlength', None)
    overlap = kwargs.pop('overlap', None)

    # get canonical window name
    if isinstance(kwargs.get('window', None), str):
        kwargs['window'] = canonical_name(kwargs['window'])

    # fftlength -> nfft
    if fftlength is None:
        fftlength = series.duration
    nfft = seconds_to_samples(fftlength, samp)

    # overlap -> noverlap
    if overlap is None and isinstance(kwargs.get('window', None), str):
        noverlap = recommended_overlap(kwargs['window'], nfft)
    elif overlap is None:
        noverlap = 0
    else:
        noverlap = seconds_to_samples(overlap, samp)

    kwargs.update({
        'nfft': nfft,
        'noverlap': noverlap,
    })
    return kwargs


def set_fft_params(func):
    """Decorate a method to automatically convert quantities to samples
    """
    @wraps(func)
    def wrapped_func(series, method_func, *args, **kwargs):
        """Wrap function to normalize FFT params before execution
        """
        if isinstance(series, tuple):
            data = series[0]
        else:
            data = series

        # extract parameters in seconds, setting recommended default overlap
        normalize_fft_params(data, kwargs)

        return func(series, method_func, *args, **kwargs)

    return wrapped_func


@set_fft_params
def psd(timeseries, method_func, *args, **kwargs):
    """Generate a PSD using a method function

    All arguments are presumed to be given in physical units

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`, `tuple`
        the data to process, or a 2-tuple of series to correlate

    method_func : `callable`
        the function that will be called to perform the signal processing

    *args, **kwargs
        other arguments to pass to ``method_func`` when calling
    """
    # decorator has translated the arguments for us, so just call psdn()
    return psdn(timeseries, method_func, *args, **kwargs)


def psdn(timeseries, method_func, *args, **kwargs):
    """Generate a PSD using a method function with FFT arguments in samples

    All arguments are presumed to be in sample counts, not physical units

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`, `tuple`
        the data to process, or a 2-tuple of series to correlate

    method_func : `callable`
        the function that will be called to perform the signal processing

    *args, **kwargs
        other arguments to pass to ``method_func`` when calling
    """
    # unpack tuple of timeseries for cross spectrum
    try:
        timeseries, other = timeseries
        return method_func(timeseries, other, kwargs.pop('nfft'),
                           *args, **kwargs)
    # or just calculate PSD
    except ValueError:
        return method_func(timeseries, kwargs.pop('nfft'), *args, **kwargs)


def average_spectrogram(timeseries, method_func, stride, *args, **kwargs):
    """Generate an average spectrogram using a method function

    Each time bin of the resulting spectrogram is a PSD generated using
    the method_func
    """
    # unpack CSD TimeSeries pair, or single timeseries
    try:
        timeseries, other = timeseries
    except ValueError:
        timeseries = timeseries
        other = None

    from ...spectrogram import Spectrogram

    nproc = kwargs.pop('nproc', 1)

    # get params
    epoch = timeseries.t0.value
    nstride = seconds_to_samples(stride, timeseries.sample_rate)
    kwargs['fftlength'] = kwargs.pop('fftlength', stride) or stride
    normalize_fft_params(timeseries, kwargs)
    nfft = kwargs['nfft']
    noverlap = kwargs['noverlap']
    window = kwargs.pop('window', None)

    # sanity check parameters
    if nstride > timeseries.size:
        raise ValueError("stride cannot be greater than the duration of "
                         "this TimeSeries")
    if nfft > nstride:
        raise ValueError("fftlength cannot be greater than stride")
    if noverlap >= nfft:
        raise ValueError("overlap must be less than fftlength")

    # generate windows and FFT plans up-front
    if method_func.__module__.endswith('.lal'):
        from .lal import (generate_fft_plan, generate_window)
        if isinstance(window, (str, tuple)):
            window = generate_window(nfft, window=window,
                                     dtype=timeseries.dtype)
        if kwargs.get('plan', None) is None:
            kwargs['plan'] = generate_fft_plan(nfft, dtype=timeseries.dtype)
    else:
        if isinstance(window, (str, tuple)):
            window = get_window(window, nfft)
    # don't operate on None, let the method_func work out its own defaults
    if window is not None:
        kwargs['window'] = window

    # set up single process Spectrogram method
    def _psd(series):
        """Calculate a single PSD for a spectrogram
        """
        try:
            psd_ = psdn(series, method_func, *args, **kwargs)
            del psd_.epoch  # fixes Segmentation fault (no idea why it faults)
            return psd_
        except Exception as exc:  # pylint: disable=broad-except
            if nproc == 1:
                raise
            return exc

    # define chunks
    tschunks = _chunk_timeseries(timeseries, nstride, noverlap)
    if other:
        otherchunks = _chunk_timeseries(other, nstride, noverlap)
        tschunks = zip(tschunks, otherchunks)

    # calculate PSDs
    psds = mp_utils.multiprocess_with_queues(nproc, _psd, tschunks,
                                             raise_exceptions=True)

    # recombobulate PSDs into a spectrogram
    return Spectrogram.from_spectra(*psds, epoch=epoch, dt=stride)


@set_fft_params
def spectrogram(timeseries, method_func, **kwargs):
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
    def _psd(series):
        """Calculate a single PSD for a spectrogram
        """
        try:
            return method_func(series, nfft=nfft, window=window,
                               **kwargs)[1]
        except Exception as exc:  # pylint: disable=broad-except
            if nproc == 1:
                raise
            return exc

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
    numtimes = 1 + int((timeseries.size - nstride) / nstride)
    numfreqs = int(nfft / 2 + 1)
    data = numpy.zeros((numtimes, numfreqs), dtype=timeseries.dtype)
    data[:len(psds)] = psds

    # create output spectrogram
    unit = fft_utils.scale_timeseries_unit(
        timeseries.unit, scaling=kwargs.get('scaling', 'density'))
    out = Spectrogram(numpy.empty((numtimes, numfreqs), dtype=timeseries.dtype),
                      copy=False, dt=nstride * timeseries.dt, t0=timeseries.t0,
                      f0=0, df=sampling/nfft, unit=unit,
                      name=timeseries.name, channel=timeseries.channel)

    # normalize over-dense grid
    density = nfft // nstride
    weights = get_window('triangle', density)
    for i in range(numtimes):
        # get indices of overlapping columns
        x = max(0, i+1-density)
        y = min(i+1, numtimes-density+1)
        if x == 0:
            wgt = weights[-y:]
        elif y == numtimes - density + 1:
            wgt = weights[:y-x]
        else:
            wgt = weights
        # calculate weighted average
        out.value[i, :] = numpy.average(data[x:y], axis=0, weights=wgt)

    return out


def _chunk_timeseries(series, nstride, noverlap):
    # define chunks
    x = y = 0
    step = nstride - int(noverlap // 2.)
    while x + nstride <= series.size:
        y = min(series.size, x + nstride + noverlap)
        yield series[x:y]
        x += step
        step = nstride
