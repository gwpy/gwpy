# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2017-2020)
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

from functools import wraps

import numpy

from scipy.signal import (
    get_window,
    periodogram as scipy_periodogram,
)

from astropy.units import Quantity

from . import _utils as fft_utils
from ...utils import mp as mp_utils
from ..window import (canonical_name, recommended_overlap)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


# -- handle physical quantities and set defaults ------------------------------

def seconds_to_samples(x, rate):
    """Convert a value in seconds to a number of samples for a given rate

    Parameters
    ----------
    x : `float`, `~astropy.units.Quantity`
        number of seconds (or `Quantity` in any time units)

    rate : `float`, `~astropy.units.Quantity`
        the rate of the relevant data, in Hertz (or `Quantity` in
        any frequency units)

    Returns
    -------
    nsamp : `int`
        the number of samples corresponding to the relevant quantities

    Examples
    --------
    >>> from astropy import units
    >>> seconds_to_samples(4, 256)
    1024
    >>> seconds_to_samples(1 * units.minute, 16)
    960
    >>> seconds_to_samples(4 * units.second, 16.384 * units.kiloHertz)
    65536
    """
    return int((Quantity(x, 's') * rate).decompose().value)


def normalize_fft_params(series, kwargs=None, func=None):
    """Normalize a set of FFT parameters for processing

    This method reads the ``fftlength`` and ``overlap`` keyword arguments
    (presumed to be values in seconds), works out sensible defaults,
    then updates ``kwargs`` in place to include ``nfft`` and ``noverlap``
    as values in sample counts.

    If a ``window`` is given, the ``noverlap`` parameter will be set to the
    recommended overlap for that window type, if ``overlap`` is not given.

    If a ``window`` is given as a `str`, it will be converted to a
    `numpy.ndarray` containing the correct window (of the correct length).

    Parameters
    ----------
    series : `gwpy.timeseries.TimeSeries`
        the data that will be processed using an FFT-based method

    kwargs : `dict`
        the dict of keyword arguments passed by the user

    func : `callable`, optional
        the FFT method that will be called

    Examples
    --------
    >>> from numpy.random import normal
    >>> from gwpy.timeseries import TimeSeries
    >>> normalize_fft_params(TimeSeries(normal(size=1024), sample_rate=256))
    {'nfft': 1024, 'noverlap': 0}
    >>> normalize_fft_params(TimeSeries(normal(size=1024), sample_rate=256),
    ...                      {'window': 'hann'})
    {'window': array([  0.00000000e+00,   9.41235870e-06, ...,
         3.76490804e-05,   9.41235870e-06]), 'noverlap': 0, 'nfft': 1024}
    """
    # parse keywords
    if kwargs is None:
        kwargs = dict()
    samp = series.sample_rate
    fftlength = kwargs.pop('fftlength', None) or series.duration
    overlap = kwargs.pop('overlap', None)
    window = kwargs.pop('window', None)

    # parse function library and name
    if func is None:
        method = library = None
    else:
        method = func.__name__
        library = _fft_library(func)

    # fftlength -> nfft
    nfft = seconds_to_samples(fftlength, samp)

    # overlap -> noverlap
    noverlap = _normalize_overlap(overlap, window, nfft, samp, method=method)

    # create window
    window = _normalize_window(window, nfft, library, series.dtype)
    if window is not None:  # allow FFT methods to use their own defaults
        kwargs['window'] = window

    # create FFT plan for LAL
    if library == 'lal' and kwargs.get('plan', None) is None:
        from ._lal import generate_fft_plan
        kwargs['plan'] = generate_fft_plan(nfft, dtype=series.dtype)

    kwargs.update({
        'nfft': nfft,
        'noverlap': noverlap,
    })
    return kwargs


def _normalize_overlap(overlap, window, nfft, samp, method='welch'):
    """Normalise an overlap in physical units to a number of samples

    Parameters
    ----------
    overlap : `float`, `Quantity`, `None`
        the overlap in some physical unit (seconds)

    window : `str`
        the name of the window function that will be used, only used
        if `overlap=None` is given

    nfft : `int`
        the number of samples that will be used in the fast Fourier
        transform

    samp : `Quantity`
        the sampling rate (Hz) of the data that will be transformed

    method : `str`
        the name of the averaging method, default: `'welch'`, only
        used to return `0` for `'bartlett'` averaging

    Returns
    -------
    noverlap : `int`
        the number of samples to be be used for the overlap
    """
    if method == 'bartlett':
        return 0
    if overlap is None and isinstance(window, str):
        return recommended_overlap(window, nfft)
    if overlap is None:
        return 0
    return seconds_to_samples(overlap, samp)


def _normalize_window(window, nfft, library, dtype):
    """Normalise a window specification for a PSD calculation

    Parameters
    ----------
    window : `str`, `numpy.ndarray`, `None`
        the input window specification

    nfft : `int`
        the length of the Fourier transform, in samples

    library : `str`
        the name of the library that provides the PSD routine

    dtype : `type`
        the required type of the window array, only used if
        `library='lal'` is given

    Returns
    -------
    window : `numpy.ndarray`, `lal.REAL8Window`
        a numpy-, or `LAL`-format window array
    """
    if library == '_lal' and isinstance(window, numpy.ndarray):
        from ._lal import window_from_array
        return window_from_array(window, dtype=dtype)
    if library == '_lal':
        from ._lal import generate_window
        return generate_window(nfft, window=window, dtype=dtype)
    if isinstance(window, str):
        window = canonical_name(window)
    if isinstance(window, (str, tuple)):
        return get_window(window, nfft)
    return window


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

        # normalise FFT parameters for all libraries
        normalize_fft_params(data, kwargs=kwargs, func=method_func)

        return func(series, method_func, *args, **kwargs)

    return wrapped_func


# -- processing functions -----------------------------------------------------

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
    return _psdn(timeseries, method_func, *args, **kwargs)


def _psdn(timeseries, method_func, *args, **kwargs):
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
    # or just calculate PSD
    except ValueError:
        return method_func(timeseries, kwargs.pop('nfft'), *args, **kwargs)
    else:
        return method_func(timeseries, other, kwargs.pop('nfft'),
                           *args, **kwargs)


def _psd(bundle):
    """Calculate a single PSD for a spectrogram
    """
    series, method_func, args, kwargs = bundle
    psd_ = _psdn(series, method_func, *args, **kwargs)
    del psd_.epoch  # fixes Segmentation fault (no idea why it faults)
    return psd_


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
    normalize_fft_params(timeseries, kwargs=kwargs, func=method_func)
    nfft = kwargs['nfft']
    noverlap = kwargs['noverlap']

    # sanity check parameters
    if nstride > timeseries.size:
        raise ValueError("stride cannot be greater than the duration of "
                         "this TimeSeries")
    if nfft > nstride:
        raise ValueError("fftlength cannot be greater than stride")
    if noverlap >= nfft:
        raise ValueError("overlap must be less than fftlength")

    # define chunks
    tschunks = _chunk_timeseries(timeseries, nstride, noverlap)
    if other is not None:
        otherchunks = _chunk_timeseries(other, nstride, noverlap)
        tschunks = zip(tschunks, otherchunks)

    # bundle inputs for _psd
    inputs = [(chunk, method_func, args, kwargs) for chunk in tschunks]

    # calculate PSDs
    psds = mp_utils.multiprocess_with_queues(nproc, _psd, inputs)

    # recombobulate PSDs into a spectrogram
    return Spectrogram.from_spectra(*psds, epoch=epoch, dt=stride)


def _periodogram(bundle):
    """Calculate a single periodogram for a spectrogram
    """
    series, kwargs = bundle
    return scipy_periodogram(series, **kwargs)[1]


def spectrogram(timeseries, **kwargs):
    """Generate a spectrogram by stacking periodograms

    Each time bin of the resulting spectrogram is a PSD estimate using
    a :func:`scipy.signal.periodogram`
    """
    from ...spectrogram import Spectrogram

    # normalise FFT parameters
    normalize_fft_params(timeseries, kwargs=kwargs)

    # get params
    sampling = timeseries.sample_rate.to('Hz').value
    nproc = kwargs.pop('nproc', 1)
    nfft = kwargs.get('nfft')
    noverlap = kwargs.pop('noverlap')
    nstride = nfft - noverlap

    # sanity check parameters
    if noverlap >= nfft:
        raise ValueError("overlap must be less than fftlength")

    # define chunks
    chunks = []
    x = 0
    while x + nfft <= timeseries.size:
        y = min(timeseries.size, x + nfft)
        chunks.append((x, y))
        x += nstride

    tschunks = (timeseries.value[i:j] for i, j in chunks)

    # bundle inputs for _psd
    inputs = [(chunk, kwargs) for chunk in tschunks]

    # calculate PSDs with multiprocessing
    psds = mp_utils.multiprocess_with_queues(
        nproc,
        _periodogram,
        inputs,
    )

    # convert PSDs to array with spacing for averages
    numtimes = 1 + int((timeseries.size - nstride) / nstride)
    numfreqs = int(nfft / 2 + 1)
    data = numpy.zeros((numtimes, numfreqs), dtype=timeseries.dtype)
    data[:len(psds)] = psds

    # create output spectrogram
    unit = fft_utils.scale_timeseries_unit(
        timeseries.unit, scaling=kwargs.get('scaling', 'density'))
    out = Spectrogram(numpy.empty((numtimes, numfreqs),
                                  dtype=timeseries.dtype),
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
    x = 0
    step = nstride - int(noverlap // 2.)  # the first step is smaller
    nfft = nstride + noverlap
    while x + nstride <= series.size:
        y = x + nfft
        if y >= series.size:
            y = series.size  # pin to end of series
            x = y - nfft  # and work back to get the correct amount of data
        yield series[x:y]
        x += step
        step = nstride  # subsequent steps are the standard size


def _fft_library(method_func):
    mod = method_func.__module__.rsplit('.', 1)[-1]
    if mod == 'median_mean':
        return "lal"
    return mod
