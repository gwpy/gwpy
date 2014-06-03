# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""This module contains the relevant methods to generate a
time-frequency coherence spectrogram from a pair of time-series.
"""

from __future__ import division

from multiprocessing import (Process, Queue as ProcessQueue)
from math import ceil

from numpy import zeros

from astropy import units

from .. import version
from .core import (Spectrogram, SpectrogramList)
from ..spectrum import psd

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version
__date__ = ""


def _from_timeseries(ts1, ts2, stride, fftlength=None, fftstride=None,
                     window=None, **kwargs):
    """Generate a time-frequency coherence
    :class:`~gwpy.spectrogram.core.Spectrogram` from a pair of
    :class:`~gwpy.timeseries.core.TimeSeries`.

    For each `stride`, a PSD :class:`~gwpy.spectrum.core.Spectrum`
    is generated, with all resulting spectra stacked in time and returned.
    """
    # check sampling rates
    if ts1.sample_rate.to('Hertz') != ts2.sample_rate.to('Hertz'):
        sampling = min(ts1.sample_rate.value, ts2.sample_rate.value)
        # resample higher rate series
        if ts1.sample_rate.value == sampling:
            ts2 = ts2.resample(sampling)
        else:
            ts1 = ts1.resample(sampling)
    else:
        sampling = ts1.sample_rate.value

    # format FFT parameters
    if fftlength is None:
        fftlength = stride
    if fftstride is None:
        fftstride = fftlength
    dt = stride
    df = 1 / fftlength

    stride *= sampling

    # get size of spectrogram
    nsteps = int(ts1.size // stride)
    nfreqs = int(fftlength * sampling // 2 + 1)

    # generate output spectrogram
    out = Spectrogram(zeros((nsteps, nfreqs)), epoch=ts1.epoch,
                      f0=0, df=df, dt=dt, copy=True)
    out.unit = 'coherence'

    if not nsteps:
        return out

    # stride through TimeSeries, recording PSDs as columns of spectrogram
    for step in range(nsteps):
        # find step TimeSeries
        idx = stride * step
        idx_end = idx + stride
        stepseries1 = ts1[idx:idx_end]
        stepseries2 = ts2[idx:idx_end]
        stepcoh = stepseries1.coherence(stepseries2, fftlength=fftlength,
                                        fftstride=fftstride, window=window,
                                        **kwargs)
        out[step] = stepcoh.data

    return out


def from_timeseries(ts1, ts2, stride, fftlength=None, fftstride=None,
                    window=None, nproc=1, **kwargs):
    """Calculate the coherence `Spectrogram` between two `TimeSeries`.

    Parameters
    ----------
    timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
        input time-series to process.
    stride : `float`
        number of seconds in single PSD (column of spectrogram).
    fftlength : `float`
        number of seconds in single FFT.
    fftstride : `int`, optiona, default: fftlength
        number of seconds between FFTs.
    window : `timeseries.window.Window`, optional, default: `None`
        window function to apply to timeseries prior to FFT.
    nproc : `int`, default: ``1``
        maximum number of independent frame reading processes, default
        is set to single-process file reading.

    Returns
    -------
    spectrogram : :class:`~gwpy.spectrogram.core.Spectrogram`
        time-frequency power spectrogram as generated from the
        input time-series.
    """
    # format FFT parameters
    if fftlength is None:
        fftlength = stride
    if fftstride is None:
        fftstride = fftlength

    sampling = min(ts1.sample_rate.value, ts2.sample_rate.value)

    # get size of spectrogram
    nFFT = int(fftlength * sampling)
    nsteps = int(ts1.size // (stride * ts1.sample_rate.value))
    nproc = min(nsteps, nproc)

    # single-process return
    if nsteps == 0 or nproc == 1:
        return _from_timeseries(ts1, ts2, stride, fftlength=fftlength,
                                fftstride=fftstride, window=window, **kwargs)

    # wrap spectrogram generator
    def _specgram(q, ts):
        try:
            q.put(_from_timeseries(ts, ts2, stride, fftlength=fftlength,
                                   fftstride=fftstride, window=window,
                                   **kwargs))
        except Exception as e:
            q.put(e)

    # otherwise build process list
    stepperproc = int(ceil(nsteps / nproc))
    nsamp = [stepperproc * ts.sample_rate.value * stride for ts in (ts1, ts2)]

    queue = ProcessQueue(nproc)
    processlist = []
    for i in range(nproc):
        process = Process(target=_specgram,
                          args=(queue, ts1[i * nsamp[0]:(i + 1) * nsamp[0]],
                                ts2[i * nsamp[1]:(i + 1) * nsamp[1]]))
        process.daemon = True
        processlist.append(process)
        process.start()
        if ((i + 1) * nsamp[0]) >= ts1.size:
            break

    # get data
    data = []
    for process in processlist:
        result = queue.get()
        if isinstance(result, Exception):
            raise result
        else:
            data.append(result)

    # and block
    for process in processlist:
        process.join()

    # format and return
    out = SpectrogramList(*data)
    out.sort(key=lambda spec: spec.epoch.gps)
    return out.join()
