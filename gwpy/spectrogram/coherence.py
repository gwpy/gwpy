# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

from multiprocessing import (Process, Queue as ProcessQueue)
from math import ceil

from numpy import zeros

from .spectrogram import (Spectrogram, SpectrogramList)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _from_timeseries(ts1, ts2, stride, fftlength=None, overlap=None,
                     window=None, **kwargs):
    """Generate a time-frequency coherence
    :class:`~gwpy.spectrogram.Spectrogram` from a pair of
    :class:`~gwpy.timeseries.TimeSeries`.

    For each `stride`, a PSD :class:`~gwpy.frequencyseries.FrequencySeries`
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
    if overlap is None:
        overlap = 0

    nstride = int(stride * sampling)

    # get size of spectrogram
    nsteps = int(ts1.size // nstride)
    nfreqs = int(fftlength * sampling // 2 + 1)

    # generate output spectrogram
    out = Spectrogram(zeros((nsteps, nfreqs)), epoch=ts1.epoch, dt=stride,
                      f0=0, df=1/fftlength, copy=True, unit='coherence')

    if not nsteps:
        return out

    # stride through TimeSeries, recording PSDs as columns of spectrogram
    for step in range(nsteps):
        # find step TimeSeries
        idx = nstride * step
        idx_end = idx + nstride
        stepseries1 = ts1[idx:idx_end]
        stepseries2 = ts2[idx:idx_end]
        stepcoh = stepseries1.coherence(stepseries2, fftlength=fftlength,
                                        overlap=overlap, window=window,
                                        **kwargs)
        out.value[step] = stepcoh.value

    return out


def from_timeseries(ts1, ts2, stride, fftlength=None, overlap=None,
                    window=None, nproc=1, **kwargs):
    """Calculate the coherence `Spectrogram` between two `TimeSeries`.

    Parameters
    ----------
    timeseries : :class:`~gwpy.timeseries.TimeSeries`
        input time-series to process.
    stride : `float`
        number of seconds in single PSD (column of spectrogram).
    fftlength : `float`
        number of seconds in single FFT.
    overlap : `int`, optiona, default: fftlength
        number of seconds of overlap between FFTs, defaults to no overlap
    window : `timeseries.window.Window`, optional, default: `None`
        window function to apply to timeseries prior to FFT.
    nproc : `int`, default: ``1``
        maximum number of independent frame reading processes, default
        is set to single-process file reading.

    Returns
    -------
    spectrogram : :class:`~gwpy.spectrogram.Spectrogram`
        time-frequency power spectrogram as generated from the
        input time-series.
    """
    # format FFT parameters
    if fftlength is None:
        fftlength = stride / 2.

    # get size of spectrogram
    nsteps = int(ts1.size // (stride * ts1.sample_rate.value))
    nproc = min(nsteps, nproc)

    # single-process return
    if nsteps == 0 or nproc == 1:
        return _from_timeseries(ts1, ts2, stride, fftlength=fftlength,
                                overlap=overlap, window=window, **kwargs)

    # wrap spectrogram generator
    def _specgram(queue_, tsa, tsb):
        try:
            queue_.put(_from_timeseries(tsa, tsb, stride, fftlength=fftlength,
                                        overlap=overlap, window=window,
                                        **kwargs))
        except Exception as exc:  # pylint: disable=broad-except
            queue_.put(exc)

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
