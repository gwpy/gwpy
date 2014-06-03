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
time-frequency spectrogram from a single time-series.
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


def _from_timeseries(timeseries, stride, fftlength=None, fftstride=None,
                     method='welch', window=None, plan=None):
    """Generate a time-frequency power spectral density
    :class:`~gwpy.spectrogram.core.Spectrogram` from a
    :class:`~gwpy.timeseries.core.TimeSeries`.

    For each `stride`, a PSD :class:`~gwpy.spectrum.core.Spectrum`
    is generate using the given `method`, with all resulting
    spectra stacked in time and returned.

    """
    # format FFT parameters
    if fftlength is None:
        fftlength = stride
    if fftstride is None:
        fftstride = fftlength
    dt = stride
    df = 1 / fftlength
    stride *= timeseries.sample_rate.value

    # get size of spectrogram
    nsteps = int(timeseries.size // stride)
    nfreqs = int(fftlength * timeseries.sample_rate.value // 2 + 1)

    # generate output spectrogram
    out = Spectrogram(zeros((nsteps, nfreqs)),
                      channel=timeseries.channel, epoch=timeseries.epoch,
                      f0=0, df=df, dt=dt, copy=True)
    try:
        out.unit = timeseries.unit / units.Hertz
    except KeyError:
        out.unit = 1 / units.Hertz

    if not nsteps:
        return out

    # stride through TimeSeries, recording PSDs as columns of spectrogram
    for step in range(nsteps):
        # find step TimeSeries
        idx = stride * step
        idx_end = idx + stride
        stepseries = timeseries[idx:idx_end]
        steppsd = stepseries.psd(fftlength, fftstride, method,
                                 window=window, plan=plan)
        out[step] = steppsd.data

    return out


def from_timeseries(timeseries, stride, fftlength=None, fftstride=None,
                    method='welch', window=None, plan=None, nproc=1):
    """Calculate the average power spectrogram of this `TimeSeries`
    using the specified average spectrum method.

    Parameters
    ----------
    timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
        input time-series to process.
    stride : `float`
        number of seconds in single PSD (column of spectrogram).
    fftlength : `float`
        number of seconds in single FFT.
    method : `str`, optional, default: 'welch'
        average spectrum method.
    fftstride : `int`, optiona, default: fftlength
        number of seconds between FFTs.
    window : `timeseries.window.Window`, optional, default: `None`
        window function to apply to timeseries prior to FFT.
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum,
        substitute type 'REAL8' as appropriate.
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

    # get size of spectrogram
    nFFT = int(fftlength * timeseries.sample_rate.value)
    nsteps = int(timeseries.size // (stride * timeseries.sample_rate.value))
    nproc = min(nsteps, nproc)

    # generate window and plan if needed
    try:
        from lal import lal
    except ImportError:
        pass
    else:
        if window is None:
            window = psd.generate_lal_window(nFFT, dtype=timeseries.dtype)
        if plan is None:
            plan = psd.generate_lal_fft_plan(nFFT, dtype=timeseries.dtype)

    # single-process return
    if nsteps == 0 or nproc == 1:
        return _from_timeseries(timeseries, stride, fftlength=fftlength,
                                fftstride=fftstride, method=method,
                                window=window)

    # wrap spectrogram generator
    def _specgram(q, ts):
        try:
            q.put(_from_timeseries(ts, stride, fftlength=fftlength,
                                   fftstride=fftstride, method=method,
                                   window=window, plan=plan))
        except Exception as e:
            q.put(e)

    # otherwise build process list
    stepperproc = int(ceil(nsteps / nproc))
    nsamp = stepperproc * timeseries.sample_rate.value * stride
    queue = ProcessQueue(nproc)
    processlist = []
    for i in range(nproc):
        process = Process(target=_specgram,
                          args=(queue, timeseries[i * nsamp:(i + 1) * nsamp]))
        process.daemon = True
        processlist.append(process)
        process.start()
        if ((i + 1) * nsamp) >= timeseries.size:
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
