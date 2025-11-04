# Copyright (c) 2014-2017 Louisiana State University
#               2017-2025 Cardiff University
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

"""Generate a time-frequency coherence spectrogram from time-series data.

This module contains the relevant methods to generate a
time-frequency coherence spectrogram from a pair of time-series.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from math import ceil
from typing import TYPE_CHECKING

from numpy import zeros

from .spectrogram import Spectrogram, SpectrogramList

if TYPE_CHECKING:
    from ..signal.window import WindowLike
    from ..timeseries import TimeSeries

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _from_timeseries(
    ts1: TimeSeries,
    ts2: TimeSeries,
    stride: float,
    fftlength: float | None = None,
    overlap: float | None = None,
    window: WindowLike = "hann",
    **kwargs,
) -> Spectrogram:
    """Generate a time-frequency coherence from a pair of time-series.

    For each `stride`, a PSD :class:`~gwpy.frequencyseries.FrequencySeries`
    is generated, with all resulting spectra stacked in time and returned.
    """
    # check sampling rates
    s1 = ts1.sample_rate.to("Hertz")
    s2 = ts2.sample_rate.to("Hertz")
    if s1 != s2:
        sampling = min(s1.value, s2.value)
        # resample higher rate series
        if s1.value == sampling:
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
    out = Spectrogram(
        zeros((nsteps, nfreqs)),
        epoch=ts1.epoch,
        dt=stride,
        f0=0,
        df=1 / fftlength,
        copy=True,
        unit="coherence",
    )

    if not nsteps:
        return out

    # stride through TimeSeries, recording PSDs as columns of spectrogram
    for step in range(nsteps):
        # find step TimeSeries
        idx = nstride * step
        idx_end = idx + nstride
        stepseries1 = ts1[idx:idx_end]
        stepseries2 = ts2[idx:idx_end]
        stepcoh = stepseries1.coherence(
            stepseries2,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )
        out.value[step] = stepcoh.value

    return out


def _from_timeseries_parallel(
    ts1: TimeSeries,
    ts2: TimeSeries,
    stride: float,
    fftlength: float | None = None,
    overlap: float | None = None,
    window: WindowLike = "hann",
    parallel: int = 1,
    **kwargs,
) -> Spectrogram:
    """Calculate the coherence `Spectrogram` between two `TimeSeries`.

    Parameters
    ----------
    ts1 : `~gwpy.timeseries.TimeSeries`
        First input time-series to process.

    ts2 : `~gwpy.timeseries.TimeSeries`
        Second input time-series to process.

    stride : `float`
        Number of seconds in single PSD (column of spectrogram).

    fftlength : `float`, optional
        Number of seconds in single FFT..

    overlap : `int`, optional
        Number of seconds of overlap between FFTs, defaults to no overlap

    window : `str`, `tuple`, optional
        Window function to apply to timeseries prior to FFT.

    parallel : `int`, optional
        Maximum number of independent frame reading processes,
        default is set to single-process file reading.

    kwargs
        Other keyword arguments passed to coherence method.

    Returns
    -------
    spectrogram : `~gwpy.spectrogram.Spectrogram`
        Time-frequency power spectrogram as generated from the
        input time-series.
    """
    # format FFT parameters
    if fftlength is None:
        fftlength = stride / 2.0

    # get size of spectrogram
    nsteps = int(ts1.size // (stride * ts1.sample_rate.value))
    parallel = min(nsteps, parallel)

    # single-process return
    if nsteps == 0 or parallel == 1:
        return _from_timeseries(
            ts1,
            ts2,
            stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            **kwargs,
        )

    # divide work into chunks
    stepperproc = ceil(nsteps / parallel)
    nsamp = [stepperproc * ts.sample_rate.value * stride for ts in (ts1, ts2)]

    # prepare arguments for each process
    chunks = []
    for i in range(parallel):
        start_idx = int(i * nsamp[0])
        end_idx = int((i + 1) * nsamp[0])
        if start_idx >= ts1.size:
            break
        chunks.append((
            (
                ts1[start_idx:end_idx],
                ts2[int(i * nsamp[1]) : int((i + 1) * nsamp[1])],
                stride,
                fftlength,
                overlap,
                window,
            ),
            kwargs,
        ))

    # process chunks in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=parallel) as executor:
        # submit all tasks
        futures = [
            executor.submit(
                _from_timeseries,
                *args_,
                **kwargs_,
            )
            for args_, kwargs_ in chunks
        ]

        # collect results
        data = [future.result() for future in futures]

    # format and return
    out = SpectrogramList(*data)
    out.sort(key=lambda spec: spec.epoch.gps)
    return out.join()
