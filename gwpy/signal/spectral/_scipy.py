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

"""GWpy API to the scipy.signal FFT routines."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy
import scipy.signal
from packaging.version import Version
from scipy import __version__ as scipy_version

from ...frequencyseries import FrequencySeries
from . import _registry as fft_registry
from ._utils import scale_timeseries_unit

if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray

    from ...timeseries import TimeSeries
    from ..window import WindowLike

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- density scaling methods ---------

def _spectral_density(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    name: str | None = None,
    sdfunc: Callable = scipy.signal.welch,
    **kwargs,
) -> FrequencySeries:
    """Calculate a generic spectral density of this `TimeSeries`."""
    if (
        Version(scipy_version) >= Version("1.16.0a0")
        and (other := kwargs.get("y")) is not None
        and timeseries.size != other.size
    ):
        # manually zero-pad the shorter array, see
        # https://github.com/scipy/scipy/issues/23036
        if (a := timeseries.size) < (b := other.size):
            timeseries = timeseries.pad((0, b - a), mode="constant")
        else:
            kwargs["y"] = numpy.pad(other, (0, a - b), mode="constant")

    # compute spectral density
    freqs, psd_ = sdfunc(
        timeseries.value,
        fs=timeseries.sample_rate.decompose().value,
        nperseg=segmentlength,
        noverlap=noverlap,
        **kwargs,
    )
    # generate FrequencySeries and return
    unit = scale_timeseries_unit(
        timeseries.unit,
        kwargs.get("scaling", "density"),
    )
    return FrequencySeries(
        psd_,
        unit=unit,
        frequencies=freqs,
        name=(name or timeseries.name),
        epoch=timeseries.epoch,
        channel=timeseries.channel,
    )


def welch(
    timeseries: TimeSeries,
    segmentlength: int,
    average: str = "mean",
    **kwargs,
) -> FrequencySeries:
    """Calculate a PSD using Welch's method."""
    return _spectral_density(
        timeseries,
        segmentlength,
        average=average,
        **kwargs,
    )


def bartlett(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int = 0,
    **kwargs,
) -> FrequencySeries:
    """Calculate a PSD using Bartlett's method."""
    return _spectral_density(
        timeseries,
        segmentlength,
        noverlap=noverlap,
        **kwargs,
    )


def median(
    timeseries: TimeSeries,
    segmentlength: int,
    average: str = "median",
    **kwargs,
) -> TimeSeries:
    """Calculate a PSD using Welch's method with a median average."""
    return _spectral_density(
        timeseries,
        segmentlength,
        average=average,
        **kwargs,
    )


# register
for func in (
    welch,
    bartlett,
    median,
):
    fft_registry.register_method(func, name=func.__name__)
    # DEPRECATED:
    fft_registry.register_method(func, name=f"scipy-{func.__name__}")


# -- others --------------------------

def rayleigh(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int = 0,
    window: WindowLike = "hann",
):
    """Calculate a Rayleigh statistic spectrum.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Input `TimeSeries` data.

    segmentlength : `int`
        Number of samples in single average.

    noverlap : `int`
        Number of samples to overlap between segments, passing `None` will
        choose based on the window method, default: ``0``.

    window : `str`, `numpy.ndarray`, optional
        Window function to apply to ``timeseries`` prior to FFT,
        see :func:`scipy.signal.get_window` for details on acceptable
        formats.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.
    """
    stepsize = segmentlength - noverlap
    if noverlap:
        numsegs = 1 + int(
            (timeseries.size - segmentlength)
            / float(noverlap),
        )
    else:
        numsegs = int(timeseries.size // segmentlength)
    tmpdata: NDArray[numpy.float64] = numpy.ndarray(
        (numsegs, int(segmentlength // 2 + 1)),
    )
    for i in range(numsegs):
        tmpdata[i, :] = welch(
            timeseries[i * stepsize:i * stepsize + segmentlength],
            segmentlength,
            window=window,
        )
    std = tmpdata.std(axis=0)
    mean = tmpdata.mean(axis=0)
    return FrequencySeries(
        std / mean,
        unit="",
        copy=False,
        f0=0,
        epoch=timeseries.epoch,
        df=timeseries.sample_rate.value / segmentlength,
        channel=timeseries.channel,
        name=f"Rayleigh spectrum of {timeseries.name}",
    )


def csd(
    timeseries: TimeSeries,
    other: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    **kwargs,
) -> TimeSeries:
    """Calculate the CSD of two `TimeSeries` using Welch's method.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Time-series of data.

    other : `~gwpy.timeseries.TimeSeries`
        Time-series of data.

    segmentlength : `int`
        Number of samples in single average.

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    kwargs
        Other keyword arguments are passed to :meth:`scipy.signal.csd`

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.

    See Also
    --------
    scipy.signal.csd
    """
    return _spectral_density(
        timeseries,
        segmentlength,
        y=other.value,
        noverlap=noverlap,
        name=f"{timeseries.name}---{other.name}",
        sdfunc=scipy.signal.csd,
        **kwargs,
    )


def coherence(
    timeseries: TimeSeries,
    other: TimeSeries,
    segmentlength: int,
    downsample: bool | None = None,
    noverlap: int | None = None,
    **kwargs,
) -> FrequencySeries:
    """Calculate the coherence between two `TimeSeries` using Welch's method.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Time-series of data.

    other : `~gwpy.timeseries.TimeSeries`
        Time-series of data.

    segmentlength : `int`
        Number of samples in single average..

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    downsample : `bool`
        Downsample the series with higher sampling frequency? SciPy assumes
        that both TimeSeries have the same rate.

    kwargs
        Other keyword arguments are passed to :meth:`scipy.signal.coherence`.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.

    See Also
    --------
    scipy.signal.coherence
    """
    # Should we warn about unequal sampling frequencies?
    warn_fs = False

    if downsample is None:
        warn_fs = True
        downsample = True

    if timeseries.sample_rate != other.sample_rate:
        # scipy assumes a single sampling frequency
        if not downsample:
            msg = (
                "Cannot calculate coherence when sampling "
                "frequencies are unequal"
            )
            raise ValueError(msg)
        if warn_fs:
            warnings.warn(
                "Sampling frequencies are unequal. Higher "
                "frequency series will be downsampled before "
                "coherence is calculated",
                category=UserWarning,
                stacklevel=2,
            )
        # downsample the one with the higher rate
        if timeseries.sample_rate > other.sample_rate:
            timeseries = timeseries.resample(other.sample_rate)
        else:
            other = other.resample(timeseries.sample_rate)

    # calculate CSD
    out = _spectral_density(
        timeseries,
        segmentlength,
        y=other.value,
        noverlap=noverlap,
        name=f"Coherence between {timeseries.name} and {other.name}",
        sdfunc=scipy.signal.coherence,
        **kwargs,
    )
    out.override_unit("coherence")
    return out
