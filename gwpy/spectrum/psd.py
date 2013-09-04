# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""A collection of average power spectral density calculation routines
"""

import numpy
from matplotlib import mlab

from astropy import units

from .core import Spectrum
from ..timeseries import window as tdwindow
from ..spectrogram import Spectrogram

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ["psd", "bartlett", "welch", "spectrogram"]


def psd(timeseries, method, *args, **kwargs):
    if method == 'bartlett':
        return bartlett(timeseries, *args, **kwargs)
    elif method == 'welch':
        return welch(timeseries, *args, **kwargs)
    else:
        raise NotImplementedError("Average spectrum method='%s' has not been "
                                  "implemented.")


def bartlett(timeseries, fft_length=None, window=tdwindow.kaiser_factory(24)):
    """Calculate the power spectral density of the given TimeSeries
    using the Bartlett average method.

    This method divides the data into chunks of length `fft_length`,
    a periodogram calculated for each, and the bin-by-bin mean returned.

    Parameters
    ---------
    timeseries: `~gwpy.timeseries.TimeSeries`
        Input TimeSeries data
    """
    fft_length = fft_length or timeseries.size
    return welch(timeseries, fft_length, 0, window=window)


def welch(timeseries, fft_length=None, overlap=0,
          window=tdwindow.kaiser_factory(24)):
    fft_length = fft_length or timeseries.size
    sampling = 1/float(timeseries.dt)
    fft_length = int(fft_length * sampling)
    overlap = int(overlap * sampling)
    psd,freqs = mlab.psd(timeseries, NFFT=fft_length, Fs=sampling,
                        noverlap=overlap, window=window)
    f0 = freqs[0]
    df = freqs[1]-freqs[0]
    return Spectrum(psd, f0=f0, df=df, name=timeseries.name,
                          unit=units.Unit("1/Hz"))


def spectrogram(timeseries, method, step, **kwargs):
    """Calculate the average power spectrogram of the given TimeSeries
    using the specified average spectrum method

    Parameters
    ----------

    timeseries : `~gwpy.data.TimeSeries`
        Input TimeSeries data
    method : str
        Average spectrum method
    step : float
        Length of single average spectrum (seconds)
    """
    # get number of time steps
    step_samp = step // float(timeseries.dt)
    nsteps = int(timeseries.size // step_samp)
    mismatch = (timeseries.size - nsteps * step_samp)
    if mismatch:
        warnings.warn("TimeSeries is %d samples too long for use in a "
                      "Spectrogram of step=%d samples, those samples will "
                      "not be used" % (mismatch, step_samp))
    # get number of frequencies
    fft_length = kwargs.pop('fft_length', step)
    fft_length_samp = fft_length // float(timeseries.dt)
    nfreqs = int(fft_length_samp // 2 + 1)

    out = Spectrogram(numpy.zeros((nsteps, nfreqs)), name=timeseries.name,
                      epoch=timeseries.epoch, dt=step)
    if not nsteps:
        return out

    for step in range(nsteps):
        idx = step_samp * step
        idx_end = idx + step_samp
        stepseries = timeseries[idx:idx_end]
        stepspectrum = psd(stepseries, method,
                           fft_length=fft_length, **kwargs)
        out.data[step,:] = stepspectrum.data[:,0]
    out.f0 = stepspectrum.f0
    out.df = stepspectrum.df
    if timeseries.unit:
        out.unit = timeseries.unit / units.Hertz
    else:
        out.unit = 1 / units.Hertz
    return out
