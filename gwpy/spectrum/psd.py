# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""A collection of average power spectral density calculation routines

Average-spectrum calculation routines are available for the following methods

    - :func:`Bartlett <bartlett>`
    - :func:`Welch <welch>`
    - :func:`Median-mean <median_mean>`
    - :func:`Median <median>`

Each of these methods utilises an existing method provided by the
LIGO Algorithm Library, wrapped into python as part of the `lal.spectrum`
module.
"""

import numpy
from matplotlib import mlab

from astropy import units

from lal.spectrum import averagespectrum as lalspectrum

from .core import Spectrum
from ..timeseries import window as tdwindow
from ..spectrogram import Spectrogram

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ['bartlett', 'welch', 'median_mean', 'median', 'spectrogram']


def bartlett(timeseries, segmentlength, window=None):
    """Calculate the power spectral density of the given `TimeSeries`
    using the Bartlett average method.

    This method divides the data into chunks of length `segmentlength`,
    a periodogram calculated for each, and the bin-by-bin mean returned.

    Parameters
    ---------
    timeseries: `TimeSeries`
        input `TimeSeries` data
    segmentlength : `int`
        number of samples in each average

    Returns
    -------
    Spectrum
        Bartlett-averaged `Spectrum`
    """
    return welch(timeseries, segmentlength, 0, window=window)


def welch(timeseries, segmentlength, overlap, window=None):
    """Calculate the power spectral density of the given `TimeSeries`
    using the Welch average method.

    For more details see :lalsuite:`XLALREAL8AverageSpectrumWelch`.

    Parameters
    ----------
    timeseries : `TimeSeries`
        input `TimeSeries` data
    method : `str`
        average method
    segmentlength : `int`
        number of samples in single average
    overlap : `int`
        number of samples between averages
    window : `timeseries.Window`, optional
        window function to apply to timeseries prior to FFT

    Returns
    -------
    Spectrum
        Welch-averaged `Spectrum`
    """
    return _lal_psd(timeseries, 'welch', segmentlength, overlap, window=window)


def median_mean(timeseries, segmentlength, overlap, window=None):
    """Calculate the power spectral density of the given `TimeSeries`
    using the median-mean average method.

    For more details see :lalsuite:`XLALREAL8AverageSpectrumMedianMean`.

    Parameters
    ----------
    timeseries : `TimeSeries`
        input `TimeSeries` data
    segmentlength : `int`
        number of samples in single average
    overlap : `int`
        number of samples between averages
    window : `timeseries.Window`, optional
        window function to apply to timeseries prior to FFT

    Returns
    -------
    Spectrum
        median-mean-averaged `Spectrum`
    """
    return _lal_psd(timeseries, 'medianmean', segmentlength, overlap,
                    window=window)


def median(timeseries, segmentlength, overlap, window=None):
    """Calculate the power spectral density of the given `TimeSeries`
    using the median-mean average method.

    For more details see :lalsuite:`XLALREAL8AverageSpectrumMean`.

    Parameters
    ----------
    timeseries : `TimeSeries`
        input `TimeSeries` data
    segmentlength : `int`
        number of samples in single average
    overlap : `int`
        number of samples between averages
    window : `timeseries.Window`, optional
        window function to apply to timeseries prior to FFT

    Returns
    -------
    Spectrum
        median-mean-averaged `Spectrum`
    """
    return _lal_psd(timeseries, 'medianmean', segmentlength, overlap,
                    window=window)



def _lal_psd(timeseries, method, segmentlength, overlap, window=None):
    """Internal wrapper to the `lal.spectrum.psd` function

    This function handles the conversion between GWpy `TimeSeries` and
    XLAL ``TimeSeries``, (e.g. :lalsuite:`XLALREAL8TimeSeries`).

    Parameters
    ----------
    timeseries : `TimeSeries`
        input `TimeSeries` data
    method : `str`
        average method
    segmentlength : `int`
        number of samples in single average
    overlap : `int`
        number of samples between averages
    window : `timeseries.Window`, optional
        window function to apply to timeseries prior to FFT

    Returns
    -------
    Spectrum
        average power `Spectrum`
    """
    lalts = timeseries.to_lal()
    lalwin = window and window.to_lal() or None
    lalfs = lalspectrum._psd(method, lalts, segmentlength, overlap,
                         window=window)
    return Spectrum.from_lal(lalfs)


def spectrogram(timeseries, method, step, segmentlength, overlap=0,
                window=None):
    """Calculate the average power spectrogram of the given TimeSeries
    using the specified average spectrum method

    Parameters
    ----------

    timeseries : `~gwpy.data.TimeSeries`
        Input TimeSeries data
    method : `str`
        Average spectrum method
    step : `int`
        number of samples for single average spectrum
    segmentlength : `int`
        number of samples in single average
    overlap : `int`
        number of samples between averages
    window : `timeseries.Window`, optional
        window function to apply to timeseries prior to FFT
    """
    # get number of time steps
    nsteps = int(timeseries.size // step)
    mismatch = (timeseries.size - nsteps * step)
    if mismatch:
        warnings.warn("TimeSeries is %d samples too long for use in a "
                      "Spectrogram of step=%d samples, those samples will "
                      "not be used" % (mismatch, step_samp))
    # get number of frequencies
    nfreqs = int(segmentlength // 2 + 1)

    out = Spectrogram(numpy.zeros((nsteps, nfreqs)), name=timeseries.name,
                      epoch=timeseries.epoch, dt=step)
    if not nsteps:
        return out

    for step in range(nsteps):
        idx = step_samp * step
        idx_end = idx + step_samp
        stepseries = timeseries[idx:idx_end]
        stepspectrum = _lal_psd(stepseries, method, segmentlength, overlap,
                                window=window)
        out.data[step,:] = stepspectrum.data[:,0]
    out.f0 = stepspectrum.f0
    out.df = stepspectrum.df
    if timeseries.unit:
        out.unit = timeseries.unit / units.Hertz
    else:
        out.unit = 1 / units.Hertz
    return out
