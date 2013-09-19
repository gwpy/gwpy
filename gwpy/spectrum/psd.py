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
    return welch(timeseries, segmentlength, segmentlength, window=window)


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
    if isinstance(segmentlength, units.Quantity):
        segmentlength = segmentlength.value
    if isinstance(overlap, units.Quantity):
        overlap = overlap.value
    lalts = timeseries.to_lal()
    lalwin = window is not None and window.to_lal() or None
    lalfs = lalspectrum._psd(method, lalts, segmentlength, overlap,
                             window=lalwin)
    spec = Spectrum.from_lal(lalfs)
    if timeseries.unit:
        spec.unit = timeseries.unit / units.Hertz
    else:
        spec.unit = 1 / units.Hertz
    return spec
