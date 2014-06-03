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
from scipy import signal

from astropy import units

from .core import Spectrum
from ..window import Window

from .. import version
__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__version__ = version.version

__all__ = ['bartlett', 'welch', 'median_mean', 'median', 'spectrogram']

# cache windows internally
LAL_WINDOWS = {}
LAL_FFTPLANS = {}
LAL_FFTPLAN_LEVEL = 1


def bartlett(timeseries, segmentlength, window=None, plan=None):
    """Calculate the power spectral density of the given `TimeSeries`
    using the Bartlett average method.

    This method divides the data into chunks of length `segmentlength`,
    a periodogram calculated for each, and the bin-by-bin mean returned.

    Parameters
    ----------
    timeseries: `TimeSeries`
        input `TimeSeries` data
    segmentlength : `int`
        number of samples in each average
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum

    Returns
    -------
    Spectrum
        Bartlett-averaged `Spectrum`
    """
    return welch(timeseries, segmentlength, segmentlength, window=window,
                 plan=None)


def welch(timeseries, segmentlength, overlap, window=None, plan=None):
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
    window : `~gwpy.window.Window`, optional
        window function to apply to timeseries prior to FFT
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum

    Returns
    -------
    Spectrum
        Welch-averaged `Spectrum`
    """
    try:
        return lal_psd(timeseries, 'welch', segmentlength, overlap,
                       window=window, plan=None)
    except ImportError:
        return scipy_psd(timeseries, 'welch', segmentlength, overlap,
                         window=window)


def median_mean(timeseries, segmentlength, overlap, window=None, plan=None):
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
    window : `~gwpy.window.Window`, optional
        window function to apply to timeseries prior to FFT
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum

    Returns
    -------
    Spectrum
        median-mean-averaged `Spectrum`
    """
    return lal_psd(timeseries, 'medianmean', segmentlength, overlap,
                    window=window, plan=None)


def median(timeseries, segmentlength, overlap, window=None, plan=None):
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
    window : `~gwpy.window.Window`, optional
        window function to apply to timeseries prior to FFT
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum

    Returns
    -------
    Spectrum
        median-mean-averaged `Spectrum`
    """
    return lal_psd(timeseries, 'medianmean', segmentlength, overlap,
                    window=window, plan=None)


def lal_psd(timeseries, method, segmentlength, overlap, window=None, plan=None):
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
    window : `~gwpy.window.Window`, optional
        window function to apply to timeseries prior to FFT
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum

    Returns
    -------
    Spectrum
        average power `Spectrum`
    """
    try:
        from lal.spectrum import averagespectrum as lalspectrum
    except ImportError as e:
        raise ImportError('%s. Try using gwpy.spectrum.scipy_psd instead'
                          % str(e))
    else:
        from lal import (lal, utils as lalutils)
    if isinstance(segmentlength, units.Quantity):
        segmentlength = segmentlength.value
    if isinstance(overlap, units.Quantity):
        overlap = overlap.value
    lalts = timeseries.to_lal()
    stype = lalutils.dtype(lalts)
    # get cached window
    if window is None:
        window = generate_lal_window(segmentlength, dtype=timeseries.dtype)
    elif isinstance(window, Window):
        window = window.to_lal()
    # get cached FFT plan
    if plan is None:
        plan = generate_lal_fft_plan(segmentlength, dtype=timeseries.dtype)
    # generate average spectrum
    lalfs = lalspectrum._psd(method, lalts, segmentlength, overlap,
                             window=window, plan=plan)
    # format and return
    spec = Spectrum.from_lal(lalfs)
    spec.channel = timeseries.channel
    if timeseries.unit:
        spec.unit = timeseries.unit ** 2 / units.Hertz
    else:
        spec.unit = 1 / units.Hertz
    return spec


def scipy_psd(timeseries, method, segmentlength, overlap, window=('kaiser', 24)):
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
    window : `~gwpy.window.Window`, optional
        window function to apply to timeseries prior to FFT

    Returns
    -------
    Spectrum
        average power `Spectrum`
    """
    methods = ['welch', 'bartlett']
    if method.lower() not in methods:
        raise ValueError("'method' must be one of: '%s'" % "','".join(methods))
    if isinstance(segmentlength, units.Quantity):
        segmentlength = segmentlength.value
    if isinstance(overlap, units.Quantity):
        overlap = overlap.value
    f, psd_ = signal.welch(timeseries.data, fs=timeseries.sample_rate.value,
                           window=window, nperseg=segmentlength,
                           noverlap=(segmentlength-overlap))
    spec = psd_.view(Spectrum)
    spec.name = timeseries.name
    spec.epoch = timeseries.epoch
    spec.channel = timeseries.channel
    spec.f0 = f[0]
    spec.df = f[1]-f[0]
    if timeseries.unit:
        spec.unit = timeseries.unit ** 2 / units.Hertz
    else:
        spec.unit = 1 / units.Hertz
    return spec


def generate_lal_fft_plan(length, level=None,
                          dtype=numpy.dtype(numpy.float64)):
    """Build a plan to use when performing a Fourier transform using
    the LIGO Algorithm Library routines.

    Parameters
    ----------
    length : `int`
        number of samples to plan for in each FFT.
    level : `int`, optional
        amount of work to do when planning the FFT, default set by
        LAL_FFTPLAN_LEVEL module variable.
    dtype : :class:`numpy.dtype`
        numeric type of data to plan for, default `numpy.dtype(numpy.float64)`

    Returns
    -------
    :lalsuite:`REAL8FFTPlan`
        FFT plan of the relevant data type
    """
    from lal import (lal, utils as lalutils)
    global LAL_FFTPLANS
    ltype = lalutils.LAL_TYPE_FROM_NUMPY[dtype.type]
    laltypestr = lalutils.LAL_TYPE_STR[ltype]
    try:
        plan = LAL_FFTPLANS[(length, laltypestr)]
    except KeyError:
        create = getattr(lal, 'CreateForward%sFFTPlan' % laltypestr)
        if level is None:
           level = LAL_FFTPLAN_LEVEL
        plan = LAL_FFTPLANS[(length, laltypestr)] = create(length, level)
    return plan


def generate_lal_window(length, type=('kaiser', 24),
                        dtype=numpy.dtype(numpy.float64)):
    """Generate a time-domain window for use in a Fourier transform using
    the LIGO Algorithm Library routines.

    Parameters
    ----------
    length : `int`
        length of window in samples.
    type : `str`, `tuple`
        name of window to generate, default: ``('kaiser', 24)``. Give
        `str` for simple windows, or tuple of ``(name, *args)`` for
         complicated windows
    dtype : :class:`numpy.dtype`
        numeric type of window, default `numpy.dtype(numpy.float64)`

    Returns
    -------
    `window` : XLAL window
        time-domain window to use for FFT
    """
    from lal import (lal, utils as lalutils)
    global LAL_WINDOWS
    ltype = lalutils.LAL_TYPE_FROM_NUMPY[dtype.type]
    laltypestr = lalutils.LAL_TYPE_STR[ltype]
    wtype = isinstance(type, (list, tuple)) and type[0] or str(type)
    try:
        window = LAL_WINDOWS[(length, wtype.lower(), laltypestr)]
    except KeyError:
        if isinstance(type, (list, tuple)):
            try:
                args = type[1:]
            except IndexError:
                args = []
        else:
            args = []
        wtype = wtype.islower() and wtype.title() or wtype
        create = getattr(lal, 'Create%s%sWindow' % (wtype, laltypestr))
        window = LAL_WINDOWS[(length, wtype.lower(), laltypestr)] = create(
                                                                 length, *args)
    return window
