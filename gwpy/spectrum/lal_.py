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

"""Spectrum generation methods using LAL.
"""

import re
import warnings

import numpy

from ..window import Window
from .core import Spectrum
from .registry import register_method
from .utils import scale_timeseries_units
from ..utils import import_method_dependency
from .. import version
__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__version__ = version.version

# cache windows internally
LAL_WINDOWS = {}
LAL_FFTPLANS = {}
LAL_FFTPLAN_LEVEL = 1


# ---------------------------------------------------------------------------
# Utilities

def generate_lal_fft_plan(length, level=None,
                          dtype=numpy.dtype(numpy.float64)):
    """Build a :lalsuite:`REAL8FFTPlan` for a fast Fourier transform.

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
    from ..utils.lal import LAL_TYPE_STR_FROM_NUMPY
    from lal import lal
    global LAL_FFTPLANS
    laltype = LAL_TYPE_STR_FROM_NUMPY[dtype.type]
    try:
        plan = LAL_FFTPLANS[(length, laltype)]
    except KeyError:
        create = getattr(lal, 'CreateForward%sFFTPlan' % laltype)
        if level is None:
            level = LAL_FFTPLAN_LEVEL
        plan = LAL_FFTPLANS[(length, laltype)] = create(length, level)
    return plan


def generate_lal_window(length, type_=('kaiser', 24),
                        dtype=numpy.dtype(numpy.float64)):
    """Generate a time-domain window for use in a Fourier transform using
    the LIGO Algorithm Library routines.

    Parameters
    ----------
    length : `int`
        length of window in samples.
    type_ : `str`, `tuple`
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
    from ..utils.lal import LAL_TYPE_STR_FROM_NUMPY
    from lal import lal
    global LAL_WINDOWS
    laltype = LAL_TYPE_STR_FROM_NUMPY[dtype.type]
    wtype = isinstance(type_, (list, tuple)) and type_[0] or str(type_)
    try:
        window = LAL_WINDOWS[(length, wtype.lower(), laltype)]
    except KeyError:
        if isinstance(type_, (list, tuple)):
            try:
                args = type_[1:]
            except IndexError:
                args = []
        else:
            args = []
        wtype = wtype.islower() and wtype.title() or wtype
        create = getattr(lal, 'Create%s%sWindow' % (wtype, laltype))
        window = LAL_WINDOWS[(length, wtype.lower(), laltype)] = create(
            length, *args)
    return window


# ---------------------------------------------------------------------------
# LAL spectrum methods

def lal_psd(timeseries, segmentlength, noverlap=None, method='welch',
            window=None, plan=None):
    """Generate a PSD `Spectrum` using XLAL.

    Parameters
    ----------
    timeseries : :class:`~gwpy.timeseries.core.TimeSeries`
        input `TimeSeries` data.
    method : `str`
        average method.
    segmentlength : `int`
        number of samples in single average.
    noverlap : `int`
        number of samples to overlap between segments, defaults to 50%.
    window : `~gwpy.window.Window`, optional
        window function to apply to timeseries prior to FFT
    plan : :lalsuite:`REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum

    Returns
    -------
    Spectrum
        average power `Spectrum`
    """
    # get LAL
    lal = import_method_dependency('lal')
    from ..utils.lal import LAL_TYPE_STR_FROM_NUMPY
    # default to 50% overlap
    if noverlap is None:
        noverlap = int(segmentlength // 2)
    stride = segmentlength - noverlap
    # get cached window
    if window is None:
        window = generate_lal_window(segmentlength, dtype=timeseries.dtype)
    elif isinstance(window, (tuple, str)):
        window = generate_lal_window(segmentlength, type_=window,
                                     dtype=timeseries.dtype)
    elif isinstance(window, Window):
        window = window.to_lal()
    # get cached FFT plan
    if plan is None:
        plan = generate_lal_fft_plan(segmentlength, dtype=timeseries.dtype)

    method = method.lower()

    # check data length
    size = timeseries.size
    numsegs = 1 + int((size - segmentlength) / stride)
    if method == 'median-mean' and numsegs % 2:
        numsegs -= 1
        if not numsegs:
            raise ValueError("Cannot calculate median-mean spectrum with "
                             "this small a TimeSeries.")

    required = int((numsegs - 1) * stride + segmentlength)
    if size != required:
        warnings.warn("Data array is the wrong size for the correct number "
                      "of averages given the input parameters. The trailing "
                      "%d samples will not be used in this calculation."
                      % (size - required))
        timeseries = timeseries[:required]

    laltypestr = LAL_TYPE_STR_FROM_NUMPY[timeseries.dtype.type]

    # generate output spectrum
    try:
        unit = lal.lalStrainUnit
    except AttributeError:
        unit = lal.StrainUnit
    create = getattr(lal, 'Create%sFrequencySeries' % laltypestr)
    lalfs = create(timeseries.name, lal.LIGOTimeGPS(timeseries.epoch.gps), 0,
                   1 / segmentlength, unit, int(segmentlength // 2 + 1))

    # calculate medianmean spectrum
    if re.match('median-mean\Z', method, re.I):
        average_spectrum = getattr(lal,
                                   "%sAverageSpectrumMedianMean" % laltypestr)
    elif re.match('median\Z', method, re.I):
        average_spectrum = getattr(lal, "%sAverageSpectrumMedian" % laltypestr)
    elif re.match('welch\Z', method, re.I):
        average_spectrum = getattr(lal, "%sAverageSpectrumWelch" % laltypestr)
    else:
        raise NotImplementedError("Sorry, only 'median' and 'median-mean' "
                                  "and 'welch' average methods are available.")
    average_spectrum(lalfs, timeseries.to_lal(), segmentlength, stride,
                     window, plan)

    # format and return
    spec = Spectrum.from_lal(lalfs)
    spec.channel = timeseries.channel
    spec.unit = scale_timeseries_units(timeseries.unit, scaling='density')
    return spec


def lal_spectrum_factory(method):
    """Wrap the `lal_psd` function with a specific method argument.
    """
    def _spectrum(*args, **kwargs):
        kwargs['method'] = method
        return lal_psd(*args, **kwargs)
    return _spectrum


# register LAL methods without overriding scipy method
for _method in ['welch', 'median-mean', 'median']:
    try:
        register_method(lal_spectrum_factory(_method), _method)
    except KeyError:
        register_method(lal_spectrum_factory(_method), 'lal-%s' % _method)
