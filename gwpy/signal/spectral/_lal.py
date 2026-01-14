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

"""GWpy API to the LAL FFT routines.

See the `LAL TimeFreqFFT.h documentation
<https://lscsoft.docs.ligo.org/lalsuite/lal/group___time_freq_f_f_t__h.html>`_
for more details

This module is deprecated and will be removed in a future release.
"""

from __future__ import annotations

import re
import warnings
from typing import (
    TYPE_CHECKING,
    Literal,
    cast,
)

import numpy
from astropy import units

from ...frequencyseries import FrequencySeries
from ..window import canonical_name
from . import _registry as fft_registry
from ._utils import scale_timeseries_unit

if TYPE_CHECKING:
    from numpy.typing import DTypeLike

    from ...timeseries import TimeSeries
    from ...utils.lal import (
        LALFFTPlanType,
        LALWindowType,
    )
    from ..window import WindowLike

    LALAverageSpectrumMethods = Literal[
        "AverageSpectrumMedianMean",
        "AverageSpectrumMedian",
        "AverageSpectrumWelch",
    ]

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"

# cache windows and FFT plans internally
LAL_WINDOWS: dict[tuple[int, str, str], LALWindowType] = {}
LAL_FFTPLANS: dict[tuple[int, bool, str], LALFFTPlanType] = {}

#: Default FFT plan level for LAL FFT plans
LAL_FFTPLAN_LEVEL = 1


# -- utilities -----------------------

def generate_fft_plan(
    length: int,
    level: int | None = None,
    dtype: DTypeLike = "float64",
    *,
    forward: bool = True,
) -> LALFFTPlanType:
    """Build a `REAL8FFTPlan` for a fast Fourier transform.

    Parameters
    ----------
    length : `int`
        Number of samples to plan for in each FFT.

    level : `int`, optional
        Amount of work to do when planning the FFT, default set by
        `LAL_FFTPLAN_LEVEL` module variable.

    dtype : :class:`numpy.dtype`, `type`, `str`, optional
        Numeric type of data to plan for

    forward : bool, optional, default: `True`
        Whether to create a forward or reverse FFT plan.

    Returns
    -------
    plan : `REAL8FFTPlan` or similar
        FFT plan of the relevant data type.
    """
    from ...utils.lal import find_typed_function, to_lal_type_str

    # generate key for caching plan
    laltype = to_lal_type_str(dtype)
    key = (length, bool(forward), laltype)

    # find existing plan
    try:
        return LAL_FFTPLANS[key]
    # or create one
    except KeyError:
        create = find_typed_function(dtype, "Create", "FFTPlan")
        if level is None:
            level = LAL_FFTPLAN_LEVEL
        LAL_FFTPLANS[key] = create(length, int(bool(forward)), level)
        return LAL_FFTPLANS[key]


def generate_window(
    length: int,
    window: WindowLike | None = None,
    dtype: DTypeLike | None = "float64",
) -> LALWindowType:
    """Generate a time-domain window for use in a LAL FFT.

    Parameters
    ----------
    length : `int`
        Length of window in samples.

    window : `str`, `tuple`
        Name of window to generate, default: ``('kaiser', 24)``. Give
        `str` for simple windows, or tuple of ``(name, *args)`` for
        complicated windows.

    dtype : `numpy.dtype`
        Numeric type of window, default ``numpy.dtype(numpy.float64)``.

    Returns
    -------
    `window` : `REAL8Window` or similar
        Time-domain window to use for FFT.
    """
    from ...utils.lal import find_typed_function, to_lal_type_str

    if window is None:
        window = ("kaiser", 24)

    if dtype is None:
        dtype = float

    # generate key for caching window
    laltype = to_lal_type_str(dtype)
    key = (length, str(window), laltype)

    # find existing window
    try:
        return LAL_WINDOWS[key]
    # or create one
    except KeyError:
        # handle arrays directly
        if isinstance(window, numpy.ndarray):
            return window_from_array(window, dtype=dtype)
        # parse window as name and arguments, e.g. ('kaiser', 24)
        if isinstance(window, list | tuple):
            window, beta = window
        else:
            beta = 0
        window = canonical_name(str(window))
        # create window
        create = find_typed_function(dtype, "CreateNamed", "Window")
        LAL_WINDOWS[key] = create(window, beta, length)
        return LAL_WINDOWS[key]


def window_from_array(
    array: numpy.ndarray,
    dtype: DTypeLike | None = None,
) -> LALWindowType:
    """Convert a `numpy.ndarray` into a LAL `Window` object."""
    from ...utils.lal import find_typed_function

    if dtype is None:
        dtype = array.dtype

    # create sequence
    seq = find_typed_function(dtype, "Create", "Sequence")(array.size)
    seq.data = numpy.asarray(array, dtype=dtype)

    # create window from sequence
    return find_typed_function(dtype, "Create", "WindowFromSequence")(seq)


# -- spectrumm methods ---------------

def _lal_spectrum(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    method: str = "welch",
    window: LALWindowType | WindowLike | None = None,
    plan: LALFFTPlanType | None = None,
) -> FrequencySeries:
    """Generate a PSD `FrequencySeries` using |lal|_.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Input `TimeSeries` data.

    segmentlength : `int`
        Number of samples in single average.

    method : `str`
        Average PSD method.

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    window : `lal.REAL8Window`, optional
        Window to apply to timeseries prior to FFT.

    plan : `lal.REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.
    """
    import lal

    from ...utils.lal import (
        LALWindowType,
        find_typed_function,
    )

    # default to 50% overlap
    if noverlap is None:
        noverlap = int(segmentlength // 2)
    stride = segmentlength - noverlap

    # get window
    if not isinstance(window, LALWindowType):
        window = generate_window(
            segmentlength,
            window=window,
            dtype=timeseries.dtype,
        )

    # get FFT plan
    if plan is None:
        plan = generate_fft_plan(segmentlength, dtype=timeseries.dtype)

    method = method.lower()

    # check data length
    size = timeseries.size
    numsegs = 1 + int((size - segmentlength) / stride)
    if method == "median-mean" and numsegs % 2:
        numsegs -= 1
        if not numsegs:
            msg = "Cannot calculate median-mean spectrum with this small a TimeSeries."
            raise ValueError(msg)

    required = int((numsegs - 1) * stride + segmentlength)
    if size != required:
        warnings.warn(
            "Data array is the wrong size for the correct number "
            "of averages given the input parameters. The trailing "
            f"{size - required} samples will not be used in this calculation.",
            stacklevel=2,
        )
        timeseries = timeseries[:required]

    # generate output spectrum
    create = find_typed_function(timeseries.dtype, "Create", "FrequencySeries")
    lalfs = create(
        timeseries.name or "",
        lal.LIGOTimeGPS(timeseries.t0.value),
        0,
        1 / segmentlength,
        lal.StrainUnit,
        int(segmentlength // 2 + 1),
    )

    # find LAL method (e.g. median-mean -> lal.REAL8AverageSpectrumMedianMean)
    methodname = cast(
        "LALAverageSpectrumMethods",
        f"AverageSpectrum{''.join(map(str.title, re.split('[-_]', method)))}",
    )
    spec_func = find_typed_function(
        timeseries.dtype,
        "",
        methodname,
    )

    # calculate spectrum
    spec_func(lalfs, timeseries.to_lal(), segmentlength, stride, window, plan)

    # format and return
    spec = FrequencySeries.from_lal(lalfs)
    spec.name = timeseries.name
    spec.channel = timeseries.channel
    spec.override_unit(
        scale_timeseries_unit(
            timeseries.unit or units.dimensionless_unscaled,
            scaling="density",
        ),
    )
    return spec


def welch(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    window: WindowLike | None = None,
    plan: LALFFTPlanType | None = None,
) -> FrequencySeries:
    """Calculate an PSD of this `TimeSeries` using Welch's method.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Input `TimeSeries` data.

    segmentlength : `int`
        Number of samples in single average.

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    window : `tuple`, `str`, optional
        Window parameters to apply to timeseries prior to FFT.

    plan : `REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.

    See Also
    --------
    lal.REAL8AverageSpectrumWelch
    """
    return _lal_spectrum(
        timeseries,
        segmentlength,
        noverlap=noverlap,
        method="welch",
        window=window,
        plan=plan,
    )


def bartlett(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,  # noqa: ARG001
    window: WindowLike | None = None,
    plan: LALFFTPlanType | None = None,
) -> FrequencySeries:
    # pylint: disable=unused-argument
    """Calculate an PSD of this `TimeSeries` using Bartlett's method.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Input `TimeSeries` data.

    segmentlength : `int`
        Number of samples in single average.

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    window : `tuple`, `str`, optional
        Window parameters to apply to timeseries prior to FFT.

    plan : `REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`

    See Also
    --------
    lal.REAL8AverageSpectrumWelch
    """
    return _lal_spectrum(
        timeseries,
        segmentlength,
        noverlap=0,
        method="welch",
        window=window,
        plan=plan,
    )


def median(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    window: WindowLike | None = None,
    plan: LALFFTPlanType | None = None,
) -> FrequencySeries:
    """Calculate a PSD of this `TimeSeries` using a median average method.

    The median average is similar to Welch's method, using a median average
    rather than mean.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Input `TimeSeries` data.

    segmentlength : `int`
        Number of samples in single average.

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    window : `tuple`, `str`, optional
        Window parameters to apply to timeseries prior to FFT.

    plan : `REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.

    See Also
    --------
    lal.REAL8AverageSpectrumMedian
    """
    return _lal_spectrum(
        timeseries,
        segmentlength,
        noverlap=noverlap,
        method="median",
        window=window,
        plan=plan,
    )


def median_mean(
    timeseries: TimeSeries,
    segmentlength: int,
    noverlap: int | None = None,
    window: WindowLike | None = None,
    plan: LALFFTPlanType | None = None,
) -> FrequencySeries:
    """Calculate a PSD of this `TimeSeries` using a median-mean average method.

    The median-mean average method divides overlapping segments into "even"
    and "odd" segments, and computes the bin-by-bin median of the "even"
    segments and the "odd" segments, and then takes the bin-by-bin average
    of these two median averages.

    Parameters
    ----------
    timeseries : `~gwpy.timeseries.TimeSeries`
        Input `TimeSeries` data.

    segmentlength : `int`
        Number of samples in single average.

    noverlap : `int`
        Number of samples to overlap between segments, defaults to 50%.

    window : `tuple`, `str`, optional
        Window parameters to apply to timeseries prior to FFT.

    plan : `REAL8FFTPlan`, optional
        LAL FFT plan to use when generating average spectrum.

    Returns
    -------
    spectrum : `~gwpy.frequencyseries.FrequencySeries`
        Average power `FrequencySeries`.

    See Also
    --------
    lal.REAL8AverageSpectrumMedianMean
    """
    return _lal_spectrum(
        timeseries,
        segmentlength,
        noverlap=noverlap,
        method="median-mean",
        window=window,
        plan=plan,
    )


# register LAL methods without overriding scipy method
for func in (welch, bartlett, median, median_mean):
    fft_registry.register_method(
        func,
        name=f"lal-{func.__name__}",
        deprecated=True,
    )
