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

"""Custom filtering utilities for the `TimeSeries`."""

from __future__ import annotations

import operator
from functools import reduce
from math import (
    log10,
    pi,
)
from typing import (
    TYPE_CHECKING,
    overload,
)

import numpy
from astropy.units import Quantity
from numpy import fft as npfft
from scipy import signal

from .window import (
    get_window,
    planck,
)

# filter type definitions
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        Any,
        Literal,
        SupportsFloat,
        TypeAlias,
    )

    from astropy.units.typing import QuantityLike
    from numpy.typing import (
        ArrayLike,
        NDArray,
    )
    from scipy.signal import lti

    from .window import WindowLike

    # FIR
    TapsType: TypeAlias = NDArray
    # IIR
    FilterTypeName: TypeAlias = Literal["butter", "cheby1", "cheby2", "ellip"]
    SosType: TypeAlias = NDArray
    ZpkType: TypeAlias = tuple[NDArray, NDArray, float]
    BAType: TypeAlias = tuple[NDArray, NDArray]
    IirFilterType: TypeAlias = SosType | ZpkType | BAType
    # generic
    FilterType = TapsType | IirFilterType

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "bandpass",
    "concatenate_zpks",
    "highpass",
    "lowpass",
    "notch",
]


def _as_float(
    x: QuantityLike,
    unit: str = "Hz",
) -> float:
    """Convert input to a float in the given units."""
    return Quantity(x, unit).value


TWO_PI: float = 2 * pi


# -- core filter design utilities ----

@overload
def _design_iir(
    wp: ArrayLike,
    ws: ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    *,
    analog: bool,
    ftype: FilterTypeName,
    output: Literal["zpk"],
) -> ZpkType: ...

@overload
def _design_iir(
    wp: ArrayLike,
    ws: ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    *,
    analog: bool,
    ftype: FilterTypeName,
    output: Literal["ba"],
) -> BAType: ...

@overload
def _design_iir(
    wp: ArrayLike,
    ws: ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    *,
    analog: bool,
    ftype: FilterTypeName,
    output: Literal["sos"],
) -> SosType: ...

def _design_iir(
    wp: ArrayLike,
    ws: ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    *,
    analog: bool = False,
    ftype: FilterTypeName = "cheby1",
    output: Literal["zpk", "ba", "sos"] = "zpk",
) -> IirFilterType:
    """Design an IIR filter using `scipy.signal.iirdesign`."""
    nyq = sample_rate / 2.
    wp = numpy.atleast_1d(wp)
    ws = numpy.atleast_1d(ws)
    if analog:  # convert Hz to rad/s
        wp *= TWO_PI
        ws *= TWO_PI
    else:  # convert Hz to half-cycles / sample
        wp /= nyq
        ws /= nyq
    z, p, k = signal.iirdesign(
        wp,
        ws,
        gpass,
        gstop,
        analog=analog,
        ftype=ftype,
        output="zpk",
    )
    if analog:  # convert back to Hz
        z /= -TWO_PI
        p /= -TWO_PI
        k *= TWO_PI ** z.size / -TWO_PI ** p.size
    if output == "zpk":
        return z, p, k
    if output == "ba":
        return signal.zpk2tf(z, p, k)
    if output == "sos":
        return signal.zpk2sos(z, p, k)
    msg = f"'{output}' is not a valid output form"
    raise ValueError(msg)


def _design_fir(
    wp: float | ArrayLike,
    ws: float | ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    window: str = "hamming",
    **kwargs,
) -> TapsType:
    """Design an FIR filter using `scipy.signal.firwin`.

    This is just an internal convenience function to calculate the number of
    taps based on the pass and stop band frequencies, and to set sensible
    defaults for a few other keyword arguments.

    See Also
    --------
    scipy.signal.firwin
        for details of how the FIR filters are actually generated
    """
    # format arguments
    wp = numpy.atleast_1d(wp)
    ws = numpy.atleast_1d(ws)
    tw = wp[0] - ws[0]

    # calculate the number of taps
    nt = num_taps(sample_rate, tw, gpass, gstop)

    # set default kw based on the filter shape
    if wp[0] > ws[0]:  # highpass
        kwargs.setdefault("pass_zero", False)
    if ws.shape == (1,):  # simple list of taps
        kwargs.setdefault("width", ws.item() - wp.item())

    kwargs.setdefault("fs", sample_rate)
    return signal.firwin(nt, wp, window=window, **kwargs)


def _design(
    type: str,
    *args,
    **kwargs,
) -> FilterType:
    """Convenience function to select between `_design_iir` and `_design_fir`."""
    design_func: Callable
    if type == "iir":
        design_func = _design_iir
    else:
        design_func = _design_fir
    return design_func(*args, **kwargs)


def num_taps(
    sample_rate: float,
    transitionwidth: float,
    gpass: float,
    gstop: float,
) -> int:
    """Returns the number of taps for an FIR filter with the given shape.

    Parameters
    ----------
    sample_rate : `float`
        Sampling rate of target data.

    transitionwidth : `float`
        The width (in the same units as `sample_rate`) of the transition
        from stop-band to pass-band.

    gpass : `float`
        The maximum loss in the passband (dB).

    gstop : `float`
        The minimum attenuation in the stopband (dB).

    Returns
    -------
    numtaps : `int`
       The number of taps for an FIR filter.

    Notes
    -----
    Credit: http://dsp.stackexchange.com/a/31077/8223
    """
    gpass = 10 ** (-gpass / 10.)
    gstop = 10 ** (-gstop / 10.)
    ntaps = int(
        2 / 3.
        * log10(1 / (10 * gpass * gstop))
        * sample_rate
        / abs(transitionwidth),
    )
    # highpass filters must have an odd number of taps
    if transitionwidth > 0 and ntaps % 2 == 0:
        return ntaps + 1
    return ntaps


def is_zpk(zpktup: Any) -> bool:
    """Return `True` if ``zpktup`` looks like a ZPK-format filter definition.

    Returns
    -------
    iszpk : `bool`
        `True` if input argument looks like a 3-tuple giving arrays of
        zeros and poles, and a gain (`float`).
    """
    return (
        isinstance(zpktup, tuple | list)
        and len(zpktup) == 3
        and isinstance(zpktup[0], list | tuple | numpy.ndarray)
        and isinstance(zpktup[1], list | tuple | numpy.ndarray)
        and isinstance(zpktup[2], float)
    )


def truncate_transfer(
    transfer: numpy.ndarray,
    ncorner: int | None = None,
) -> numpy.ndarray:
    """Smoothly zero the edges of a frequency domain transfer function.

    Parameters
    ----------
    transfer : `numpy.ndarray`
        Transfer function to start from, must have at least ten samples.

    ncorner : `int`, optional
        Number of extra samples to zero off at low frequency.

    Returns
    -------
    out : `numpy.ndarray`
        The smoothly-truncated transfer function.

    Notes
    -----
    By default, the input transfer function will have five samples tapered
    off at the left and right boundaries. If `ncorner` is not `None`, then
    `ncorner` extra samples will be zeroed on the left as a hard highpass
    filter.

    See Also
    --------
    gwpy.signal.window.planck
    """
    nsamp = transfer.size
    ncorner = ncorner if ncorner else 0
    out = transfer.copy()
    out[0:ncorner] = 0
    out[ncorner:nsamp] *= planck(nsamp - ncorner, nleft=5, nright=5)
    return out


def truncate_impulse(
    impulse: numpy.ndarray,
    ntaps: int,
    window: WindowLike = "hann",
):
    """Smoothly truncate a time domain impulse response.

    Parameters
    ----------
    impulse : `numpy.ndarray`
        The impulse response to start from.

    ntaps : `int`
        Number of taps in the final filter.

    window : `str`, `float`, `tuple`, `numpy.ndarray`
        Window to truncate with, see `scipy.signal.get_window`
        for details on acceptable formats.

    Returns
    -------
    out : `numpy.ndarray`
        The smoothly truncated impulse response.
    """
    out = impulse.copy()
    trunc_start = int(ntaps / 2)
    trunc_stop = out.size - trunc_start
    window = get_window(window, ntaps)
    out[0:trunc_start] *= window[trunc_start:ntaps]
    out[trunc_stop:out.size] *= window[0:trunc_start]
    out[trunc_start:trunc_stop] = 0
    return out


def fir_from_transfer(
    transfer: numpy.ndarray,
    ntaps: int,
    window: WindowLike = "hann",
    ncorner: int | None = None,
):
    """Design a Type II FIR filter given an arbitrary transfer function.

    Parameters
    ----------
    transfer : `numpy.ndarray`
        Transfer function to start from, must have at least ten samples.

    ntaps : `int`
        Number of taps in the final filter, must be an even number.

    window : `str`, `numpy.ndarray`, optional
        Window function to truncate with, see `scipy.signal.get_window`
        for details on acceptable formats.

    ncorner : `int`, optional
        Number of extra samples to zero off at low frequency.

    Returns
    -------
    out : `numpy.ndarray`
        A time domain FIR filter of length `ntaps`.

    Notes
    -----
    The final FIR filter will use `~numpy.fft.rfft` FFT normalisation.

    If `ncorner` is not `None`, then `ncorner` extra samples will be zeroed
    on the left as a hard highpass filter.

    See Also
    --------
    scipy.signal.remez
        An alternative FIR filter design using the Remez exchange algorithm.
    """
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = npfft.irfft(transfer)
    impulse = truncate_impulse(
        impulse,
        ntaps=ntaps,
        window=window,
    )
    # wrap around and normalise to construct the filter
    return numpy.roll(impulse, int(ntaps / 2 - 1))[0:ntaps]


def convert_zpk_units(
    filt: ZpkType,
    unit: str,
):
    """Convert zeros and poles created for a freq response in Hz to rad/s.

    Parameters
    ----------
    filt : `tuple`
        Zeros, poles, gain.

    unit : `str`
        ``'Hz'`` or ``'rad/s'``.

    Returns
    -------
    zeros : `numpy.array` of `numpy.cfloat`
    poles : `numpy.array` of `numpy.cfloat`
    gain : input, unadjusted gain
    """
    zeros, poles, gain = filt

    if unit == "Hz":
        for zi in range(len(zeros)):
            zeros[zi] *= -2. * numpy.pi
        for pj in range(len(poles)):
            poles[pj] *= -2. * numpy.pi
    elif unit not in [
        "rad/s",
        "rad/sample",
    ]:
        msg = (
            "zpk can only be given with unit='Hz', 'rad/s', or 'rad/sample', "
            f"not '{unit}'"
        )
        raise ValueError(msg)

    return zeros, poles, gain


@overload
def convert_to_digital(
    args: TapsType | BAType | tuple[TapsType | BAType],
) -> tuple[Literal["ba"], BAType]: ...

@overload
def convert_to_digital(
    args: SosType | ZpkType | lti | tuple[SosType | ZpkType | lti],
) -> tuple[Literal["zpk"], ZpkType]: ...

def convert_to_digital(
    filter: FilterType | lti | tuple[FilterType | lti],
    sample_rate: float,
) -> tuple[Literal["ba", "zpk"], IirFilterType]:
    """Convert an analog filter to digital via bilinear functions.

    Parameters
    ----------
    filter: `tuple`
        Input filter to convert.

    sample_rate: `float`
        Sample rate of digital data that will be filtered.

    Returns
    -------
    dform : `str`
        Type of filter.

    dfilter : 'tuple'
        Digital filter values.
    """
    # This will always end up returning zpk form.
    # If FIR, bilinear will convert it to IIR.
    # If IIR, only if p_i = -2 * fs will it yield poles at zero.
    # See gwpy/signal/tests/test_filter_design for more information.

    form, filter = parse_filter(filter)

    if form == "ba":
        b, a = filter
        return form, signal.bilinear(b, a, fs=sample_rate)

    if form == "zpk":
        return form, signal.bilinear_zpk(*filter, fs=sample_rate)

    msg = f"cannot convert '{form}', only 'zpk' or 'ba'"
    raise ValueError(msg)


@overload
def parse_filter(
    args: TapsType | BAType | tuple[TapsType | BAType],
) -> tuple[Literal["ba"], BAType]: ...

@overload
def parse_filter(
    args: SosType | ZpkType | lti | tuple[SosType | ZpkType | lti],
) -> tuple[Literal["zpk"], ZpkType]: ...

def parse_filter(
    args: FilterType | lti | tuple[FilterType | lti],
) -> tuple[Literal["ba", "zpk"], BAType | ZpkType]:
    """Parse arbitrary input args into a TF or ZPK filter definition.

    Parameters
    ----------
    args : `tuple`, `~scipy.signal.lti`
        Filter definition, normally just captured positional ``*args``
        from a function call.

    Returns
    -------
    ftype : `str`
        Either ``'ba'`` or ``'zpk'``.

    filt : `tuple`
        The filter components for the returned `ftype`, either a 2-tuple
        for with transfer function components, or a 3-tuple for ZPK.
    """
    # unpack filter
    if isinstance(args, tuple) and len(args) == 1:
        # either packed defintion ((z, p, k)) or simple definition (lti,)
        args = args[0]

    # parse FIR filter
    if isinstance(args, numpy.ndarray) and args.ndim == 1:  # fir
        b, a = args, numpy.ones(1)
        return "ba", (b, a)

    # parse IIR filter
    try:
        lti = args.to_zpk()  # type: ignore[union-attr]
    except AttributeError:
        if (
            isinstance(args, numpy.ndarray)
            and args.ndim == 2
            and args.shape[1] == 6
        ):
            lti = signal.lti(*signal.sos2zpk(args))
        else:
            lti = signal.lti(*args)
        lti = lti.to_zpk()

    return "zpk", (lti.zeros, lti.poles, lti.gain)


# -- user methods --------------------

def lowpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: QuantityLike | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: str = "iir",
    **kwargs,
) -> FilterType:
    """Design a low-pass filter for the given cutoff frequency.

    Parameters
    ----------
    frequency : `float`
        Corner frequency of low-pass filter (Hertz).

    sample_rate : `float`
        Sampling rate of target data (Hertz).

    fstop : `float`, optional
        Edge-frequency of stop-band (Hertz).

    gpass : `float`, optional, default: 2
        The maximum loss in the passband (dB).

    gstop : `float`, optional, default: 30
        The minimum attenuation in the stopband (dB).

    type : `str`, optional, default: ``'iir'``
        The filter type, either ``'iir'`` or ``'fir'``.

    kwargs
        Other keyword arguments are passed directly to
        :func:`~scipy.signal.iirdesign` or :func:`~scipy.signal.firwin`.

    Returns
    -------
    filter
        The formatted filter.
        The output format for an IIR filter depends on the input arguments,
        default is a tuple of ``(zeros, poles, gain)``.

    Notes
    -----
    By default a digital filter is returned, meaning the zeros and poles
    are given in the Z-domain in units of radians/sample.

    Examples
    --------
    To create a low-pass filter at 1000 Hz for 4096 Hz-sampled data:

    >>> from gwpy.signal.filter_design import lowpass
    >>> lp = lowpass(1000, 4096)

    To view the filter, you can use the `~gwpy.plot.BodePlot`:

    >>> from gwpy.plot import BodePlot
    >>> plot = BodePlot(lp, sample_rate=4096)
    >>> plot.show()
    """
    sample_rate = _as_float(sample_rate)
    frequency = _as_float(frequency)
    if fstop is None:
        fstop = min(frequency * 1.5, sample_rate / 2.)
    fstop = _as_float(fstop)
    return _design(
        type,
        frequency,
        fstop,
        sample_rate,
        gpass,
        gstop,
        **kwargs,
    )


def highpass(
    frequency: SupportsFloat,
    sample_rate: SupportsFloat,
    fstop: float | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: str = "iir",
    **kwargs,
) -> FilterType:
    """Design a high-pass filter for the given cutoff frequency.

    Parameters
    ----------
    frequency : `float`, `~astropy.units.Quantity`
        Corner frequency of high-pass filter.

    sample_rate : `float`, `~astropy.units.Quantity`
        Sampling rate of target data.

    fstop : `float`
        Edge-frequency of stop-band.

    gpass : `float`
        The maximum loss in the passband (dB)

    gstop : `float`
        The minimum attenuation in the stopband (dB).

    type : `str`
        The filter type, either ``'iir'`` or ``'fir'``.

    kwargs
        Other keyword arguments are passed directly to
        :func:`~scipy.signal.iirdesign` or :func:`~scipy.signal.firwin`.

    Returns
    -------
    filter
        The formatted filter.
        The output format for an IIR filter depends on the input arguments,
        default is a tuple of ``(zeros, poles, gain)``.

    Notes
    -----
    By default a digital filter is returned, meaning the zeros and poles
    are given in the Z-domain in units of radians/sample.

    Examples
    --------
    To create a high-pass filter at 100 Hz for 4096 Hz-sampled data:

    >>> from gwpy.signal.filter_design import highpass
    >>> hp = highpass(100, 4096)

    To view the filter, you can use the `~gwpy.plot.BodePlot`:

    >>> from gwpy.plot import BodePlot
    >>> plot = BodePlot(hp, sample_rate=4096)
    >>> plot.show()
    """
    sample_rate = _as_float(sample_rate)
    frequency = _as_float(frequency)
    if fstop is None:
        fstop = frequency * 2 / 3.
    return _design(
        type,
        frequency,
        fstop,
        sample_rate,
        gpass,
        gstop,
        **kwargs,
    )


def bandpass(
    flow: SupportsFloat,
    fhigh: SupportsFloat,
    sample_rate: SupportsFloat,
    fstop: tuple[float, float] | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: str = "iir",
    **kwargs,
) -> FilterType:
    """Design a band-pass filter for the given cutoff frequencies.

    Parameters
    ----------
    flow : `float`, `~astropy.units.Quantity`
        Lower corner frequency of pass band.

    fhigh : `float`, `~astropy.units.Quantity`
        Upper corner frequency of pass band.

    sample_rate : `float`, `~astropy.units.Quantity`
        Sampling rate of target data.

    fstop : `tuple` of `float`
        `(low, high)` edge-frequencies of stop band.

    gpass : `float`
        The maximum loss in the passband (dB).

    gstop : `float`
        The minimum attenuation in the stopband (dB).

    type : `str`
        The filter type, either ``'iir'`` or ``'fir'``.

    kwargs
        Other keyword arguments are passed directly to
        :func:`~scipy.signal.iirdesign` or :func:`~scipy.signal.firwin`.

    Returns
    -------
    filter
        The formatted filter.
        The output format for an IIR filter depends on the input arguments,
        default is a tuple of `(zeros, poles, gain)`.

    Notes
    -----
    By default a digital filter is returned, meaning the zeros and poles
    are given in the Z-domain in units of radians/sample.

    Examples
    --------
    To create a band-pass filter for 100-1000 Hz for 4096 Hz-sampled data:

    >>> from gwpy.signal.filter_design import bandpass
    >>> bp = bandpass(100, 1000, 4096)

    To view the filter, you can use the `~gwpy.plot.BodePlot`:

    >>> from gwpy.plot import BodePlot
    >>> plot = BodePlot(bp, sample_rate=4096)
    >>> plot.show()
    """
    sample_rate = _as_float(sample_rate)
    flow = _as_float(flow)
    fhigh = _as_float(fhigh)
    if fstop is None:
        fstop = (
            flow * 2 / 3.,
            min(fhigh * 1.5, sample_rate / 2.),
        )
    fstop = (_as_float(fstop[0]), _as_float(fstop[1]))
    if type == "fir":
        kwargs.setdefault("pass_zero", False)
    return _design(
        type,
        (flow, fhigh),
        fstop,
        sample_rate,
        gpass,
        gstop,
        **kwargs,
    )


def notch(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    type: Literal["iir"] = "iir",
    output: Literal["zpk", "ba", "sos"] = "zpk",
    **kwargs,
) -> IirFilterType:
    """Design a ZPK notch filter for the given frequency and sampling rate.

    Parameters
    ----------
    frequency : `float`, `~astropy.units.Quantity`
        Frequency (default in Hertz) at which to apply the notch.

    sample_rate : `float`, `~astropy.units.Quantity`
        Number of samples per second for `TimeSeries` to which this notch
        filter will be applied.

    type : `str`, optional, default: 'iir'
        Type of filter to apply, currently only 'iir' is supported.

    output : `str`, optional, default: 'zpk'
        Output format for notch.

    kwargs
        Other keyword arguments to pass to `scipy.signal.iirdesign`.

    Returns
    -------
    filter
        The formatted filter.
        The output format for an IIR filter depends on the input arguments,
        default is a tuple of ``(zeros, poles, gain)``.

    See Also
    --------
    scipy.signal.iirdesign
        For details on the IIR filter design method and the output formats.

    Notes
    -----
    By default a digital filter is returned, meaning the zeros and poles
    are given in the Z-domain in units of radians/sample.

    Examples
    --------
    To create a low-pass filter at 1000 Hz for 4096 Hz-sampled data:

    >>> from gwpy.signal.filter_design import notch
    >>> n = notch(100, 4096)

    To view the filter, you can use the `~gwpy.plot.BodePlot`:

    >>> from gwpy.plot import BodePlot
    >>> plot = BodePlot(n, sample_rate=4096)
    >>> plot.show()
    """
    frequency = _as_float(frequency)
    sample_rate = _as_float(sample_rate)
    df = 1.0
    df2 = 0.1
    low1 = (frequency - df)
    high1 = (frequency + df)
    low2 = (frequency - df2)
    high2 = (frequency + df2)
    if type == "iir":
        kwargs.setdefault("gpass", 1)
        kwargs.setdefault("gstop", 10)
        kwargs.setdefault("ftype", "ellip")
        return _design_iir(
            [low1, high1],
            [low2, high2],
            sample_rate,
            output=output,
            **kwargs,
        )
    msg = f"Generating {type} notch filters has not been implemented yet"
    raise NotImplementedError(msg)


def concatenate_zpks(*zpks: ZpkType) -> ZpkType:
    """Concatenate a list of zero-pole-gain (ZPK) filters.

    Parameters
    ----------
    *zpks
        One or more zero-pole-gain format, each one should be a 3-`tuple`
        containing an array of zeros, an array of poles, and a gain `float`.

    Returns
    -------
    zeros : `numpy.ndarray`
        The concatenated array of zeros.
    poles : `numpy.ndarray`
        The concatenated array of poles.
    gain : `float`
        The overall gain.

    Examples
    --------
    Create a lowpass and a highpass filter, and combine them:

    >>> from gwpy.signal.filter_design import (
    ...     highpass, lowpass, concatenate_zpks)
    >>> hp = highpass(100, 4096)
    >>> lp = lowpass(1000, 4096)
    >>> zpk = concatenate_zpks(hp, lp)

    Plot the filter:

    >>> from gwpy.plot import BodePlot
    >>> plot = BodePlot(zpk, sample_rate=4096)
    >>> plot.show()
    """
    zeros, poles, gains = zip(*zpks, strict=True)
    return (
        numpy.concatenate(zeros),
        numpy.concatenate(poles),
        reduce(operator.mul, gains, 1),
    )
