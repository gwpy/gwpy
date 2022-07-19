# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
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

"""Custom filtering utilities for the `TimeSeries`
"""

import operator
from functools import reduce
from math import (pi, log10)

import numpy
from numpy import fft as npfft

from scipy import signal

from astropy.units import (Unit, Quantity)

from .window import planck

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = ['lowpass', 'highpass', 'bandpass', 'notch', 'concatenate_zpks']


def _as_float(x):
    try:
        return float(x.value)
    except AttributeError:
        return float(x)


TWO_PI = 2 * pi
FIRWIN_DEFAULTS = {
    'window': 'hann',
}


# -- core filter design utilities ---------------------------------------------

def _design_iir(wp, ws, sample_rate, gpass, gstop,
                analog=False, ftype='cheby1', output='zpk'):
    # pylint: disable=invalid-name
    nyq = sample_rate / 2.
    wp = numpy.atleast_1d(wp)
    ws = numpy.atleast_1d(ws)
    if analog:  # convert Hz to rad/s
        wp *= TWO_PI
        ws *= TWO_PI
    else:  # convert Hz to half-cycles / sample
        wp /= nyq
        ws /= nyq
    z, p, k = signal.iirdesign(wp, ws, gpass, gstop, analog=analog,
                               ftype=ftype, output='zpk')
    if analog:  # convert back to Hz
        z /= -TWO_PI
        p /= -TWO_PI
        k *= TWO_PI ** z.size / -TWO_PI ** p.size
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return signal.zpk2tf(z, p, k)
    elif output == 'sos':
        return signal.zpk2sos(z, p, k)
    else:
        raise ValueError("'%s' is not a valid output form." % output)


def _design_fir(wp, ws, sample_rate, gpass, gstop, window='hamming', **kwargs):
    # pylint: disable=invalid-name
    wp = numpy.atleast_1d(wp)
    ws = numpy.atleast_1d(ws)
    tw = abs(wp[0] - ws[0])
    nt = num_taps(sample_rate, tw, gpass, gstop)
    if wp[0] > ws[0]:
        kwargs.setdefault('pass_zero', False)
    if ws.shape == (1,):
        kwargs.setdefault('width', ws - wp)
    kwargs.setdefault('nyq', sample_rate/2.)
    return signal.firwin(nt, wp, window=window, **kwargs)


def num_taps(sample_rate, transitionwidth, gpass, gstop):
    """Returns the number of taps for an FIR filter with the given shape

    Parameters
    ----------
    sample_rate : `float`
        sampling rate of target data

    transitionwidth : `float`
        the width (in the same units as `sample_rate` of the transition
        from stop-band to pass-band

    gpass : `float`
        the maximum loss in the passband (dB)

    gstop : `float`
        the minimum attenuation in the stopband (dB)

    Returns
    -------
    numtaps : `int`
       the number of taps for an FIR filter

    Notes
    -----
    Credit: http://dsp.stackexchange.com/a/31077/8223
    """
    gpass = 10 ** (-gpass / 10.)
    gstop = 10 ** (-gstop / 10.)
    return int(
        2/3.
        * log10(1 / (10 * gpass * gstop))
        * sample_rate
        / transitionwidth
    )


def is_zpk(zpktup):
    """Determin whether the given tuple is a ZPK-format filter definition

    Returns
    -------
    iszpk : `bool`
        `True` if the ``zpktup`` looks like a ZPK-format filter definition,
        otherwise `False`
    """
    return (
        isinstance(zpktup, (tuple, list))
        and len(zpktup) == 3
        and isinstance(zpktup[0], (list, tuple, numpy.ndarray))
        and isinstance(zpktup[1], (list, tuple, numpy.ndarray))
        and isinstance(zpktup[2], float)
    )


def truncate_transfer(transfer, ncorner=None):
    """Smoothly zero the edges of a frequency domain transfer function

    Parameters
    ----------
    transfer : `numpy.ndarray`
        transfer function to start from, must have at least ten samples

    ncorner : `int`, optional
        number of extra samples to zero off at low frequency, default: `None`

    Returns
    -------
    out : `numpy.ndarray`
        the smoothly truncated transfer function

    Notes
    -----
    By default, the input transfer function will have five samples tapered
    off at the left and right boundaries. If `ncorner` is not `None`, then
    `ncorner` extra samples will be zeroed on the left as a hard highpass
    filter.

    See :func:`~gwpy.signal.window.planck` for more information.
    """
    nsamp = transfer.size
    ncorner = ncorner if ncorner else 0
    out = transfer.copy()
    out[0:ncorner] = 0
    out[ncorner:nsamp] *= planck(nsamp-ncorner, nleft=5, nright=5)
    return out


def truncate_impulse(impulse, ntaps, window='hann'):
    """Smoothly truncate a time domain impulse response

    Parameters
    ----------
    impulse : `numpy.ndarray`
        the impulse response to start from

    ntaps : `int`
        number of taps in the final filter

    window : `str`, `numpy.ndarray`, optional
        window function to truncate with, default: ``'hann'``
        see :func:`scipy.signal.get_window` for details on acceptable formats

    Returns
    -------
    out : `numpy.ndarray`
        the smoothly truncated impulse response
    """
    out = impulse.copy()
    trunc_start = int(ntaps / 2)
    trunc_stop = out.size - trunc_start
    window = signal.get_window(window, ntaps)
    out[0:trunc_start] *= window[trunc_start:ntaps]
    out[trunc_stop:out.size] *= window[0:trunc_start]
    out[trunc_start:trunc_stop] = 0
    return out


def fir_from_transfer(transfer, ntaps, window='hann', ncorner=None):
    """Design a Type II FIR filter given an arbitrary transfer function

    Parameters
    ----------
    transfer : `numpy.ndarray`
        transfer function to start from, must have at least ten samples

    ntaps : `int`
        number of taps in the final filter, must be an even number

    window : `str`, `numpy.ndarray`, optional
        window function to truncate with, default: ``'hann'``
        see :func:`scipy.signal.get_window` for details on acceptable formats

    ncorner : `int`, optional
        number of extra samples to zero off at low frequency, default: `None`

    Returns
    -------
    out : `numpy.ndarray`
        A time domain FIR filter of length `ntaps`

    Notes
    -----
    The final FIR filter will use `~numpy.fft.rfft` FFT normalisation.

    If `ncorner` is not `None`, then `ncorner` extra samples will be zeroed
    on the left as a hard highpass filter.

    See also
    --------
    scipy.signal.remez
        an alternative FIR filter design using the Remez exchange algorithm
    """
    # truncate and highpass the transfer function
    transfer = truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = npfft.irfft(transfer)
    impulse = truncate_impulse(impulse, ntaps=ntaps, window=window)
    # wrap around and normalise to construct the filter
    out = numpy.roll(impulse, int(ntaps/2 - 1))[0:ntaps]
    return out


def bilinear_zpk(zeros, poles, gain, fs=1.0, unit='Hz'):
    """Convert an analogue ZPK filter to digital using a bilinear transform

    Parameters
    ----------
    zeros : array-like
        list of zeros

    poles : array-like
        list of poles

    gain : `float`
        filter gain

    fs : `float`, `~astropy.units.Quantity`
        sampling rate at which to evaluate bilinear transform, default: 1.

    unit : `str`, `~astropy.units.Unit`
        unit of inputs, one or 'Hz' or 'rad/s', default: ``'Hz'``

    Returns
    -------
    zpk : `tuple`
        digital version of input zpk
    """
    zeros = numpy.array(zeros, dtype=float, copy=False)
    zeros = zeros[numpy.isfinite(zeros)]
    poles = numpy.array(poles, dtype=float, copy=False)
    gain = gain

    # convert from Hz to rad/s if needed
    unit = Unit(unit)
    if unit == Unit('Hz'):
        zeros *= -2 * pi
        poles *= -2 * pi
    elif unit != Unit('rad/s'):
        raise ValueError("zpk can only be given with unit='Hz' "
                         "or 'rad/s'")

    # convert to Z-domain via bilinear transform
    fs = 2 * Quantity(fs, 'Hz').value
    dpoles = (1 + poles/fs) / (1 - poles/fs)
    dzeros = (1 + zeros/fs) / (1 - zeros/fs)
    dzeros = numpy.concatenate((
        dzeros, -numpy.ones(len(dpoles) - len(dzeros)),
    ))
    dgain = gain * numpy.prod(fs - zeros)/numpy.prod(fs - poles)
    return dzeros, dpoles, dgain


def parse_filter(args, analog=False, sample_rate=None):
    """Parse arbitrary input args into a TF or ZPK filter definition

    Parameters
    ----------
    args : `tuple`, `~scipy.signal.lti`
        filter definition, normally just captured positional ``*args``
        from a function call

    analog : `bool`, optional
        `True` if filter definition has analogue coefficients

    sample_rate : `float`, optional
        sampling frequency at which to convert analogue filter to digital
        via bilinear transform, required if ``analog=True``

    Returns
    -------
    ftype : `str`
        either ``'ba'`` or ``'zpk'``
    filt : `tuple`
        the filter components for the returned `ftype`, either a 2-tuple
        for with transfer function components, or a 3-tuple for ZPK
    """
    if analog and not sample_rate:
        raise ValueError("Must give sample_rate frequency to convert "
                         "analog filter to digital")

    # unpack filter
    if isinstance(args, tuple) and len(args) == 1:
        # either packed defintion ((z, p, k)) or simple definition (lti,)
        args = args[0]

    # parse FIR filter
    if isinstance(args, numpy.ndarray) and args.ndim == 1:  # fir
        b, a = args, [1.]
        if analog:
            return 'ba', signal.bilinear(b, a)
        return 'ba', (b, a)

    # parse IIR filter
    try:
        lti = args.to_zpk()
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

    # convert to digital components
    if analog:
        return 'zpk', bilinear_zpk(lti.zeros, lti.poles, lti.gain,
                                   fs=sample_rate)
    # return zpk
    return 'zpk', (lti.zeros, lti.poles, lti.gain)


# -- user methods -------------------------------------------------------------

def lowpass(frequency, sample_rate, fstop=None, gpass=2, gstop=30, type='iir',
            **kwargs):
    """Design a low-pass filter for the given cutoff frequency

    Parameters
    ----------
    frequency : `float`
        corner frequency of low-pass filter (Hertz)

    sample_rate : `float`
        sampling rate of target data (Hertz)

    fstop : `float`, optional
        edge-frequency of stop-band (Hertz)

    gpass : `float`, optional, default: 2
        the maximum loss in the passband (dB)

    gstop : `float`, optional, default: 30
        the minimum attenuation in the stopband (dB)

    type : `str`, optional, default: ``'iir'``
        the filter type, either ``'iir'`` or ``'fir'``

    **kwargs
        other keyword arguments are passed directly to
        :func:`~scipy.signal.iirdesign` or :func:`~scipy.signal.firwin`

    Returns
    -------
    filter
        the formatted filter. the output format for an IIR filter depends
        on the input arguments, default is a tuple of `(zeros, poles, gain)`

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
        fstop = min(frequency * 1.5, sample_rate/2.)
    if type == 'iir':
        return _design_iir(frequency, fstop, sample_rate, gpass, gstop,
                           **kwargs)
    return _design_fir(frequency, fstop, sample_rate, gpass, gstop, **kwargs)


def highpass(frequency, sample_rate, fstop=None, gpass=2, gstop=30, type='iir',
             **kwargs):
    """Design a high-pass filter for the given cutoff frequency

    Parameters
    ----------
    frequency : `float`
        corner frequency of high-pass filter

    sample_rate : `float`
        sampling rate of target data

    fstop : `float`, optional
        edge-frequency of stop-band

    gpass : `float`, optional, default: 2
        the maximum loss in the passband (dB)

    gstop : `float`, optional, default: 30
        the minimum attenuation in the stopband (dB)

    type : `str`, optional, default: ``'iir'``
        the filter type, either ``'iir'`` or ``'fir'``

    **kwargs
        other keyword arguments are passed directly to
        :func:`~scipy.signal.iirdesign` or :func:`~scipy.signal.firwin`

    Returns
    -------
    filter
        the formatted filter. the output format for an IIR filter depends
        on the input arguments, default is a tuple of `(zeros, poles, gain)`

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
        fstop = frequency * 2/3.
    if type == 'iir':
        return _design_iir(frequency, fstop, sample_rate, gpass, gstop,
                           **kwargs)
    return _design_fir(frequency, fstop, sample_rate, gpass, gstop,
                       **kwargs)


def bandpass(flow, fhigh, sample_rate, fstop=None, gpass=2, gstop=30,
             type='iir', **kwargs):
    """Design a band-pass filter for the given cutoff frequencies

    Parameters
    ----------
    flow : `float`
        lower corner frequency of pass band

    fhigh : `float`
        upper corner frequency of pass band

    sample_rate : `float`
        sampling rate of target data

    fstop : `tuple` of `float`, optional
        `(low, high)` edge-frequencies of stop band

    gpass : `float`, optional, default: 2
        the maximum loss in the passband (dB)

    gstop : `float`, optional, default: 30
        the minimum attenuation in the stopband (dB)

    type : `str`, optional, default: ``'iir'``
        the filter type, either ``'iir'`` or ``'fir'``

    **kwargs
        other keyword arguments are passed directly to
        :func:`~scipy.signal.iirdesign` or :func:`~scipy.signal.firwin`

    Returns
    -------
    filter
        the formatted filter. the output format for an IIR filter depends
        on the input arguments, default is a tuple of `(zeros, poles, gain)`

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
        fstop = (flow * 2/3.,
                 min(fhigh * 1.5, sample_rate/2.))
    fstop = (_as_float(fstop[0]), _as_float(fstop[1]))
    if type == 'iir':
        return _design_iir((flow, fhigh), fstop, sample_rate, gpass, gstop,
                           **kwargs)
    return _design_fir((flow, fhigh), fstop, sample_rate, gpass, gstop,
                       pass_zero=False, **kwargs)


def notch(frequency, sample_rate, type='iir', output='zpk', **kwargs):
    """Design a ZPK notch filter for the given frequency and sampling rate

    Parameters
    ----------
    frequency : `float`, `~astropy.units.Quantity`
        frequency (default in Hertz) at which to apply the notch
    sample_rate : `float`, `~astropy.units.Quantity`
        number of samples per second for `TimeSeries` to which this notch
        filter will be applied
    type : `str`, optional, default: 'iir'
        type of filter to apply, currently only 'iir' is supported
    output : `str`, optional, default: 'zpk'
        output format for notch
    **kwargs
        other keyword arguments to pass to `scipy.signal.iirdesign`

    Returns
    -------
    filter
        the formatted filter; the output format for an IIR filter depends
        on the input arguments, default is a tuple of `(zeros, poles, gain)`

    See also
    --------
    scipy.signal.iirdesign
        for details on the IIR filter design method and the output formats

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
    frequency = Quantity(frequency, 'Hz').value
    sample_rate = Quantity(sample_rate, 'Hz').value
    nyq = 0.5 * sample_rate
    df = 1.0  # pylint: disable=invalid-name
    df2 = 0.1
    low1 = (frequency - df)/nyq
    high1 = (frequency + df)/nyq
    low2 = (frequency - df2)/nyq
    high2 = (frequency + df2)/nyq
    if type == 'iir':
        kwargs.setdefault('gpass', 1)
        kwargs.setdefault('gstop', 10)
        kwargs.setdefault('ftype', 'ellip')
        return signal.iirdesign([low1, high1], [low2, high2], output=output,
                                **kwargs)
    else:
        raise NotImplementedError("Generating %r notch filters has not been "
                                  "implemented yet" % type)


def concatenate_zpks(*zpks):
    """Concatenate a list of zero-pole-gain (ZPK) filters

    Parameters
    ----------
    *zpks
        one or more zero-pole-gain format, each one should be a 3-`tuple`
        containing an array of zeros, an array of poles, and a gain `float`

    Returns
    -------
    zeros : `numpy.ndarray`
        the concatenated array of zeros
    poles : `numpy.ndarray`
        the concatenated array of poles
    gain : `float`
        the overall gain

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
    zeros, poles, gains = zip(*zpks)
    return (numpy.concatenate(zeros),
            numpy.concatenate(poles),
            reduce(operator.mul, gains, 1))
