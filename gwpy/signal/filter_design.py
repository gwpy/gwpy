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

"""Custom filtering utilities for the `TimeSeries`
"""

from __future__ import division
import operator
from math import (pi, log10)

from six.moves import reduce

from numpy import (atleast_1d, concatenate)

from scipy import signal

from astropy.units import Quantity

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
    nyq = sample_rate / 2.
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
    if analog:
        wp *= TWO_PI
        ws *= TWO_PI
    else:
        wp /= nyq
        ws /= nyq
    z, p, k = signal.iirdesign(wp, ws, gpass, gstop, analog=analog,
                               ftype=ftype, output='zpk')
    if analog:
        z /= -TWO_PI
        p /= -TWO_PI
    if output == 'zpk':
        return z, p, k
    elif output == 'ba':
        return signal.zpk2tf(z, p, k)
    elif output == 'sos':
        return signal.zpk2sos(z, p, k)
    else:
        raise ValueError("'%s' is not a valid output form." % output)


def _design_fir(wp, ws, sample_rate, gpass, gstop, window='hamming', **kwargs):
    wp = atleast_1d(wp)
    ws = atleast_1d(ws)
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
    return int(2/3. * log10(1 / (10 * gpass * gstop)) *
               sample_rate / transitionwidth)


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

    .. plot::
       :context: reset

       from gwpy.signal import lowpass
       zpk = lowpass(1000, 4096)

    To view the filter, you can use the `~gwpy.plotter.BodePlot`:

    .. plot::
       :context:

       from gwpy.plotter import BodePlot
       plot = BodePlot(zpk, sample_rate=4096)
       plot.show()

    """
    sample_rate = _as_float(sample_rate)
    frequency = _as_float(frequency)
    nyq = sample_rate / 2.
    if fstop is None:
        fstop = min(frequency * 1.5, sample_rate/2.)
    if type == 'iir':
        return _design_iir(frequency, fstop, sample_rate, gpass, gstop,
                           **kwargs)
    else:
        return _design_fir(frequency, fstop, sample_rate, gpass, gstop,
                           **kwargs)


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

    .. plot::
       :context: reset

       from gwpy.signal import highpass
       zpk = highpass(100, 4096)

    To view the filter, you can use the `~gwpy.plotter.BodePlot`:

    .. plot::
       :context:

       from gwpy.plotter import BodePlot
       plot = BodePlot(zpk, sample_rate=4096)
       plot.show()

    """
    sample_rate = _as_float(sample_rate)
    frequency = _as_float(frequency)
    nyq = sample_rate / 2.
    if fstop is None:
        fstop = frequency * 2/3.
    if type == 'iir':
        return _design_iir(frequency, fstop, sample_rate, gpass, gstop,
                           **kwargs)
    else:
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

    .. plot::
       :context: reset

       from gwpy.signal import bandpass
       zpk = bandpass(100, 1000, 4096)

    To view the filter, you can use the `~gwpy.plotter.BodePlot`:

    .. plot::
       :context:

       from gwpy.plotter import BodePlot
       plot = BodePlot(zpk, sample_rate=4096)
       plot.show()
    """
    sample_rate = _as_float(sample_rate)
    flow = _as_float(flow)
    fhigh = _as_float(fhigh)
    nyq = sample_rate / 2.
    if fstop is None:
        fstop = (flow * 2/3.,
                 min(fhigh * 1.5, sample_rate/2.))
    fstop = (_as_float(fstop[0]), _as_float(fstop[1]))
    if type == 'iir':
        return _design_iir((flow, fhigh), fstop, sample_rate, gpass, gstop,
                           **kwargs)
    else:
        return _design_fir((flow, fhigh), fstop, sample_rate, gpass, gstop,
                           pass_zero=False, **kwargs)


def notch(frequency, sample_rate, type='iir', **kwargs):
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
    **kwargs
        other keyword arguments to pass to `scipy.signal.iirdesign`

    Returns
    -------
    zpk : `tuple` of `complex` or `float`
       the filter components in digital zero-pole-gain format

    See Also
    --------
    scipy.signal.iirdesign
        for details on the IIR filter design method

    Notes
    -----
    By default a digital filter is returned, meaning the zeros and poles
    are given in the Z-domain in units of radians/sample.

    Examples
    --------
    To create a low-pass filter at 1000 Hz for 4096 Hz-sampled data:

    .. plot::
       :context: reset

       from gwpy.signal import notch
       n = notch(100, 4096)

    To view the filter, you can use the `~gwpy.plotter.BodePlot`:

    .. plot::
       :context:

       from gwpy.plotter import BodePlot
       plot = BodePlot(n, sample_rate=4096)
       plot.show()
    """
    frequency = Quantity(frequency, 'Hz').value
    sample_rate = Quantity(sample_rate, 'Hz').value
    nyq = 0.5 * sample_rate
    df = 1.0
    df2 = 0.1
    low1 = (frequency - df)/nyq
    high1 = (frequency + df)/nyq
    low2 = (frequency - df2)/nyq
    high2 = (frequency + df2)/nyq
    if type == 'iir':
        kwargs.setdefault('gpass', 1)
        kwargs.setdefault('gstop', 10)
        kwargs.setdefault('ftype', 'ellip')
        return signal.iirdesign([low1, high1], [low2, high2], output='zpk',
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
    .. plot::
       :context: reset

       from gwpy.signal import (highpass, lowpass, concatenate_zpks)
       hp = highpass(100, 4096)
       lp = lowpass(1000, 4096)
       zpk = concatenate_zpks(hp, lp)

       from gwpy.plotter import BodePlot
       plot = BodePlot(zpk, sample_rate=4096)
       plot.show()
    """
    zs, ps, ks = zip(*zpks)
    return concatenate(zs), concatenate(ps), reduce(operator.mul, ks, 1)
