# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2019)
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

"""This module provides methods to calculate the astrophysical sensitive
distance of an instrumental power-spectral density.
"""

import warnings
from functools import wraps
from math import pi

from scipy import integrate

from astropy import (units, constants)

from ..timeseries import TimeSeries
from ..spectrogram import Spectrogram

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'
__credits__ = 'Alex Urban <alexander.urban@ligo.org>'

DEFAULT_FFT_METHOD = "welch"


def _get_spectrogram(hoft, **kwargs):
    """Check that the input is a spectrogram, or compute one if compatible

    Parameters
    ----------
    hoft : `~gwpy.timeseries.TimeSeries` or `~gwpy.spectrogram.Spectrogram`
        record of gravitational-wave strain output from a detector

    **kwargs : `dict`, optional
        additional keyword arguments to
        `~gwpy.timeseries.TimeSeries.spectrogram`

    Returns
    -------
    hoft : `~gwpy.spectrogram.Spectrogram`
        a time-frequency `Spectrogram` of the input
    """
    if not isinstance(hoft, Spectrogram):
        try:
            hoft = hoft.spectrogram(**kwargs)
        except (AttributeError, TypeError):
            msg = ('Could not produce a spectrogram from the input, please '
                   'pass an instance of gwpy.timeseries.TimeSeries or '
                   'gwpy.spectrogram.Spectrogram')
            raise TypeError(msg)
    return hoft


def _preformat_psd(func):
    @wraps(func)
    def decorated_func(psd, *args, **kwargs):
        if psd.unit != 1/units.Hz:  # force PSD to have the right units
            psd = psd.view()
            psd.override_unit('1/Hz')
        return func(psd, *args, **kwargs)
    return decorated_func


@_preformat_psd
def inspiral_range_psd(psd, snr=8, mass1=1.4, mass2=1.4, horizon=False):
    """Compute the inspiral sensitive distance PSD from a GW strain PSD

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data

    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range,
        default: `8`

    mass1 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the first binary
        component, default: `1.4`

    mass2 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the second binary
        component, default: `1.4`

    horizon : `bool`, optional
        if `True`, return the maximal 'horizon' sensitive distance, otherwise
        return the angle-averaged range, default: `False`

    Returns
    -------
    rspec : `~gwpy.frequencyseries.FrequencySeries`
        the calculated inspiral sensitivity PSD [Mpc^2 / Hz]
    """
    # compute chirp mass and symmetric mass ratio
    mass1 = units.Quantity(mass1, 'solMass').to('kg')
    mass2 = units.Quantity(mass2, 'solMass').to('kg')
    mtotal = mass1 + mass2
    mchirp = (mass1 * mass2) ** (3/5.) / mtotal ** (1/5.)

    # compute ISCO
    fisco = (constants.c ** 3 / (constants.G * 6**1.5 * pi * mtotal)).to('Hz')

    # calculate integral pre-factor
    prefactor = (
        (1.77**2 * 5 * constants.c ** (1/3.) *
         (mchirp * constants.G / constants.c ** 2) ** (5/3.)) /
        (96 * pi ** (4/3.) * snr ** 2)
    )

    # calculate inspiral range ASD in m^2/Hz
    integrand = 1 / psd * psd.frequencies ** (-7/3.) * prefactor

    # restrict to ISCO
    integrand = integrand[psd.frequencies.value < fisco.value]

    # normalize and return
    if integrand.f0.value == 0.0:
        integrand[0] = 0.0
    if horizon:
        integrand *= 2.26 ** 2
    return integrand.to('Mpc^2 / Hz')


def inspiral_range(psd, snr=8, mass1=1.4, mass2=1.4, fmin=None, fmax=None,
                   horizon=False):
    """Calculate the inspiral sensitive distance from a GW strain PSD

    The method returns the distance (in megaparsecs) to which an compact
    binary inspiral with the given component masses would be detectable
    given the instrumental PSD. The calculation is as defined in:

    https://dcc.ligo.org/LIGO-T030276/public

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data

    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range,
        default: `8`

    mass1 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the first binary
        component, default: `1.4`

    mass2 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the second binary
        component, default: `1.4`

    fmin : `float`, optional
        the lower frequency cut-off of the integral, default: `psd.df`

    fmax : `float`, optional
        the maximum frequency limit of the integral, defaults to
        innermost stable circular orbit (ISCO) frequency

    horizon : `bool`, optional
        if `True`, return the maximal 'horizon' sensitive distance, otherwise
        return the angle-averaged range, default: `False`

    Returns
    -------
    range : `~astropy.units.Quantity`
        the calculated inspiral range [Mpc]

    Examples
    --------
    Grab some data for LIGO-Livingston around GW150914 and generate a PSD

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
    >>> hoff = hoft.psd(fftlength=4)

    Now we can calculate the :func:`inspiral_range`:

    >>> from gwpy.astro import inspiral_range
    >>> r = inspiral_range(hoff, fmin=30)
    >>> print(r)
    70.4612102889 Mpc
    """
    mass1 = units.Quantity(mass1, 'solMass').to('kg')
    mass2 = units.Quantity(mass2, 'solMass').to('kg')
    mtotal = mass1 + mass2

    # compute ISCO
    fisco = (constants.c ** 3 / (constants.G * 6**1.5 * pi * mtotal)).to('Hz')

    # format frequency limits
    fmax = units.Quantity(fmax or fisco, 'Hz')
    if fmax > fisco:
        warnings.warn("Upper frequency bound greater than %s-%s ISCO "
                      "frequency of %s, using ISCO" % (mass1, mass2, fisco))
        fmax = fisco
    if fmin is None:
        fmin = psd.df  # avoid using 0 as lower limit
    fmin = units.Quantity(fmin, 'Hz')

    # integrate
    f = psd.frequencies.to('Hz')
    condition = (f >= fmin) & (f < fmax)
    integrand = inspiral_range_psd(psd[condition], snr=snr, mass1=mass1,
                                   mass2=mass2, horizon=horizon)
    result = units.Quantity(
        integrate.trapz(integrand.value, f.value[condition]),
        unit=integrand.unit * units.Hertz)

    return (result ** (1/2.)).to('Mpc')


@_preformat_psd
def burst_range_spectrum(psd, snr=8, energy=1e-2):
    """Calculate the frequency-dependent GW burst range from a strain PSD

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data

    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range,
        default: `8`

    energy : `float`, optional
        the relative energy output of the GW burst,
        default: `0.01` (GRB-like burst)

    Returns
    -------
    rangespec : `~gwpy.frequencyseries.FrequencySeries`
        the burst range `FrequencySeries` [Mpc (default)]
    """
    # calculate frequency dependent range in parsecs
    a = (constants.G * energy * constants.M_sun * 0.4 /
         (pi**2 * constants.c))**(1/2.)
    dspec = psd ** (-1/2.) * a / (snr * psd.frequencies)

    # convert to output unit
    rspec = dspec.to('Mpc')

    # rescale 0 Hertz (which has 0 range always)
    if rspec.f0.value == 0.0:
        rspec[0] = 0.0

    return rspec


def burst_range(psd, snr=8, energy=1e-2, fmin=100, fmax=500):
    """Calculate the integrated GRB-like GW burst range from a strain PSD

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data

    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range,
        default: ``8``

    energy : `float`, optional
        the relative energy output of the GW burst, defaults to ``1e-2``
        for a GRB-like burst

    fmin : `float`, optional
        the lower frequency cutoff of the burst range integral,
        default: ``100 Hz``

    fmax : `float`, optional
        the upper frequency cutoff of the burst range integral,
        default: ``500 Hz``

    Returns
    -------
    range : `~astropy.units.Quantity`
        the GRB-like-burst sensitive range [Mpc (default)]

    Examples
    --------
    Grab some data for LIGO-Livingston around GW150914 and generate a PSD

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.fetch_open_data('H1', 1126259446, 1126259478)
    >>> hoff = hoft.psd(fftlength=4)

    Now we can calculate the :func:`burst_range`:

    >>> from gwpy.astro import burst_range
    >>> r = burst_range(hoff, fmin=30)
    >>> print(r)
    42.5055584195 Mpc
    """
    freqs = psd.frequencies.value
    # restrict integral
    if not fmin:
        fmin = psd.f0
    if not fmax:
        fmax = psd.span[1]
    condition = (freqs >= fmin) & (freqs < fmax)
    # calculate integrand and integrate
    integrand = burst_range_spectrum(
        psd[condition], snr=snr, energy=energy) ** 3
    result = integrate.trapz(integrand.value, freqs[condition])
    # normalize and return
    r = units.Quantity(result / (fmax - fmin), unit=integrand.unit) ** (1/3.)
    return r.to('Mpc')


def range_timeseries(hoft, stride=None, fftlength=None, overlap=None,
                     window='hann', method=DEFAULT_FFT_METHOD, nproc=1,
                     **rangekwargs):
    """Measure timeseries trends of astrophysical detector range (Mpc)
    directly from strain

    Parameters
    ----------
    hoft : `~gwpy.timeseries.TimeSeries` or `~gwpy.spectrogram.Spectrogram`
        record of gravitational-wave strain output from a detector

    stride : `float`, optional
        desired step size (seconds) of range timeseries, required if
        `hoft` is an instance of `TimeSeries`

    fftlength : `float`, optional
        number of seconds in a single FFT

    overlap : `float`, optional
        number of seconds of overlap between FFTs, defaults to the
        recommended overlap for the given window (if given), or 0

    window : `str`, `numpy.ndarray`, optional
        window function to apply to timeseries prior to FFT, see
        :func:`scipy.signal.get_window` for details on acceptable
        formats

    method : `str`, optional
        FFT-averaging method, see
        :meth:`~gwpy.timeseries.TimeSeries.spectrogram` for
        more details

    nproc : `int`, optional
        number of CPUs to use in parallel processing of FFTs, default: 1

    **rangekwargs : `dict`, optional
        additional keyword arguments to :func:`burst_range` or
        :func:`inspiral_range` (see "Notes" below), defaults to
        inspiral range with `mass1 = mass2 = 1.4` solar masses

    Returns
    -------
    out : `~gwpy.timeseries.TimeSeries`
        timeseries trends of astrophysical range

    Notes
    -----
    This method is designed to quantify a gravitational-wave detector's
    sensitive range as a function of time. It supports the range to
    compact binary inspirals and to unmodelled GW bursts, each a class
    of transient event.

    See also
    --------
    gwpy.timeseries.TimeSeries.spectrogram
        for the underlying power spectral density estimator
    inspiral_range
        for the function that computes inspiral range
    burst_range
        for the function that computes burst range
    range_spectrogram
        for a `~gwpy.spectrogram.Spectrogram` of the range integrand
    """
    out = []
    rangekwargs = rangekwargs or {'mass1': 1.4, 'mass2': 1.4}
    range_func = (burst_range if 'energy' in rangekwargs
                  else inspiral_range)
    hoft = _get_spectrogram(
        hoft, stride=stride, fftlength=fftlength, overlap=overlap,
        window=window, method=method, nproc=nproc)
    # loop over time bins
    for psd in hoft:
        out.append(range_func(psd, **rangekwargs).value)
    # finalise output
    out = TimeSeries(out)
    out.__array_finalize__(hoft)
    out.override_unit('Mpc')
    return out


def range_spectrogram(hoft, stride=None, fftlength=None, overlap=None,
                      window='hann', method=DEFAULT_FFT_METHOD, nproc=1,
                      **rangekwargs):
    """Calculate the average range or range power spectrogram (Mpc or
    Mpc^2 / Hz) directly from strain

    Parameters
    ----------
    hoft : `~gwpy.timeseries.TimeSeries`  or `~gwpy.spectrogram.Spectrogram`
        record of gravitational-wave strain output from a detector

    stride : `float`, optional
        number of seconds in a single PSD (i.e., step size of spectrogram),
        required if `hoft` is an instance of `TimeSeries`

    fftlength : `float`, optional
        number of seconds in a single FFT

    overlap : `float`, optional
        number of seconds of overlap between FFTs, defaults to the
        recommended overlap for the given window (if given), or 0

    window : `str`, `numpy.ndarray`, optional
        window function to apply to timeseries prior to FFT, see
        :func:`scipy.signal.get_window` for details on acceptable
        formats

    method : `str`, optional
        FFT-averaging method, see
        :meth:`~gwpy.timeseries.TimeSeries.spectrogram` for
        more details

    nproc : `int`, optional
        number of CPUs to use in parallel processing of FFTs, default: 1

    fmin : `float`, optional
        low frequency cut-off (Hz), defaults to `1/fftlength`

    fmax : `float`, optional
        high frequency cut-off (Hz), defaults to Nyquist frequency of `hoft`

    **rangekwargs : `dict`, optional
        additional keyword arguments to :func:`burst_range_spectrum` or
        :func:`inspiral_range_psd` (see "Notes" below), defaults to
        inspiral range with `mass1 = mass2 = 1.4` solar masses

    Returns
    -------
    out : `~gwpy.spectrogram.Spectrogram`
        time-frequency spectrogram of astrophysical range

    Notes
    -----
    This method is designed to show the contribution to a
    gravitational-wave detector's sensitive range across frequency bins
    as a function of time. It supports the range to compact binary
    inspirals and to unmodelled GW bursts, each a class of transient
    event.

    If inspiral range is requested and `fmax` exceeds the frequency of the
    innermost stable circular orbit (ISCO), the output will extend only up
    to the latter.

    See also
    --------
    gwpy.timeseries.TimeSeries.spectrogram
        for the underlying power spectral density estimator
    inspiral_range_psd
        for the function that computes inspiral range integrand
    burst_range_spectrum
        for the function that computes burst range integrand
    range_timeseries
        for `TimeSeries` trends of the astrophysical range
    """
    out = []
    rangekwargs = rangekwargs or {'mass1': 1.4, 'mass2': 1.4}
    range_func = (burst_range_spectrum if 'energy' in rangekwargs
                  else inspiral_range_psd)
    hoft = _get_spectrogram(
        hoft, stride=stride, fftlength=fftlength, overlap=overlap,
        window=window, method=method, nproc=nproc)
    # set frequency limits
    f = hoft.frequencies.to('Hz')
    fmin = units.Quantity(
        rangekwargs.pop('fmin', hoft.df),
        'Hz',
    )
    fmax = units.Quantity(
        rangekwargs.pop('fmax', f[-1]),
        'Hz',
    )
    frange = (f >= fmin) & (f < fmax)
    # loop over time bins
    for psd in hoft:
        out.append(range_func(psd[frange], **rangekwargs).value)
    # finalise output
    out = Spectrogram(out)
    out.__array_finalize__(hoft)
    out.f0 = fmin
    out.override_unit('Mpc' if 'energy' in rangekwargs
                      else 'Mpc^2 / Hz')
    return out
