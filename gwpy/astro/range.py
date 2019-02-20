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

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


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
