# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014)
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
from math import pi

from scipy import integrate

from astropy import (units, constants)

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def inspiral_range_psd(psd, snr=8, mass1=1.4, mass2=1.4, horizon=False):
    """Compute the inspiral sensitive distance PSD for the given GW strain PSD

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data
    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range
    mass1 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the first binary
        component
    mass2 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the second binary
    horizon : `bool`, optional
        if `True`, return the maximal 'horizon' sensitive distance, otherwise
        return the angle-averaged range

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
    # calculate inspiral range ASD in m
    integrand = (
        prefactor.value * psd.frequencies.value ** (-7/3.) / psd /
        units.Mpc.decompose().scale ** 2)
    integrand.override_unit(units.Unit('Mpc^2 / Hz'))
    # restrict to ISCO
    integrand = integrand[psd.frequencies.value < fisco.value]
    # normalize and return
    if integrand.f0.value == 0.0:
        integrand[0] = 0.0
    if horizon:
        integrand *= 2.26 ** 2
    return integrand


def inspiral_range(psd, snr=8, mass1=1.4, mass2=1.4, fmin=0, fmax=None,
                   horizon=False, unit='Mpc'):
    """Calculate the inspiral sensitive distance for the given PSD

    The method returns the distance (in megaparsecs) to which an compact
    binary inspiral with the given component masses would be detectable
    given the instrumental PSD. The calculation is as defined in:

    https://dcc.ligo.org/LIGO-T030276/public

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data
    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range
    mass1 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the first binary
        component
    mass2 : `float`, `~astropy.units.Quantity`, optional
        the mass (`float` assumed in solar masses) of the second binary
    fmin : `float`, optional
        the lower frequency cut-off of the integral
    fmax : `float`, optional
        the maximum frequency limit of the integral, if not given or `None`,
        the innermost stable circular orbit (ISCO) frequency is used
    horizon : `bool`, optional
        if `True`, return the maximal 'horizon' sensitive distance, otherwise
        return the angle-averaged range

    Returns
    -------
    range : `~astropy.units.Quantity`
        the calculated inspiral range [Mpc]
    """
    mass1 = units.Quantity(mass1, 'solMass').to('kg')
    mass2 = units.Quantity(mass2, 'solMass').to('kg')
    mtotal = mass1 + mass2

    # compute ISCO
    fisco = (constants.c ** 3 / (constants.G * 6**1.5 * pi * mtotal)).to('Hz')
    if not fmax:
        fmax = fisco
    fmax = units.Quantity(fmax, 'Hz')
    if fmax > fisco:
        warnings.warn("Upper frequency bound greater than %s-%s ISCO "
                      "frequency of %s, using ISCO" % (mass1, mass2, fisco))
        fmax = fisco
    fmin = units.Quantity(fmin, 'Hz')

    # integrate
    f = psd.frequencies.to('Hz')
    condition = (f >= fmin) & (f < fmax)
    integrand = inspiral_range_psd(psd[condition], snr=snr, mass1=mass1,
                                   mass2=mass2, horizon=horizon)
    if fmin.value == 0.0:
        integrand[0] = 0.0
    result = units.Quantity(integrate.trapz(integrand.value,
                                            f.value[condition]),
                            unit=integrand.unit * units.Hertz)

    return (result ** (1/2.)).to(unit)


def burst_range_spectrum(psd, snr=8, energy=1e-2, unit='Mpc'):
    """Calculate the frequency-dependent GW burst range for the given PSD

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data
    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range
    energy : `float`, optional
        the relative energy output of the GW burst, defaults to 1e-2 for
        a GRB-like burst
    unit : `str`, `~astropy.units.Unit`
        desired unit of the returned `~astropy.units.Quantity`

    Returns
    -------
    rangespec : `~gwpy.frequencyseries.FrequencySeries`
        the burst range `FrequencySeries` [Mpc (default)]
    """
    unit = units.Unit(unit)
    a = (constants.G.value * energy * constants.M_sun.value * 0.4 /
         (pi**2 * constants.c.value))**(1/2.)
    dspec = a / (snr * psd**(1/2.) * psd.frequencies) / constants.pc.value
    conv = units.pc.get_converter(unit)
    rspec = conv(dspec)
    rspec.override_unit(unit)
    if rspec.f0.value == 0.0:
        rspec[0] = 0.0
    return rspec


def burst_range(psd, snr=8, energy=1e-2, fmin=100, fmax=500, unit='Mpc'):
    """Calculate the integrated GRB-like burst range for the given PSD

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        the instrumental power-spectral-density data
    snr : `float`, optional
        the signal-to-noise ratio for which to calculate range
    energy : `float`, optional
        the relative energy output of the GW burst, defaults to 1e-2 for
        a GRB-like burst
    fmin : `float`, optional
        the lower frequency cutoff of the burst range integral
    fmax : `float, optional
        the upper frequency cutoff of the burst range integral
    unit : `str`, `~astropy.units.Unit`
        desired unit of the returned `~astropy.units.Quantity`

    Returns
    -------
    range : `~astropy.units.Quantity`
        the GRB-like-burst sensitive range [Mpc (default)]
    """
    freqs = psd.frequencies.value
    # restrict integral
    if not fmin:
        fmin = freqs.min()
    if not fmax:
        fmax = freqs.max()
    condition = (freqs >= fmin) & (freqs < fmax)
    # calculate integrand and integrate
    integrand = burst_range_spectrum(
        psd[condition], snr=snr, energy=energy) ** 3
    result = integrate.trapz(integrand.value, freqs[condition])
    # normalize and return
    r = units.Quantity(result / (fmax - fmin), unit=integrand.unit) ** (1/3.)
    return r.to(unit)
