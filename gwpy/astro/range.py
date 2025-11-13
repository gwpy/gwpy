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

"""Methods to calculate the sensitive distance."""

from __future__ import annotations

import warnings
from functools import wraps
from math import pi
from typing import (
    TYPE_CHECKING,
    cast,
)

from astropy import (
    constants,
    units,
)
from scipy.integrate import trapezoid
from scipy.interpolate import interp1d

from ..spectrogram import Spectrogram
from ..timeseries import TimeSeries
from ..utils import round_to_power

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        ParamSpec,
        TypeVar,
    )

    import numpy
    from inspiral_range.waveform import CBCWaveform

    from ..frequencyseries import FrequencySeries

    P = ParamSpec("P")
    R = TypeVar("R")

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__credits__ = "Alex Urban <alexander.urban@ligo.org>"

DEFAULT_FFT_METHOD = "median"

PSD_UNIT = 1 / units.Hz


def _get_isco_frequency(
    mass1: float,
    mass2: float,
) -> units.Quantity:
    """Determine the innermost stable circular orbit (ISCO) frequency.

    Parameters
    ----------
    mass1 : `float`
        The mass (in solar masses) of the first binary component.

    mass2 : `float`
        The mass (in solar masses) of the second binary component.

    Returns
    -------
    fisco : `~astropy.units.Quantity`
        Linear frequency (Hz) of the innermost stable circular orbit.
    """
    mtotal = units.Quantity(mass1 + mass2, "solMass").to("kg")
    return (
        constants.c ** 3
        / (constants.G * 6 ** 1.5 * pi * mtotal)
    ).to("Hz")


def _get_spectrogram(
    hoft: TimeSeries | Spectrogram,
    **kwargs,
) -> Spectrogram:
    """Check that the input is a spectrogram, or compute one if compatible.

    Parameters
    ----------
    hoft : `~gwpy.timeseries.TimeSeries` or `~gwpy.spectrogram.Spectrogram`
        Record of gravitational-wave strain output from a detector.

    **kwargs
        Additional keyword arguments to
        `~gwpy.timeseries.TimeSeries.spectrogram`.

    Returns
    -------
    hoft : `~gwpy.spectrogram.Spectrogram`
        A time-frequency `Spectrogram` of the input.
    """
    if not isinstance(hoft, Spectrogram):
        try:
            hoft = hoft.spectrogram(**kwargs)
        except (
            AttributeError,  # object doesn't have a `.spectrogram()` method
            TypeError,  # something else went wrong
        ) as exc:
            msg = (
                "Could not produce a spectrogram from the input, please "
                "pass an instance of gwpy.timeseries.TimeSeries or "
                "gwpy.spectrogram.Spectrogram"
            )
            raise TypeError(msg) from exc
    return hoft


def _preformat_psd(
    func: Callable[P, R],
) -> Callable[P, R]:
    """Wrap a function to ensure that the incoming PSD has the right units."""

    @wraps(func)
    def decorated_func(*args: P.args, **kwargs: P.kwargs) -> R:
        psd = cast("FrequencySeries", args[0])
        # Force PSD to have the right units
        if psd.unit != PSD_UNIT:
            psd = psd.view()
            psd.override_unit(PSD_UNIT)
        args = (psd, *args[1:])  # type: ignore[assignment]
        return func(*args, **kwargs)
    return decorated_func


# -- SenseMon range ------------------

@_preformat_psd
def sensemon_range_psd(
    psd: FrequencySeries,
    snr: float = 8,
    mass1: float = 1.4,
    mass2: float = 1.4,
    *,
    horizon: bool = False,
) -> FrequencySeries:
    """Approximate the inspiral sensitive distance PSD from a GW strain PSD.

    This method returns the power spectral density (in ``Mpc**2 / Hz``) to
    which a compact binary inspiral with the given component masses would
    be detectable given the instrumental PSD. The calculation is defined in:
    :dcc:`LIGO-T030276`.

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        The instrumental power-spectral-density data.

    snr : `float`, optional
        The signal-to-noise ratio for which to calculate range.

    mass1 : `float`, `~astropy.units.Quantity`, optional
        The mass of the first binary component
        (`float` assumed in solar masses).

    mass2 : `float`, `~astropy.units.Quantity`, optional
        The mass of the second binary component
        (`float` assumed in solar masses).

    horizon : `bool`, optional
        If `True`, return the maximal 'horizon' sensitive distance, otherwise
        return the angle-averaged range.

    Returns
    -------
    rspec : `~gwpy.frequencyseries.FrequencySeries`
        The calculated inspiral sensitivity PSD [Mpc^2 / Hz].
    """
    frange = (psd.frequencies > 0)
    # compute total mass and chirp mass
    mass1 = units.Quantity(mass1, "solMass").to("kg")
    mass2 = units.Quantity(mass2, "solMass").to("kg")
    mtotal = mass1 + mass2
    mchirp = (mass1 * mass2) ** (3 / 5.) / mtotal ** (1 / 5.)
    # calculate integrand with pre-factor
    prefactor = (
        (  # numerator
            (16 if horizon else 1.77 ** 2)
            * 5
            * constants.c ** (1 / 3.)
            * (mchirp * constants.G / constants.c ** 2) ** (5 / 3.)
        )  # denominator
        / (
            96
            * pi ** (4 / 3.)
            * snr ** 2
        )
    )
    return (  # inspiral range PSD, avoiding DC value
        1
        / psd[frange]
        * psd.frequencies[frange] ** (-7 / 3.)
        * prefactor
    ).to("Mpc^2 / Hz")


def sensemon_range(
    psd: FrequencySeries,
    snr: float = 8,
    mass1: float = 1.4,
    mass2: float = 1.4,
    fmin: float | None = None,
    fmax: float | None = None,
    *,
    horizon: bool = False,
) -> units.Quantity:
    """Approximate the inspiral sensitive distance from a GW strain PSD.

    This method returns the distance (in megaparsecs) to which a compact
    binary inspiral with the given component masses would be detectable
    given the instrumental PSD. The calculation is as defined in:
    :dcc:`LIGO-T030276`.

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        The instrumental power-spectral-density data

    snr : `float`, optional
        The signal-to-noise ratio for which to calculate range.

    mass1 : `float`, `~astropy.units.Quantity`, optional
        The mass of the first binary component
        (`float` assumed in solar masses).

    mass2 : `float`, `~astropy.units.Quantity`, optional
        The mass of the second binary component
        (`float` assumed in solar masses).

    fmin : `float`, optional
        The lower frequency cut-off of the integral, default: `psd.df`.

    fmax : `float`, optional
        The maximum frequency limit of the integral, defaults to
        innermost stable circular orbit (ISCO) frequency

    horizon : `bool`, optional
        If `True`, return the maximal 'horizon' sensitive distance, otherwise
        return the angle-averaged range.

    Returns
    -------
    range : `~astropy.units.Quantity`
        the calculated inspiral range [Mpc]

    Examples
    --------
    Grab some data for LIGO-Livingston around GW150914 and generate a PSD:

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
    >>> hoff = hoft.psd(fftlength=4)

    Now we can calculate the :func:`sensemon_range`:

    >>> from gwpy.astro import sensemon_range
    >>> r = sensemon_range(hoff, fmin: float = 30)
    >>> print(r)
    70.4612102889 Mpc
    """
    fisco = _get_isco_frequency(mass1, mass2)
    # format frequency limits
    fmin = units.Quantity(fmin or psd.df, "Hz")  # avoid DC value
    fmax = units.Quantity(fmax or fisco, "Hz")
    if fmax > fisco:
        warnings.warn(
            f"Upper frequency bound greater than {mass1}-{mass2} ISCO "
            f"frequency of {fisco}, using ISCO",
            stacklevel=2,
        )
        fmax = fisco
    # integrate and return
    f = psd.frequencies.to("Hz")
    frange = (f >= fmin) & (f < fmax)
    integrand = sensemon_range_psd(
        psd[frange],
        snr=snr,
        mass1=mass1,
        mass2=mass2,
        horizon=horizon,
    )
    return (units.Quantity(
        trapezoid(integrand.value, f.value[frange]),
        unit=integrand.unit * units.Hertz,
    ) ** (1 / 2.)).to("Mpc")


# -- inspiral range ------------------

MISSING_INSPIRAL_RANGE_MESSAGE = (
    "gwpy.astro's inspiral_range and inspiral_range_psd functions "
    "require the extra package 'inspiral-range' to provide "
    "cosmologically-corrected distance calculations, please install "
    "that package and try again, or install gwpy with the 'astro' "
    "extra via `python -m pip install gwpy[astro]`"
)


def _cbc_waveform(
    frequencies: numpy.ndarray,
    m1: float,
    m2: float,
    **kwargs,
) -> CBCWaveform:
    """Generate a CBCWaveform."""
    try:
        from inspiral_range.waveform import CBCWaveform
    except ImportError as exc:
        exc.args = (f"{exc}; {MISSING_INSPIRAL_RANGE_MESSAGE}",)
        raise

    return CBCWaveform(
        frequencies,
        m1=m1,
        m2=m2,
        **kwargs,
    )


@_preformat_psd
def inspiral_range_psd(
    psd: FrequencySeries,
    snr: float = 8,
    mass1: float = 1.4,
    mass2: float = 1.4,
    *,
    horizon: bool = False,
    **kwargs,
) -> FrequencySeries:
    """Calculate the cosmology-corrected inspiral sensitive distance PSD.

    This method returns the power spectral density (in ``Mpc**2 / Hz``) to
    which a compact binary inspiral with the given component masses would
    be detectable given the instrumental PSD. The calculation is defined in
    Belczynski et. al (2014):

    https://dx.doi.org/10.1088/0004-637x/789/2/120

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        The instrumental power-spectral-density data.

    snr : `float`, optional
        The signal-to-noise ratio for which to calculate range.

    mass1 : `float`, `~astropy.units.Quantity`, optional
        The mass of the first binary component
        (`float` assumed in solar masses).

    mass2 : `float`, `~astropy.units.Quantity`, optional
        The mass of the second binary component
        (`float` assumed in solar masses).

    horizon : `bool`, optional
        If `True`, return the maximal 'horizon' luminosity distance, otherwise
        return the angle-averaged comoving distance.

    **kwargs
        Additional keyword arguments to `~inspiral_range.waveform.CBCWaveform`.

    Returns
    -------
    rspec : `~gwpy.frequencyseries.FrequencySeries`
        The calculated inspiral sensitivity PSD [Mpc^2 / Hz].

    See Also
    --------
    sensemon_range_psd
        For the method based on :dcc:`LIGO-T030276`, also known as LIGO SenseMonitor.

    inspiral-range
        The package which does heavy lifting for waveform simulation and
        cosmology calculations.
    """
    try:
        from inspiral_range import (
            find_root_redshift,
            range as range_func,
        )
    except ImportError as exc:
        exc.msg = f"{exc}; {MISSING_INSPIRAL_RANGE_MESSAGE}"
        raise

    f = psd.frequencies.to("Hz").value
    frange = (f > 0)

    # generate a waveform for this system
    inspiral = _cbc_waveform(
        f[frange],
        m1=mass1,
        m2=mass2,
        **kwargs,
    )

    # determine the detector horizon redshift
    z_hor = find_root_redshift(
        lambda z: inspiral.SNR(psd.value[frange], z) - snr,
    )
    if horizon:
        dist = inspiral.cosmo.luminosity_distance(z_hor)
    else:
        dist = range_func(
            f[frange],
            psd.value[frange],
            z_hor=z_hor,
            H=inspiral,
            detection_snr=snr,
        )

    # calculate the sensitive distance PSD
    (fz, hz) = inspiral.z_scale(z_hor)
    hz = interp1d(
        fz,
        hz,
        bounds_error=False,
        fill_value=(hz[0], 0),
    )(f)
    out = type(psd)(
        4
        * (dist / snr)**2
        * (hz**2 / psd.value)[f > 0],
    )

    # finalize properties and return
    out.__array_finalize__(psd)
    out.override_unit("Mpc^2 / Hz")
    out.f0 = f[f > 0][0]
    return out


@_preformat_psd
def inspiral_range(
    psd: FrequencySeries,
    snr: float = 8,
    mass1: float = 1.4,
    mass2: float = 1.4,
    fmin: float | None = None,
    fmax: float | None = None,
    *,
    horizon: bool = False,
    **kwargs,
) -> units.Quantity:
    """Calculate the cosmology-corrected inspiral sensitive distance.

    This method returns the distance (in megaparsecs) to which a compact
    binary inspiral with the given component masses would be detectable
    given the instrumental PSD. The calculation is defined in Belczynski
    et. al (2014):

    https://dx.doi.org/10.1088/0004-637x/789/2/120

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        The instrumental power-spectral-density data.

    snr : `float`, optional
        The signal-to-noise ratio for which to calculate range.

    mass1 : `float`, `~astropy.units.Quantity`, optional
        The mass of the first binary component
        (`float` assumed in solar masses).

    mass2 : `float`, `~astropy.units.Quantity`, optional
        The mass of the second binary component
        (`float` assumed in solar masses).

    fmin : `float`, optional
        The lower frequency cut-off of the integral, default: `psd.df`.

    fmax : `float`, optional
        The maximum frequency limit of the integral, defaults to the rest-frame
        innermost stable circular orbit (ISCO) frequency.

    horizon : `bool`, optional
        If `True`, return the maximal 'horizon' luminosity distance, otherwise
        return the angle-averaged comoving distance.

    **kwargs
        Additional keyword arguments to `~inspiral_range.waveform.CBCWaveform`.

    Returns
    -------
    range : `~astropy.units.Quantity`
        The calculated inspiral range [Mpc].

    Examples
    --------
    Grab some data for LIGO-Livingston around GW150914 and generate a PSD:

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
    >>> hoff = hoft.psd(fftlength=4)

    Now, we can calculate the :func:`inspiral_range`:

    >>> from gwpy.astro import inspiral_range
    >>> r = inspiral_range(hoff, fmin: float = 30)
    >>> print(r)
    70.4612102889 Mpc

    See Also
    --------
    sensemon_range
        For the method based on :dcc:`LIGO-T030276`, also known as LIGO SenseMonitor.

    inspiral-range
        The package which does heavy lifting for waveform simulation and
        cosmology calculations.
    """
    try:
        from inspiral_range import (
            find_root_redshift,
            range as range_func,
        )
    except ImportError as exc:
        exc.msg = f"{exc}; {MISSING_INSPIRAL_RANGE_MESSAGE}"
        raise

    # format frequency limits
    f = psd.frequencies.to("Hz").value
    fmin = fmin or psd.df.value  # avoid DC value
    fmax = fmax or round_to_power(f[-1], which="lower")
    frange = (f >= fmin) & (f < fmax)

    # generate a waveform for this system
    inspiral = _cbc_waveform(
        f[frange],
        m1=mass1,
        m2=mass2,
        **kwargs,
    )

    # determine the detector horizon redshift
    z_hor = find_root_redshift(
        lambda z: inspiral.SNR(psd.value[frange], z=z) - snr,
    )

    # return the sensitive distance metric
    if horizon:
        dist = inspiral.cosmo.luminosity_distance(z_hor)
    else:
        dist = range_func(
            f[frange],
            psd.value[frange],
            z_hor=z_hor,
            H=inspiral,
            detection_snr=snr,
        )
    return units.Quantity(dist, unit="Mpc")


# -- burst range ---------------------

@_preformat_psd
def burst_range_spectrum(
    psd: FrequencySeries,
    snr: float = 8,
    energy: float = 1e-2,
) -> FrequencySeries:
    """Calculate the frequency-dependent GW burst range from a strain PSD.

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        The instrumental power-spectral-density data.

    snr : `float`, optional
        The signal-to-noise ratio for which to calculate range.

    energy : `float`, optional
        The relative energy output of the GW burst.

    Returns
    -------
    rangespec : `~gwpy.frequencyseries.FrequencySeries`
        The burst range `FrequencySeries` [Mpc (default)].
    """
    frange = (psd.frequencies > 0)
    # calculate frequency dependent range in parsecs
    a = (
        constants.G
        * energy
        * constants.M_sun
        * 0.4
        / (pi**2 * constants.c)
    ) ** (1 / 2.)
    return (  # burst range spectrum, avoiding DC value
        psd[frange] ** (-1 / 2.)
        * a
        / (snr * psd.frequencies[frange])
    ).to("Mpc")


def burst_range(
    psd: FrequencySeries,
    snr: float = 8,
    energy: float = 1e-2,
    fmin: float = 100,
    fmax: float = 500,
) -> units.Quantity:
    """Calculate the integrated GRB-like GW burst range from a strain PSD.

    Parameters
    ----------
    psd : `~gwpy.frequencyseries.FrequencySeries`
        The instrumental power-spectral-density data.

    snr : `float`, optional
        The signal-to-noise ratio for which to calculate range.

    energy : `float`, optional
        The relative energy output of the GW burst.

    fmin : `float`, optional
        The lower frequency cutoff of the burst range integral.

    fmax : `float`, optional
        The upper frequency cutoff of the burst range integral.

    Returns
    -------
    range : `~astropy.units.Quantity`
        The GRB-like-burst sensitive range [Mpc (default)].

    Examples
    --------
    Grab some data for LIGO-Livingston around GW150914 and generate a PSD

    >>> from gwpy.timeseries import TimeSeries
    >>> hoft = TimeSeries.fetch_open_data("H1", 1126259446, 1126259478)
    >>> hoff = hoft.psd(fftlength=4)

    Now we can calculate the :func:`burst_range`:

    >>> from gwpy.astro import burst_range
    >>> r = burst_range(hoff, fmin: float = 30)
    >>> print(r)
    42.5055584195 Mpc
    """
    f = psd.frequencies.to("Hz").value
    # restrict integral
    fmin = fmin or psd.df.to("Hz").value
    fmax = fmax or f[-1].value
    frange = (f >= fmin) & (f < fmax)
    # calculate integrand and integrate
    integrand = burst_range_spectrum(
        psd[frange],
        snr=snr,
        energy=energy,
    ) ** 3
    out = trapezoid(integrand.value, f[frange])
    # normalize and return
    return (units.Quantity(
        out / (fmax - fmin),
        unit=integrand.unit,
    ) ** (1 / 3.)).to("Mpc")


# -- timeseries/spectrogram wrappers -

def range_timeseries(
    hoft: TimeSeries | Spectrogram,
    stride: float | None = None,
    fftlength: float | None = None,
    overlap: float | None = None,
    window: str | numpy.ndarray = "hann",
    method: str = DEFAULT_FFT_METHOD,
    nproc: int = 1,
    range_func: Callable | None = None,
    **rangekwargs,
) -> TimeSeries:
    """Estimate timeseries of astrophysical range (Mpc).

    Parameters
    ----------
    hoft : `~gwpy.timeseries.TimeSeries`, `~gwpy.spectrogram.Spectrogram`
        Detector (strain) data from which to estimate range.

    stride : `float`, optional
        Desired step size (seconds) of range timeseries, required if
        `hoft` is an instance of `TimeSeries`.

    fftlength : `float`, optional
        Number of seconds in a single FFT.

    overlap : `float`, optional
        Number of seconds of overlap between FFTs, defaults to the
        recommended overlap for the given window (if given), or ``0``.

    window : `str`, `numpy.ndarray`, optional
        Window function to apply to timeseries prior to FFT, see
        :func:`scipy.signal.get_window` for details on acceptable
        formats.

    method : `str`, optional
        FFT-averaging method, defaults to median averaging, see
        :meth:`~gwpy.timeseries.TimeSeries.spectrogram` for
        more details

    nproc : `int`, optional
        number of CPUs to use in parallel processing of FFTs, default: 1

    range_func : `callable`, optional
        the function to call to generate the range for each stride,
        defaults to ``inspiral_range`` unless ``energy`` is given
        as a keyword argument to the range function

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

    See Also
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
    if not isinstance(hoft, Spectrogram):
        hoft = _get_spectrogram(
            hoft,
            stride=stride,
            fftlength=fftlength,
            overlap=overlap,
            window=window,
            method=method,
            nproc=nproc,
        )

    rangekwargs = rangekwargs or {
        "mass1": 1.4,
        "mass2": 1.4,
    }
    if range_func is None and "energy" in rangekwargs:
        range_func = burst_range
    elif range_func is None:
        range_func = inspiral_range

    # calculate the timeseries of range
    out = TimeSeries(
        [range_func(psd, **rangekwargs).value for psd in hoft],
    )
    out.__array_finalize__(hoft)
    out.override_unit("Mpc")
    return out


def range_spectrogram(
    hoft: TimeSeries | Spectrogram,
    stride: float | None = None,
    fftlength: float | None = None,
    overlap: float | None = None,
    window: str | numpy.ndarray = "hann",
    method: str = DEFAULT_FFT_METHOD,
    nproc: int = 1,
    range_func: Callable | None = None,
    **rangekwargs,
) -> Spectrogram:
    """Calculate the average range spectrogram (Mpc or Mpc^2 / Hz) from strain.

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
        FFT-averaging method, defaults to median averaging, see
        :meth:`~gwpy.timeseries.TimeSeries.spectrogram` for
        more details

    nproc : `int`, optional
        number of CPUs to use in parallel processing of FFTs, default: 1

    fmin : `float`, optional
        low frequency cut-off (Hz), defaults to `1/fftlength`

    fmax : `float`, optional
        high frequency cut-off (Hz), defaults to Nyquist frequency of `hoft`

    range_func : `callable`, optional
        the function to call to generate the range for each stride,
        defaults to ``inspiral_range`` unless ``energy`` is given
        as a keyword argument to the range function

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

    See Also
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
    rangekwargs = rangekwargs or {
        "mass1": 1.4,
        "mass2": 1.4,
    }
    if range_func is None and "energy" in rangekwargs:
        range_func = burst_range_spectrum
    elif range_func is None:
        range_func = inspiral_range_psd
    hoft = _get_spectrogram(
        hoft,
        stride=stride,
        fftlength=fftlength,
        overlap=overlap,
        window=window,
        method=method,
        nproc=nproc,
    )
    # set frequency limits
    f = hoft.frequencies.to("Hz")
    fmin = units.Quantity(
        rangekwargs.pop("fmin", hoft.df),
        "Hz",
    )
    fmax = units.Quantity(
        rangekwargs.pop(
            "fmax",
            round_to_power(f[-1].value, which="lower"),
        ),
        "Hz",
    )
    frange = (f >= fmin) & (f < fmax)
    # loop over time bins
    out = Spectrogram(
        [range_func(psd[frange], **rangekwargs).value for psd in hoft],
    )
    # finalise output
    out.__array_finalize__(hoft)
    out.override_unit("Mpc" if "energy" in rangekwargs else "Mpc^2 / Hz")
    out.f0 = fmin
    return out
