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

"""Analogue and digital filter design utilities.

This module is mainly a wrapper around `scipy.signal` filter design,
with convenience functions for common filter types, and support for
LIGO-specific filter design conventions.
"""

from __future__ import annotations

import operator
from functools import reduce
from math import (
    log10,
    pi,
)
from typing import (
    TYPE_CHECKING,
    cast,
    overload,
)

import numpy
from astropy.units import (
    Quantity,
    Unit,
)
from numpy import fft as npfft
from scipy import signal
from scipy.signal.windows import tukey

from .window import get_window

# filter type definitions
if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import (
        Literal,
        TypeAlias,
        TypeVar,
    )

    from astropy.units import UnitBase
    from astropy.units.typing import QuantityLike
    from numpy.typing import (
        ArrayLike,
        NDArray,
    )

    from ..typing import (
        Array1D,
        ArrayLike1D,
    )
    from .window import WindowLike

    TArray = TypeVar("TArray", bound=Array1D)

    LinearTimeInvariant: TypeAlias = signal.lti | signal.dlti

    # FIR
    TapsType = Array1D
    # IIR
    FilterTypeName: TypeAlias = Literal["butter", "cheby1", "cheby2", "ellip"]
    SosType: TypeAlias = numpy.ndarray[tuple[int, Literal[6]]]
    ZpkCompatible: TypeAlias = tuple[ArrayLike1D, ArrayLike1D, float]
    ZpkType: TypeAlias = tuple[Array1D, Array1D, float]
    BAType: TypeAlias = tuple[Array1D, Array1D]
    IirFilterCompatible: TypeAlias = SosType | ZpkCompatible | BAType
    IirFilterType: TypeAlias = SosType | ZpkType | BAType
    # generic
    FilterCompatible: TypeAlias = TapsType | IirFilterCompatible | LinearTimeInvariant
    FilterType: TypeAlias = TapsType | IirFilterType | LinearTimeInvariant

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"
__all__ = [
    "bandpass",
    "concatenate_zpks",
    "fir_from_transfer",
    "frequency_response",
    "highpass",
    "is_sos",
    "is_zpk",
    "lowpass",
    "notch",
    "parse_filter",
    "prepare_analog_filter",
    "prepare_digital_filter",
]

HERTZ = Unit("Hz")
RAD_S = Unit("rad/s")

TWO_PI: float = 2 * pi

# All ZPK filters have three components
ZPK_NARGS = 3

# All SOS filters are 2D arrays with six coefficients per section
SOS_NDIM = 2
SOS_NCOEFF = 6


# -- Utilities -----------------------

def _as_float(
    x: QuantityLike,
    unit: str | UnitBase = "Hz",
) -> float:
    """Convert input to a float in the given units."""
    return Quantity(x, unit).value


def _is_hertz(unit: str | UnitBase) -> bool:
    """Return `True` if ``unit`` represents Hertz."""
    try:
        return Unit(unit) == HERTZ
    except ValueError:
        return False


def _is_radians(unit: str | UnitBase) -> bool:
    """Return `True` if ``unit`` represents radians per second or sample."""
    # rad/sample isn't an astropy Unit, so check string manually
    if unit in ("rad/s", "rad/sample"):
        return True
    try:
        return Unit(unit) == RAD_S
    except ValueError:
        return False


def is_zpk(zpktup: object) -> bool:
    """Return `True` if ``zpktup`` looks like a ZPK-format filter definition.

    Returns
    -------
    iszpk : `bool`
        `True` if input argument looks like a 3-tuple giving arrays of
        zeros and poles, and a gain (`float`).
    """
    return (
        # Input object is a length 3 tuple/list
        isinstance(zpktup, tuple | list)
        and len(zpktup) == ZPK_NARGS
        # First two elements are 1D arrays/lists/tuples
        and isinstance(zpktup[0], list | tuple | numpy.ndarray)
        and numpy.ndim(cast("ArrayLike", zpktup[0])) == 1
        and isinstance(zpktup[1], list | tuple | numpy.ndarray)
        and numpy.ndim(cast("ArrayLike", zpktup[1])) == 1
        # Third element is a float
        and isinstance(zpktup[2], float)
    )


def is_sos(sos: object) -> bool:
    """Return `True` if ``sos`` looks like a SOS-format filter definition.

    Returns
    -------
    issos : `bool`
        `True` if input argument looks like a 2D-array giving second-order
        sections.
    """
    return (
        isinstance(sos, numpy.ndarray)
        and sos.ndim == SOS_NDIM
        and sos.shape[1] == SOS_NCOEFF
    )


def concatenate_zpks(*zpks: ZpkCompatible) -> ZpkType:
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


def num_taps(
    sample_rate: float,
    transitionwidth: float,
    gpass: float,
    gstop: float,
) -> int:
    """Return the number of taps for an FIR filter with the given shape.

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


# -- Formatting / conversions --------

@overload
def _transform(
    filt: BAType | ZpkType | SosType,
    from_: Literal["zpk", "ba", "sos"],
    to_: Literal["zpk"],
) -> ZpkType: ...
@overload
def _transform(
    filt: BAType | ZpkType | SosType,
    from_: Literal["zpk", "ba", "sos"],
    to_: Literal["ba"],
) -> BAType: ...
@overload
def _transform(
    filt: BAType | ZpkType | SosType,
    from_: Literal["zpk", "ba", "sos"],
    to_: Literal["sos"],
) -> SosType: ...

def _transform(  # noqa: PLR0911
    filt: BAType | ZpkType | SosType,
    from_: Literal["zpk", "ba", "sos"],
    to_: Literal["zpk", "ba", "sos"],
    **kwargs,
) -> BAType | ZpkType | SosType:
    """Transform a filter between different representations.

    Parameters
    ----------
    filt : `tuple` or `numpy.ndarray`
        Filter definition in one of the supported formats.

    from_ : `str`
        Format of input filter definition.
        One of: ``'zpk'``, ``'ba'``, ``'sos'``.

    to_ : `str`
        Format of output filter definition.
        One of: ``'zpk'``, ``'ba'``, ``'sos'``.

    kwargs
        Additional keyword arguments passed to the underlying
        `scipy.signal` conversion functions.

    Returns
    -------
    filt_converted : `tuple` or `numpy.ndarray`
        Filter definition in the requested format.

    Raises
    ------
    ValueError
        If `from_` or `to_` are not one of the supported formats.

    See Also
    --------
    scipy.signal.sos2tf
    scipy.signal.sos2zpk
    scipy.signal.tf2sos
    scipy.signal.tf2zpk
    scipy.signal.zpk2sos
    scipy.signal.zpk2tf
    """
    # Null transformation
    if from_ == to_:
        return filt

    if to_ == "zpk" and from_ == "ba":
        filt = cast("BAType", filt)
        return signal.tf2zpk(*filt, **kwargs)

    if to_ == "zpk" and from_ == "sos":
        filt = cast("SosType", filt)
        return signal.sos2zpk(filt, **kwargs)

    if to_ == "ba" and from_ == "zpk":
        filt = cast("ZpkType", filt)
        return signal.zpk2tf(*filt, **kwargs)

    if to_ == "ba" and from_ == "sos":
        filt = cast("SosType", filt)
        return signal.sos2tf(filt, **kwargs)

    if to_ == "sos" and from_ == "zpk":
        filt = cast("ZpkType", filt)
        return signal.zpk2sos(*filt, **kwargs)

    if to_ == "sos" and from_ == "ba":
        filt = cast("BAType", filt)
        return signal.tf2sos(*filt, **kwargs)

    msg = f"unknown filter transformation: '{from_}' -> '{to_}'"
    raise ValueError(msg)


def _normalize_gain_hz_to_rad(
    zeros: Array1D,
    poles: Array1D,
    gain: float,
) -> float:
    """Normalize gain when converting Hz to rad/s.

    This preserves the frequency response magnitude by scaling the gain
    according to: k' = k * |∏p_i / ∏z_i| * (2π)^(n_p - n_z)

    Parameters
    ----------
    zeros : `numpy.ndarray`
        Zeros in rad/s (already converted from Hz).

    poles : `numpy.ndarray`
        Poles in rad/s (already converted from Hz).

    gain : `float`
        Original gain (from Hz specification).

    Returns
    -------
    normalized_gain : `float`
        Gain scaled to preserve frequency response.
    """
    # Compute product of non-zero poles and zeros
    nonzero_poles = poles[poles != 0]
    nonzero_zeros = zeros[zeros != 0]
    if len(nonzero_zeros) > 0 and len(nonzero_poles) > 0:
        # Scale by ratio of pole to zero magnitudes
        gain *= numpy.abs(numpy.prod(nonzero_poles) / numpy.prod(nonzero_zeros))

    # Account for poles/zeros at the origin
    n_poles_at_origin = poles.size - nonzero_poles.size
    n_zeros_at_origin = zeros.size - nonzero_zeros.size

    # Poles at origin: multiply by (2π)^n_p
    # Zeros at origin: divide by (2π)^n_z
    if n_poles_at_origin > 0 or n_zeros_at_origin > 0:
        gain *= TWO_PI ** (n_poles_at_origin - n_zeros_at_origin)

    return gain


def _convert_zpk_units(
    zpk: ZpkCompatible,
    unit: str | UnitBase,
    *,
    normalize_gain: bool = False,
) -> ZpkType:
    """Convert an analogue ZPK between unit conventions.

    Parameters
    ----------
    zpk : `tuple`
        (zeros, poles, gain) tuple.

    unit : `str`
        The units in which the zeros and poles are specified.
        Either ``'Hz'`` or ``'rad/s'``.

    normalize_gain : `bool`, optional
        Whether to normalize the gain when converting from Hz to rad/s.

        - `False` (default):
          Multiply zeros/poles by -2π but leave gain unchanged.
          This matches the LIGO GDS **'f' plane** convention
          (``plane='f'`` in ``s2z()``).

        - `True`:
          Normalize gain to preserve frequency response magnitude.
          Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
          Use this when your filter was designed with the transfer
          function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
          This matches the LIGO GDS **'n' plane** convention
          (``plane='n'`` in ``s2z()``).

        Ignored when `unit='rad/s'`.

    Returns
    -------
    zeros : `numpy.ndarray` of `numpy.cfloat`
        Converted zeros in rad/s.

    poles : `numpy.ndarray` of `numpy.cfloat`
        Converted poles in rad/s.

    gain : `float`
        Adjusted gain accounting for unit conversion and normalization.

    Warns
    -----
    UserWarning
        If `normalize_gain` is set when `unit='rad/s'` (normalization is ignored).

    Raises
    ------
    ValueError
        If `unit` is not 'Hz' or 'rad/s'.

    Notes
    -----
    This function converts TO rad/s (s-plane). For conversions to other
    representations, use `scipy.signal.bilinear_zpk()` for z-plane, or
    `scipy.signal.zpk2tf()` for transfer function form.
    """
    zeros, poles, gain = zpk
    zeros = numpy.array(zeros, dtype=numpy.complex128)
    poles = numpy.array(poles, dtype=numpy.complex128)

    if _is_hertz(unit):
        # Convert frequencies from Hz to rad/s
        for zi in range(len(zeros)):
            zeros[zi] *= -TWO_PI
        for pj in range(len(poles)):
            poles[pj] *= -TWO_PI

        # Apply gain normalization if requested
        if normalize_gain:
            gain = _normalize_gain_hz_to_rad(zeros, poles, gain)

    elif _is_radians(unit):
        # Already in correct units, warn if normalize_gain was set
        if normalize_gain:
            import warnings
            warnings.warn(
                f"normalize_gain parameter is ignored when unit='{unit}' "
                "(filter is already in rad/s)",
                UserWarning,
                stacklevel=2,
            )
    else:
        msg = (
            "zpk can only be given with unit='Hz', 'rad/s', or 'rad/sample', "
            f"not '{unit}'"
        )
        raise ValueError(msg)

    return zeros, poles, gain


@overload
def _convert_to_digital(
    filt: SosType | ZpkType | LinearTimeInvariant,
    sample_rate: float,
) -> tuple[Literal["zpk"], ZpkType]: ...
@overload
def _convert_to_digital(
    filt: TapsType | BAType,
    sample_rate: float,
) -> tuple[Literal["ba"], BAType]: ...

def _convert_to_digital(
    filt: FilterType,
    sample_rate: float,
) -> tuple[Literal["ba", "zpk"], IirFilterType]:
    """Convert an analog filter to digital via bilinear functions.

    Parameters
    ----------
    filt: `tuple`
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
    form, filt_ = parse_filter(filt)

    if form == "ba":
        filt_ = cast("BAType", filt_)
        return form, signal.bilinear(*filt_, fs=sample_rate)
    if form == "zpk":
        filt_ = cast("ZpkType", filt_)
        return form, signal.bilinear_zpk(*filt_, fs=sample_rate)

    msg = f"cannot convert '{form}' to digital"
    raise ValueError(msg)


@overload
def parse_filter(
    filt: TapsType | BAType | signal.TransferFunction,
) -> tuple[Literal["ba"], BAType]: ...
@overload
def parse_filter(
    filt: SosType,
) -> tuple[Literal["sos"], SosType]: ...
@overload
def parse_filter(
    filt: ZpkCompatible | LinearTimeInvariant,
) -> tuple[Literal["zpk"], ZpkType]: ...

def parse_filter(
    filt: FilterCompatible,
) -> tuple[Literal["ba", "sos", "zpk"], BAType | ZpkType | SosType]:
    """Parse arbitrary input args into a TF, ZPK, or SOS filter definition.

    No transformations are applied to the filter, it is simply unpacked
    into a standard form.

    Parameters
    ----------
    filt : `numpy.ndarray` or `tuple`
        Filter definition.
        Any of the following formats are supported:

        - FIR taps as a 1D `numpy.ndarray`
        - IIR transfer function as a 2-tuple of 1D `numpy.ndarray`s
        - IIR zero-pole-gain as a 3-tuple of 1D `numpy.ndarray`s and a `float`
        - IIR second-order-sections as a 2D `numpy.ndarray`
        - `scipy.signal.lti` or `scipy.signal.dlti` object

    Returns
    -------
    ftype : `str`
        Either ``'ba'`` ``'zpk'`` or ``'sos'``.
    filt : `numpy.ndarray` or `tuple`
        The filter components for the returned ``ftype``.
        If ``'ba'``, a 2-tuple of numerator and denominator arrays.
        If ``'zpk'``, a 3-tuple of zeros, poles, and gain.
        If ``'sos'``, a 2D array of second-order sections.
    """
    # Parse SOS filter
    if is_sos(filt):  # sos
        return "sos", cast("SosType", filt)

    # Parse FIR taps
    if isinstance(filt, numpy.ndarray) and filt.ndim == 1:  # fir
        b, a = cast("numpy.ndarray[tuple[int]]", filt), numpy.ones(1)
        return "ba", (b, a)

    # Use lti to parse the rest
    if isinstance(filt, signal.lti | signal.dlti):
        lti_ = filt
    else:
        lti_ = signal.lti(*filt)  # ty: ignore[invalid-argument-type]
    if isinstance(lti_, signal.TransferFunction):
        return "ba", (lti_.num, lti_.den)
    if not isinstance(lti_, signal.ZerosPolesGain):
        lti_ = lti_.to_zpk()      # ty: ignore[invalid-argument-type]
    return "zpk", (lti_.zeros, lti_.poles, lti_.gain)


def _prewarp_zpk(
    zeros: Array1D,
    poles: Array1D,
    gain: float,
    fs: float,
) -> tuple[Array1D, Array1D, float]:
    """Apply prewarping to analogue ZPK before bilinear transform.

    This implements the GDS prewarping step (iirutil.cc lines 154-158):
    For each pole/zero at frequency ω (rad/s):
        mag = |ω|
        g = (2*fs / mag) * tan(mag / (2*fs))
        ω_warped = ω * g
        gain *= g  (for zeros: divide, for poles: multiply)

    Parameters
    ----------
    zeros : `numpy.ndarray`
        Zeros in rad/s.

    poles : `numpy.ndarray`
        Poles in rad/s.

    gain : `float`
        System gain.

    fs : `float`
        Sampling frequency in Hz.

    Returns
    -------
    zeros_warped : `numpy.ndarray`
        Prewarped zeros.

    poles_warped : `numpy.ndarray`
        Prewarped poles.

    gain_warped : `float`
        Adjusted gain accounting for prewarping.

    Notes
    -----
    This prewarping step ensures that after the bilinear transform,
    the digital filter's frequency response matches the analog design
    at the specified pole/zero frequencies. This is the default behavior
    in LIGO GDS (fPrewarp = true).
    """
    zeros_warped = zeros.copy()
    poles_warped = poles.copy()
    gain_warped = gain

    fs2 = 2.0 * fs

    # Prewarp zeros
    for i in range(len(zeros)):
        mag = numpy.abs(zeros[i])
        if mag > 0:
            g = (fs2 / mag) * numpy.tan(mag / fs2)
            zeros_warped[i] = zeros[i] * g
            gain_warped /= g  # Gain adjustment for zeros

    # Prewarp poles
    for i in range(len(poles)):
        mag = numpy.abs(poles[i])
        if mag > 0:
            g = (fs2 / mag) * numpy.tan(mag / fs2)
            poles_warped[i] = poles[i] * g
            gain_warped *= g  # Gain adjustment for poles

    return zeros_warped, poles_warped, gain_warped


def prepare_analog_filter(
    filt: FilterCompatible,
    *,
    unit: str | UnitBase = "Hz",
    normalize_gain: bool = False,
) -> tuple[Literal["ba", "zpk", "sos"], BAType | ZpkType | SosType]:
    """Prepare an analog filter by parsing and converting units.

    This handles:

    1. Parsing filter specification
    2. Converting Hz → rad/s for ZPK filters (if unit='Hz')
    3. Applying gain normalization (if requested)

    Does NOT:

    - Convert to digital
    - Extract gain for SOS
    - Apply prewarping

    Parameters
    ----------
    filt : filter specification
        Filter as ``(b, a)``, ``(z, p, k)``, sos array,
        or `~scipy.signal.lti` object.

    unit : `str`, optional
        For analogue ZPK filters, the units in which the zeros and poles are
        specified. Either ``'Hz'`` or ``'rad/s'`` (default).

    normalize_gain : `bool`, optional
        Whether to normalize the gain when converting from Hz to rad/s.

        - `False` (default):
          Multiply zeros/poles by -2π but leave gain unchanged.
          This matches the LIGO GDS **'f' plane** convention
          (``plane='f'`` in ``s2z()``).

        - `True`:
          Normalize gain to preserve frequency response magnitude.
          Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
          Use this when your filter was designed with the transfer
          function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
          This matches the LIGO GDS **'n' plane** convention
          (``plane='n'`` in ``s2z()``).

        Ignored when `unit='rad/s'`.

    Returns
    -------
    form : `str`
        Filter form: ``'ba'``, ``'zpk'``, or ``'sos'``.

    filt : `tuple` or `numpy.ndarray`
        Filter coefficients in the identified form.

    See Also
    --------
    prepare_digital_filter
        Prepare filter for digital filtering with prewarping and gain extraction.

    parse_filter
        Parse filter specification without unit conversion.
    """
    form, filter_tuple = parse_filter(filt)

    if form == "zpk" and _is_hertz(unit):
        filter_tuple = cast("ZpkType", filter_tuple)
        filter_tuple = _convert_zpk_units(
            filter_tuple,
            unit=unit,
            normalize_gain=normalize_gain,
        )

    return form, filter_tuple


@overload
def prepare_digital_filter(
    filt: FilterCompatible,
    *,
    analog: bool = ...,
    sample_rate: QuantityLike = ...,
    unit: str | UnitBase = ...,
    normalize_gain: bool = ...,
    output: Literal["ba"] = ...,
) -> BAType: ...
@overload
def prepare_digital_filter(
    filt: FilterCompatible,
    *,
    analog: bool = ...,
    sample_rate: QuantityLike = ...,
    unit: str | UnitBase = ...,
    normalize_gain: bool = ...,
    output: Literal["zpk"] = ...,
) -> ZpkType: ...
@overload
def prepare_digital_filter(
    filt: FilterCompatible,
    *,
    analog: bool = ...,
    sample_rate: QuantityLike = ...,
    unit: str | UnitBase = ...,
    normalize_gain: bool = ...,
    output: Literal["sos"] = ...,
) -> SosType: ...

def prepare_digital_filter(
    filt: FilterCompatible,
    *,
    analog: bool = False,
    sample_rate: QuantityLike = 1.0,
    unit: str | UnitBase = "Hz",
    normalize_gain: bool = False,
    prewarp: bool = True,
    output: Literal["zpk", "ba", "sos"] = "zpk",
) -> BAType | ZpkType | SosType:
    """Prepare a filter for digital filtering.

    This function parses the input filter specification, optionally converts
    an analog filter to digital using prewarping + bilinear transform,
    and returns the filter in the requested format.

    For incoming digital filters, this function basically does nothing.

    Parameters
    ----------
    filt : filter specification
        Filter as ``(b, a)``, ``(z, p, k)``, sos array,
        or `~scipy.signal.lti` object.

        For digital filters (``analog=False``), the input filter coefficients
        are assumed to already be in digital form (z-domain).

        For analog filters (``analog=True``), the input filter coefficients
        should be in the units specified by ``unit`` (default: Hz).

    analog : `bool`, optional
        When `True`, the input filter is analog and will be converted to
        digital using prewarping + bilinear transform (GDS method).
        When `False` (default), the input filter is already digital.

    sample_rate : `float`, `~astropy.units.Quantity`, optional
        Sampling frequency (Hz) of the digital system.
        Required when ``analog=True``.

    unit : `str`, optional
        For analogue ZPK filters, the units in which the zeros and poles are
        specified. Either ``'Hz'`` or ``'rad/s'`` (default).

    normalize_gain : `bool`, optional
        Whether to normalize the gain when converting from Hz to rad/s.

        - `False` (default):
          Multiply zeros/poles by -2π but leave gain unchanged.
          This matches the LIGO GDS **'f' plane** convention
          (``plane='f'`` in ``s2z()``).

        - `True`:
          Normalize gain to preserve frequency response magnitude.
          Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
          Use this when your filter was designed with the transfer
          function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
          This matches the LIGO GDS **'n' plane** convention
          (``plane='n'`` in ``s2z()``).

        Only used for analogue filters in Hz (``analog=True, unit="Hz"``).

    prewarp : `bool`, optional
        If `True`, apply prewarping before bilinear transform (default).
        If `False`, skip prewarping (not recommended).
        Ignored when ``analog=False``.

    output : `str`, optional
        Desired output filter form:

        - ``'zpk'``: zero-pole-gain tuple (default)
        - ``'ba'``: numerator-denominator tuple
        - ``'sos'``: second-order sections array with separated gain

    Returns
    -------
    filt_out : `tuple`
        Filter coefficients in the requested form:

        - ``output='ba'``: 2-tuple of (b, a) arrays
        - ``output='zpk'``: 3-tuple of (zeros, poles, gain)
        - ``output='sos'``: 2-tuple of (sos_array, gain)

        For ``output='sos'``, the returned SOS array has unit gain in all
        sections, and the overall gain is returned separately. This must be
        applied to the filter output: ``filtered_data * gain``.

    Raises
    ------
    ValueError
        If ``analog=True`` but ``sample_rate`` is not provided.

    See Also
    --------
    prepare_analog_filter
        Prepare analog filter without digital conversion.

    scipy.signal.sosfilt
        Apply SOS filter to data.

    Notes
    -----
    **Prewarping**

    Unlike `scipy.signal.bilinear` and `scipy.signal.bilinear_zpk` which
    do NOT perform prewarping, this function applies prewarping by default.
    Prewarping ensures the digital filter's frequency response matches the
    analogue design at the pole/zero frequencies.

    Examples
    --------
    Prepare a digital filter from analog specification:

    >>> from gwpy.signal.filter_design import prepare_digital_filter
    >>> zpk = ([100], [10], 1.0)  # 100 Hz zero, 10 Hz pole, gain 1.0
    >>> sos = prepare_digital_filter(
    ...     zpk,
    ...     analog=True,
    ...     sample_rate=4096,
    ...     unit='Hz',
    ...     output='sos',
    ... )

    Apply to data:

    >>> from scipy import signal
    >>> filtered = signal.sosfilt(sos, data)
    """
    # Parse filter to standardize format
    form, filter_tuple = parse_filter(filt)

    # Convert sample_rate to Hz
    sample_rate = _as_float(sample_rate, unit="Hz")

    # Step 1-2: Prepare analog filter (parse + unit conversion)
    if analog:
        if form == "zpk" and unit == "Hz":
            filter_tuple = cast("ZpkType", filter_tuple)
            filter_tuple = _convert_zpk_units(
                filter_tuple,
                unit=unit,
                normalize_gain=normalize_gain,
            )

        # Convert to ZPK for prewarping and bilinear transform
        zeros, poles, gain = _transform(filter_tuple, form, "zpk")

        # Manual prewarping (GDS default: enabled)
        if prewarp:
            zeros, poles, gain = _prewarp_zpk(zeros, poles, gain, sample_rate)

        # Apply bilinear transform
        filter_tuple = signal.bilinear_zpk(zeros, poles, gain, fs=sample_rate)
        form = "zpk"

    # For ba/zpk output, transform as needed
    return _transform(filter_tuple, form, output)


# -- FIR from transfer function -------

def _truncate_transfer(
    transfer: TArray,
    nsamples: int = 5,
    ncorner: int | None = None,
) -> TArray:
    """Smoothly zero the edges of a frequency domain transfer function.

    Parameters
    ----------
    transfer : `numpy.ndarray`
        Transfer function to start from, must have at least ten samples.

    nsamples : `int`, optional
        Number of samples to taper on each side.

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
    scipy.signal.windows.tukey
    """
    nsamp = transfer.size
    ncorner = ncorner if ncorner else 0
    out = transfer.copy()
    out[0:ncorner] = 0
    # Apply tukey window to taper edges (5 samples on each side)
    ntaper = nsamp - ncorner
    if ntaper > 0:
        window_full = tukey(2 * nsamples, alpha=1.0)
        out[ncorner:ncorner+nsamples] *= window_full[:nsamples]
        out[nsamp-nsamples:nsamp] *= window_full[-nsamples:]
    return out


def _truncate_impulse(
    impulse: TArray,
    ntaps: int,
    window: WindowLike = "hann",
) -> TArray:
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
    transfer: Array1D,
    ntaps: int,
    window: WindowLike = "hann",
    ncorner: int | None = None,
) -> NDArray:
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
    transfer = _truncate_transfer(transfer, ncorner=ncorner)
    # compute and truncate the impulse response
    impulse = npfft.irfft(transfer)
    impulse = _truncate_impulse(
        impulse,
        ntaps=ntaps,
        window=window,
    )
    # wrap around and normalise to construct the filter
    return numpy.roll(impulse, int(ntaps / 2 - 1))[0:ntaps]


# -- Filter design -------------------

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


@overload
def _design(
    type_: Literal["iir"],
    wp: float | ArrayLike,
    ws: float | ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    output: Literal["zpk"],
    **kwargs,
) -> ZpkType: ...

@overload
def _design(
    type_: Literal["iir"],
    wp: float | ArrayLike,
    ws: float | ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    output: Literal["ba"],
    **kwargs,
) -> BAType: ...

@overload
def _design(
    type_: Literal["iir"],
    wp: float | ArrayLike,
    ws: float | ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    output: Literal["sos"],
    **kwargs,
) -> SosType: ...

@overload
def _design(
    type_: Literal["fir"],
    wp: float | ArrayLike,
    ws: float | ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    **kwargs,
) -> TapsType: ...

def _design(
    type_: str,
    wp: float | ArrayLike,
    ws: float | ArrayLike,
    sample_rate: float,
    gpass: float,
    gstop: float,
    output: Literal["zpk", "ba", "sos"] = "zpk",
    **kwargs,
) -> FilterType:
    """Design an IIR or FIR filter."""
    designer: Callable
    if type_ == "iir":
        kwargs["output"] = output
        designer = _design_iir
    else:
        designer = _design_fir
    return designer(wp, ws, sample_rate, gpass, gstop, **kwargs)


# -- user methods --------------------

@overload
def lowpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: QuantityLike | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["zpk"] = "zpk",
    **kwargs,
) -> ZpkType: ...

@overload
def lowpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: QuantityLike | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["ba"] = "ba",
    **kwargs,
) -> BAType: ...

@overload
def lowpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: QuantityLike | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["sos"] = "sos",
    **kwargs,
) -> SosType: ...

@overload
def lowpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: QuantityLike | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["fir"] = "fir",
    **kwargs,
) -> TapsType: ...

def lowpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: QuantityLike | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir", "fir"] = "iir",  # noqa: A002
    output: Literal["zpk", "ba", "sos"] = "zpk",
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

    output : `str`, optional, default: ``'zpk'``
        The output format for an IIR filter,
        either ``'zpk'``, ``'ba'``, or ``'sos'``.

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
        output=output,
        **kwargs,
    )


@overload
def highpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: float | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["zpk"] = "zpk",
    **kwargs,
) -> ZpkType: ...

@overload
def highpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: float | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["ba"] = "ba",
    **kwargs,
) -> BAType: ...

@overload
def highpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: float | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["sos"] = "sos",
    **kwargs,
) -> SosType: ...

@overload
def highpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: float | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["fir"] = "fir",
    **kwargs,
) -> TapsType: ...

def highpass(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    fstop: float | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir", "fir"] = "iir",  # noqa: A002
    output: Literal["zpk", "ba", "sos"] = "zpk",
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

    output : `str`, optional, default: ``'zpk'``
        The output format for an IIR filter,
        either ``'zpk'``, ``'ba'``, or ``'sos'``.

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
        output=output,
        **kwargs,
    )


@overload
def bandpass(
    flow: QuantityLike,
    fhigh: QuantityLike,
    sample_rate: QuantityLike,
    fstop: tuple[QuantityLike, QuantityLike] | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["zpk"] = "zpk",
    **kwargs,
) -> ZpkType: ...

@overload
def bandpass(
    flow: QuantityLike,
    fhigh: QuantityLike,
    sample_rate: QuantityLike,
    fstop: tuple[QuantityLike, QuantityLike] | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["ba"] = "ba",
    **kwargs,
) -> BAType: ...

@overload
def bandpass(
    flow: QuantityLike,
    fhigh: QuantityLike,
    sample_rate: QuantityLike,
    fstop: tuple[QuantityLike, QuantityLike] | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir"] = "iir",
    output: Literal["sos"] = "sos",
    **kwargs,
) -> SosType: ...

@overload
def bandpass(
    flow: QuantityLike,
    fhigh: QuantityLike,
    sample_rate: QuantityLike,
    fstop: tuple[QuantityLike, QuantityLike] | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["fir"] = "fir",
    **kwargs,
) -> TapsType: ...

def bandpass(
    flow: QuantityLike,
    fhigh: QuantityLike,
    sample_rate: QuantityLike,
    fstop: tuple[QuantityLike, QuantityLike] | None = None,
    gpass: float = 2,
    gstop: float = 30,
    type: Literal["iir", "fir"] = "iir",  # noqa: A002
    output: Literal["zpk", "ba", "sos"] = "zpk",
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

    output : `str`, optional, default: ``'zpk'``
        The output format for an IIR filter,
        either ``'zpk'``, ``'ba'``, or ``'sos'``.

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
        [flow, fhigh],
        fstop,
        sample_rate,
        gpass,
        gstop,
        output=output,
        **kwargs,
    )


@overload
def notch(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    type: Literal["iir"] = "iir",
    output: Literal["zpk"] = "zpk",
    **kwargs,
) -> ZpkType: ...

@overload
def notch(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    type: Literal["iir"] = "iir",
    output: Literal["ba"] = "ba",
    **kwargs,
) -> BAType: ...

@overload
def notch(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    type: Literal["iir"] = "iir",
    output: Literal["sos"] = "sos",
    **kwargs,
) -> SosType: ...

def notch(
    frequency: QuantityLike,
    sample_rate: QuantityLike,
    type: Literal["iir"] = "iir",  # noqa: A002
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


# -- Frequency response --------------

def frequency_response(
    filt: FilterCompatible,
    frequencies: NDArray | int | None,
    *,
    analog: bool = False,
    sample_rate: QuantityLike = 1.0,
    unit: str | UnitBase = "rad/s",
    normalize_gain: bool = False,
) -> tuple[numpy.ndarray[tuple[int]], numpy.ndarray[tuple[int]]]:
    """Compute the frequency response of a filter at given frequencies.

    Parameters
    ----------
    filt : `FilterCompatible`
        Filter definition in one of the supported formats.

    frequencies : `numpy.ndarray`
        Frequencies (in Hz) at which to compute the response.

    analog : `bool`, optional
        Whether the filter is analogue (`True`) or digital (`False`).

    sample_rate : `astropy.units.Quantity`-like, optional
        Sample rate of the digital data (only used if `analog` is `False`).

    unit : `str`, optional
        For analogue ZPK filters, the units in which the zeros and poles are
        specified. Either ``'Hz'`` or ``'rad/s'`` (default).

    normalize_gain : `bool`, optional
        Whether to normalize the gain when converting from Hz to rad/s.

        - `False` (default):
          Multiply zeros/poles by -2π but leave gain unchanged.
          This matches the LIGO GDS **'f' plane** convention
          (``plane='f'`` in ``s2z()``).

        - `True`:
          Normalize gain to preserve frequency response magnitude.
          Gain is scaled by :math:`|∏p_i/∏z_i| · (2π)^{(n_p - n_z)}`.
          Use this when your filter was designed with the transfer
          function :math:`H(f) = k·∏(f-z_i)/∏(f-p_i)` in Hz.
          This matches the LIGO GDS **'n' plane** convention
          (``plane='n'`` in ``s2z()``).

        Only used for analogue filters in Hz (``analog=True, unit="Hz"``).

    Returns
    -------
    frequencies : `numpy.ndarray`
        Frequencies at which the response was computed.
        If ``analog=True, unit='rad/s'``, these are in rad/s, otherwise
        they are in Hz.
    response : `numpy.ndarray`
        Frequency response of the filter at the given frequencies.

    See Also
    --------
    scipy.signal.freqs
    scipy.signal.freqz
    scipy.signal.freqz_sos
    scipy.signal.freqs_zpk
    scipy.signal.freqz_zpk
    """
    filt_: BAType | ZpkType | SosType
    if analog:
        form, filt_ = prepare_analog_filter(
            filt,
            unit=unit,
            normalize_gain=normalize_gain,
        )
    else:
        form = "zpk"
        filt_ = prepare_digital_filter(
            filt,
            sample_rate=sample_rate,
            unit=unit,
            normalize_gain=normalize_gain,
            output=form,
        )

    # Use angular frequencies
    if frequencies is not None and not isinstance(frequencies, int):
        frequencies *= TWO_PI

    # Compute the filter response
    if analog and form == "sos":
        msg = "analog SOS frequency response not supported"
        raise ValueError(msg)

    # Digital SOS
    if form == "sos":
        wfreq, fresp = signal.freqz_sos(filt_, worN=frequencies)
    # Analogue ZPK
    elif analog and form == "zpk":
        filt_ = cast("ZpkType", filt_)
        wfreq, fresp = signal.freqs_zpk(*filt_, worN=frequencies)
    # Digital ZPK
    elif form == "zpk":
        filt_ = cast("ZpkType", filt_)
        wfreq, fresp = signal.freqz_zpk(*filt_, worN=frequencies)
    # Analogue BA
    elif analog:
        filt_ = cast("BAType", filt_)
        wfreq, fresp = signal.freqs(*filt_, worN=frequencies)
    # Digital BA
    else:
        filt_ = cast("BAType", filt_)
        wfreq, fresp = signal.freqz(*filt_, worN=frequencies)

    # Convert from digital angular frequency to Hz
    if not analog:
        wfreq *= _as_float(sample_rate, unit="Hz") / TWO_PI
    elif unit == "Hz":
        wfreq /= TWO_PI


    return (
        wfreq,
        fresp,
    )
