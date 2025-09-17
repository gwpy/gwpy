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

"""Utilities for FFT routines."""

from astropy import units

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def scale_timeseries_unit(
    tsunit: units.UnitBase,
    scaling: str = "density",
) -> units.UnitBase:
    """Scale the unit of a `TimeSeries` to match that of a `FrequencySeries`.

    Parameters
    ----------
    tsunit : `~astropy.units.UnitBase`
        Input unit from `TimeSeries`.

    scaling : `str`
        Type of frequency series, either ``"density"`` for a PSD, or
        ``"spectrum"`` for a power spectrum.

    Returns
    -------
    unit : `~astropy.units.Unit`
        Unit to be applied to the resulting `FrequencySeries`.
    """
    # set units
    if scaling == "density":
        baseunit = units.Hertz
    elif scaling == "spectrum":
        baseunit = units.dimensionless_unscaled
    else:
        msg = f"unknown scaling: '{scaling}'"
        raise ValueError(msg)
    if tsunit:
        specunit = tsunit ** 2 / baseunit
    else:
        specunit = baseunit ** -1
    return specunit
