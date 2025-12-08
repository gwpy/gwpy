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

"""Common utilities for frequency-domain operations.

This module holds code used by both the `FrequencySeries` and `Spectrogram`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy

from ..signal import filter_design

if TYPE_CHECKING:
    from typing import TypeVar

    from astropy.units.typing import QuantityLike

    from ..frequencyseries import FrequencySeries
    from ..signal.filter_design import FilterCompatible
    from ..spectrogram import Spectrogram

    FreqDomainData = TypeVar("FreqDomainData", FrequencySeries, Spectrogram)

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


def _fdfilter(
    data: FreqDomainData,
    filt: FilterCompatible,
    *,
    sample_rate: QuantityLike | None = None,
    analog: bool = False,
    unit: str = "rad/s",
    normalize_gain: bool = False,
    inplace: bool = False,
    **kwargs,
) -> FreqDomainData:
    """Filter a frequency-domain data object.

    See Also
    --------
    gwpy.frequencyseries.FrequencySeries.filter
    gwpy.spectrogram.Spectrogram.filter
    """
    freqs = data.frequencies.to("Hz").value
    if sample_rate is None:
        sample_rate = freqs[-1] * 2

    _, fresp = numpy.abs(filter_design.frequency_response(
        filt,
        freqs,
        analog=analog,
        sample_rate=sample_rate,
        unit=unit,
        normalize_gain=normalize_gain,
        **kwargs,
    ))

    # apply to array
    if inplace:
        data *= fresp
        return data
    return data * fresp
