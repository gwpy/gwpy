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

"""Unit tests for :mod:`gwpy.cli.spectrum`."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

import pytest
from astropy.time import Time
from numpy.random import default_rng

from ...frequencyseries import FrequencySeries
from .. import SpectrumProduct
from .base import _TestFrequencyDomainProduct

if TYPE_CHECKING:
    from typing import ClassVar

SpectrumProductType = TypeVar("SpectrumProductType", bound=SpectrumProduct)


class TestSpectrumProduct(
    _TestFrequencyDomainProduct[SpectrumProductType],
    Generic[SpectrumProductType],
):
    """Tests for `gwpy.cli.SpectrumProduct`."""

    TEST_CLASS: ClassVar[type[SpectrumProduct]] = SpectrumProduct
    ACTION = "spectrum"

    @pytest.fixture
    @classmethod
    def dataprod(cls, prod: SpectrumProductType) -> SpectrumProductType:
        """Return a `TestFrequencyDomainProduct` with data."""
        cls._prod_add_data(prod)
        fftlength = prod.args.secpfft
        for i, ts in enumerate(prod.timeseries):
            nsamp = int(fftlength * 512 / 2.) + 1
            rng = default_rng(i)
            fs = FrequencySeries(
                rng.random(nsamp),
                x0=0,
                dx=1/fftlength,
                channel=ts.channel,
                name=ts.name,
            )
            prod.spectra.append(fs)
        return prod

    def test_get_title(self, prod: SpectrumProductType):
        """Test `SpectrumProduct.get_title`."""
        epoch = prod.start_list[0]
        utc = Time(epoch, format="gps", scale="utc").iso
        t = ", ".join([
            f"{utc} | {epoch} ({prod.duration})",
            f"fftlength={prod.args.secpfft}",
            f"overlap={prod.args.overlap}",
        ])
        assert prod.get_title() == t

    def test_get_suptitle(self, prod: SpectrumProductType):
        """Test `SpectrumProduct.get_suptitle`."""
        assert prod.get_suptitle() == f"Spectrum: {prod.chan_list[0]}"
