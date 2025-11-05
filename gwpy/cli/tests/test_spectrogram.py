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

"""Unit tests for :mod:`gwpy.cli.spectrogram`."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
    cast,
)

import pytest

from .. import SpectrogramProduct
from .base import (
    _TestFFTMixin,
    _TestImageProduct,
    _TestTimeDomainProduct,
)

if TYPE_CHECKING:
    from typing import ClassVar

    from ...spectrogram import Spectrogram

SpectrogramProductType = TypeVar("SpectrogramProductType", bound=SpectrogramProduct)


class TestSpectrogramProduct(
    _TestFFTMixin,
    _TestTimeDomainProduct[SpectrogramProductType],
    _TestImageProduct[SpectrogramProductType],
    Generic[SpectrogramProductType],
):
    """Tests for `gwpy.cli.SpectrogramProduct`."""

    TEST_CLASS: ClassVar[type[SpectrogramProduct]] = SpectrogramProduct
    ACTION = "spectrogram"

    @pytest.fixture
    @classmethod
    def dataprod(cls, prod: SpectrogramProductType) -> SpectrogramProductType:
        """Return a `SpectrogramProduct` with data."""
        cls._prod_add_data(prod)
        prod.result = prod.get_spectrogram()
        return prod

    @pytest.fixture
    @classmethod
    def plotprod(cls, dataprod: SpectrogramProductType) -> SpectrogramProductType:
        """Return a `SpectrogramProduct` with data and a plot."""
        cls._plotprod_init(dataprod)
        result = cast("Spectrogram", dataprod.result)
        dataprod.plot.gca().pcolormesh(result)
        return dataprod

    def test_get_title(self, prod: SpectrogramProductType):
        """Test `SpectrogramProduct.get_title()`."""
        assert prod.get_title() == ", ".join([
            f"fftlength={prod.args.secpfft}",
            f"overlap={prod.args.overlap}",
        ])

    def test_get_suptitle(self, prod: SpectrogramProductType):
        """Test `SpectrogramProduct.get_suptitle()`."""
        assert prod.get_suptitle() == f"Spectrogram: {prod.chan_list[0]}"
