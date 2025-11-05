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

"""Tests for :mod:`gwpy.cli.coherence`."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Generic,
    TypeVar,
)

from ... import cli
from .base import _TestCliProduct
from .test_spectrum import TestSpectrumProduct as _TestSpectrumProduct

if TYPE_CHECKING:
    from typing import ClassVar

CoherenceProductType = TypeVar("CoherenceProductType", bound=cli.CoherenceProduct)


class TestCoherenceProduct(
    _TestSpectrumProduct[CoherenceProductType],
    Generic[CoherenceProductType],
):
    """Tests for `gwpy.cli.CoherenceProduct`."""

    TEST_CLASS: ClassVar[type[cli.CoherenceProduct]] = cli.CoherenceProduct
    ACTION = "coherence"
    TEST_ARGS: ClassVar[list[str]] = [
        *_TestCliProduct.TEST_ARGS,
        "--chan",
         "Y1:TEST-CHANNEL",
         "--secpfft", "0.25",
    ]

    def test_init_ref_chan(self, prod: CoherenceProductType):
        """Test that `CoherenceProduct.ref_chan` gets initialised properly."""
        assert prod.ref_chan == prod.chan_list[0]

    def test_get_suptitle(self, prod: CoherenceProductType):
        """Test `CoherenceProduct.get_suptitle`."""
        assert prod.get_suptitle() == f"Coherence: {prod.chan_list[0]}"
