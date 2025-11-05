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

"""Unit tests for :mod:`gwpy.cli.coherencegram`."""

from ... import cli
from .test_coherence import TestCoherenceProduct as _TestCoherenceProduct
from .test_spectrogram import TestSpectrogramProduct as _TestSpectrogramProduct


class TestCoherencegramProduct(_TestSpectrogramProduct):
    """Tests for `CoherencegramProduct`."""

    TEST_CLASS = cli.CoherencegramProduct
    ACTION = "coherencegram"
    TEST_ARGS = _TestCoherenceProduct.TEST_ARGS

    def test_finalize_arguments(self, prod):
        """Test that `CoherencegramProduct.finalize_arguments` sets defaults."""
        assert prod.args.cmap == "plasma"
        assert prod.args.color_scale == "linear"
        assert prod.args.imin == 0.
        assert prod.args.imax == 1.

    def test_get_suptitle(self, prod):
        """Test `CoherencegramProduct.get_suptitle()`."""
        assert prod.get_suptitle() == (
            "Coherence spectrogram: "
            f"{prod.chan_list[0]} vs {prod.chan_list[1]}"
        )
