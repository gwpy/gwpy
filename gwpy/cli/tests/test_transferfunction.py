# Copyright (c) 2021 Evan Goetz
#               2021-2025 Cardiff University
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

"""Unit tests for :mod:`gwpy.cli.transferfunction`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from astropy.time import Time
from numpy.random import default_rng

from ...frequencyseries import FrequencySeries
from .. import TransferFunctionProduct
from .base import _TestFrequencyDomainProduct

if TYPE_CHECKING:
    from typing import ClassVar

__author__ = "Evan Goetz <evan.goetz@ligo.org>"


class TestTransferFunctionProduct(_TestFrequencyDomainProduct):
    """Tests for `gwpy.cli.transferfunction.TransferFunctionProduct`."""

    TEST_CLASS = TransferFunctionProduct
    ACTION = "transferfunction"
    TEST_ARGS: ClassVar[list[str]] = [
        *_TestFrequencyDomainProduct.TEST_ARGS,
        "--chan",
        "Y1:TEST-CHANNEL",
        "--secpfft",
        "0.25",
    ]

    @pytest.fixture
    @classmethod
    def dataprod(cls, prod):
        """Return a `TestTransferFunctionProduct` with data."""
        cls._prod_add_data(prod)
        fftlength = prod.args.secpfft
        for i, ts in enumerate(prod.timeseries):
            nsamp = int(fftlength * 512 / 2.) + 1
            if i % 2 == 0:
                rng = default_rng(i)
                name = f"{prod.timeseries[i+1].name}---{ts.name}"
                fs = FrequencySeries(
                    rng.random(nsamp) + 1j*rng.random(nsamp),
                    x0=0,
                    dx=1/fftlength,
                    channel=prod.timeseries[i+1].channel,
                    name=f"{name}",
                    dtype=complex,
                )
                prod.test_chan = prod.timeseries[i+1].name
            prod.tfs.append(fs)
        return prod

    def test_init(self, prod):
        """Test initialization of `TransferFunctionProduct`."""
        assert prod.chan_list == ["X1:TEST-CHANNEL", "Y1:TEST-CHANNEL"]
        assert prod.ref_chan == prod.chan_list[0]
        assert prod.test_chan == prod.chan_list[1]

    def test_get_suptitle(self, prod):
        """Test suptitle generation."""
        assert prod.get_suptitle() == (
            f"Transfer function: {prod.chan_list[1]}/{prod.chan_list[0]}"
        )

    def test_get_title(self, prod):
        """Test title generation."""
        epoch = prod.start_list[0]
        utc = Time(epoch, format="gps", scale="utc").iso
        t = ", ".join(
            [
                f"{utc} | {epoch} ({prod.duration})",
                f"fftlength={prod.args.secpfft}",
                f"overlap={prod.args.overlap}",
            ],
        )
        assert prod.get_title() == t

    def test_set_plot_properties(self, plotprod):
        """Test setting plot properties."""
        assert plotprod.ax.get_title() == (plotprod.args.title or "")
        assert plotprod.plot.get_suptitle() == (plotprod.args.suptitle or "")
