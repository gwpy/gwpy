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

"""Command-line utilities for GWpy.

The `gwpy.cli` module provides methods and functionality to power the
`gwpy-plot` command-line executable (distributed with GWpy).
"""

# ruff: noqa: TC001

from __future__ import annotations

from .cliproduct import CliProduct
from .timeseries import TimeSeriesProduct
from .spectrum import SpectrumProduct
from .spectrogram import SpectrogramProduct
from .coherence import CoherenceProduct
from .coherencegram import CoherencegramProduct
from .qtransform import QtransformProduct
from .transferfunction import TransferFunctionProduct

__author__ = "Joseph Areeda <joseph.areeda@ligo.org>"

PRODUCTS: dict[str, type[CliProduct]] = {
    x.action: x
    for x in (
        TimeSeriesProduct,
        SpectrumProduct,
        SpectrogramProduct,
        CoherenceProduct,
        CoherencegramProduct,
        QtransformProduct,
        TransferFunctionProduct,
    )
}
