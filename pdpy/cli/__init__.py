# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2014-2020)
#
# This file is part of pyDischarge.
#
# pyDischarge is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# pyDischarge is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with pyDischarge.  If not, see <http://www.gnu.org/licenses/>.

"""Command-line utilities for pyDischarge

The `pydischarge.cli` module provides methods and functionality to power the
`pydischarge-plot` command-line executable (distributed with pyDischarge).
"""

from collections import OrderedDict as _od

from .timeseries import TimeSeries
from .spectrum import Spectrum
from .spectrogram import Spectrogram
from .coherence import Coherence
from .coherencegram import Coherencegram
from .qtransform import Qtransform
from .transferfunction import TransferFunction

__author__ = 'Joseph Areeda <joseph.areeda@ligo.org>'

PRODUCTS = _od((x.action, x) for x in (
    TimeSeries,
    Spectrum,
    Spectrogram,
    Coherence,
    Coherencegram,
    Qtransform,
    TransferFunction,
))
