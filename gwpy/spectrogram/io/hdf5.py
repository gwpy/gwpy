# -*- coding: utf-8 -*-
# Copyright (C) Duncan Macleod (2013)
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

"""This module attaches the HDF5 input output methods to the Spectrogram.
"""

from ...types.io.hdf5 import (read_hdf5_array,
                              format_series_attrs, write_hdf5_array)
from ...io import registry as io_registry
from ...io.hdf5 import identify_hdf5
from .. import Spectrogram

__author__ = 'Duncan Macleod <duncan.macleod@ligo.org>'


def read_spectrogram(*args, **kwargs):
    kwargs.setdefault('array_type', Spectrogram)
    return read_hdf5_array(*args, **kwargs)


def format_attrs(specgram):
    """Format default metadata for this `Spectrogram`.
    """
    attrs = format_series_attrs(specgram)
    attrs.update({
        'yunit': str(specgram.yunit),
        'y0': specgram.y0.to(specgram.yunit).value,
        'dy': specgram.dy.to(specgram.yunit).value,
    })
    return attrs


def write_spectrogram(specgram, output, path=None, attrs=None, **kwargs):
    """Write a `Spectrogram` to HDF5.
    """
    if attrs is None:
        attrs = format_attrs(specgram)
    return write_hdf5_array(specgram, output, path=path, attrs=attrs, **kwargs)


# register
io_registry.register_reader('hdf5', Spectrogram, read_spectrogram)
io_registry.register_writer('hdf5', Spectrogram, write_spectrogram)
io_registry.register_identifier('hdf5', Spectrogram, identify_hdf5)
