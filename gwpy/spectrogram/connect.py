# Copyright (c) 2024-2025 Cardiff University
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

"""Unified I/O read/write for `Spectrogram` objects."""

from __future__ import annotations

from ..types.connect import (
    Array2DRead,
    Array2DWrite,
)

# -- Spectrogram -----------------

class SpectrogramRead(Array2DRead):
    """Read data into a `Spectrogram`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    source : `str`, `list`
        Source of data, any of the following:

        - `str` path of single data file,
        - `str` path of LAL-format cache file,
        - `list` of paths.

    *args
        Other arguments are (in general) specific to the given
        ``format``.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    **kwargs
        Other keywords are (in general) specific to the given ``format``.

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""


class SpectrogramWrite(Array2DWrite):
    """Write this `Spectrogram` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        output filename

    format : `str`, optional
        output format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    Notes
    -----"""
