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

"""Unified I/O read/write for `FrequencySeries` objects."""

from __future__ import annotations

from ..types.connect import (
    Array2DRead,
    Array2DWrite,
    SeriesRead,
    SeriesWrite,
)

# -- FrequencySeries -----------------

class FrequencySeriesRead(SeriesRead):
    """Read data into a `FrequencySeries`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    source : `str`, `os.PathLike`, `file`, or `list` of these
        Source of data, any of the following:

        - Path of a single data file
        - List of data file paths
        - Path of LAL-format cache file

    args
        Other arguments are (in general) specific to the given ``format``.

    format : `str`, optional
        Source format identifier.
        If not given, the format will be detected if possible.
        See below for list of acceptable formats.

    kwargs
        Other keywords are (in general) specific to the given ``format``.

    Raises
    ------
    IndexError
        If ``source`` is an empty list.

    Notes
    -----"""


class FrequencySeriesWrite(SeriesWrite):
    """Write this `FrequencySeries` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        Output filename.

    format : `str`, optional
        Output format identifier.
        If not given, the format will be detected if possible.
        See below for list of acceptable formats.

    Notes
    -----"""


# -- SpectralVariance ----------------

class SpectralVarianceRead(Array2DRead):
    """Read data into a `SpectralVariance`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    source : `str`, `os.PathLike`, `file`, or `list` of these
        Source of data, any of the following:

        - Path of a single data file
        - List of data file paths
        - Path of LAL-format cache file

    args
        Other positional arguments are (in general) specific to the given ``format``.
        See below for list of acceptable formats.

    format : `str`, optional
        Source format identifier.
        If not given, the format will be detected if possible.
        See below for list of acceptable formats.

    kwargs
        Other keywords are (in general) specific to the given ``format``.
        See below for list of acceptable formats.

    Raises
    ------
    IndexError
        If ``source`` is an empty list.

    Notes
    -----"""


class SpectralVarianceWrite(Array2DWrite):
    """Write this `SpectralVariance` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`
        Output filename.

    format : `str`, optional
        Output format identifier.
        If not given, the format will be detected if possible.
        See below for list of acceptable formats.

    Notes
    -----"""
