# Copyright (C) Cardiff University (2024-)
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

"""Unified I/O read/write for Array objects."""

from __future__ import annotations

import typing
from functools import partial

from ..io.registry import (
    UnifiedRead,
    UnifiedWrite,
)

if typing.TYPE_CHECKING:
    from .series import Series


def _combine_series(
    listofseries: list[Series],
    pad: float | None = None,
    gap: str | None = None,
) -> Series:
    """Combine a list of `Series` objects into one `Series`.

    Must be given at least one series.
    """
    out = listofseries.pop(0)
    while listofseries:
        out.append(listofseries.pop(0), gap=gap, pad=pad)
    return out


# -- Series --------------------------

class SeriesRead(UnifiedRead):
    """Read data into a `Series`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    source : `str`, `pathlib.Path`, `file`, `list`
        Source of data, any of the following:

        - open (readable) file object
        - path to a file
        - list of files (open or paths)

    format : `str`,
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    kwargs
        Other keywords arguments depend on the format, see the online
        documentation for details.

    Returns
    -------
    data : `Series`

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""
    def __call__(
        self,
        *args,
        pad: float | None = None,
        gap: str | None = None,
        **kwargs,
    ):
        combiner = partial(_combine_series, pad=pad, gap=gap)
        return super().__call__(
            combiner,
            *args,
            **kwargs,
        )


class SeriesWrite(UnifiedWrite):
    """Write this `Series` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`, `Path`, `file`
        Output file path or file opened in write mode.

    format : `str`, optional
        Output format identifier. Default format will be detected if possible.
        See below for list of acceptable formats.

    Notes
    -----"""


# -- Array2D -------------------------

class Array2DRead(UnifiedRead):
    """Read data into a `Array2D`.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    source : `str`, `pathlib.Path`, `file`, `list`
        Source of data, any of the following:

        - open (readable) file object
        - path to a file
        - list of files (open or paths)

    format : `str`,
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    kwargs
        Other keywords arguments depend on the format, see the online
        documentation for details.

    Returns
    -------
    data : `Array2D`

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""
    def __call__(
        self,
        *args,
        pad: float | None = None,
        gap: str | None = None,
        **kwargs,
    ):
        combiner = partial(_combine_series, pad=pad, gap=gap)
        return super().__call__(
            combiner,
            *args,
            **kwargs,
        )


class Array2DWrite(UnifiedWrite):
    """Write this `Array2D` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format, the
    parameters below are common to most formats.

    Parameters
    ----------
    target : `str`, `Path`, `file`
        Output file path or file opened in write mode.

    format : `str`, optional
        Output format identifier. Default format will be detected if possible.
        See below for list of acceptable formats.

    Notes
    -----"""
