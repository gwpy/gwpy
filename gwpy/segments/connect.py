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

"""Unified I/O read/write for Segment objects."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from ..io.registry import (
    UnifiedRead,
    UnifiedWrite,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ..io.utils import NamedReadable
    from . import (
        DataQualityDict,
        DataQualityFlag,
        SegmentList,
    )


class SegmentListRead(UnifiedRead):
    """Read segments from file into a `SegmentList`.

    Parameters
    ----------
    filename : `str`
        Path of file to read.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    coalesce : `bool`, optional
        If `True`, coalesce the segment list before returning,
        otherwise return exactly as contained in file(s).

    kwargs
        Other keyword arguments depend on the format,
        see `SegmentList.read.help(format=<format>)` for details, or
        see the online documentation (:ref:`gwpy-segments-io`).

    Returns
    -------
    segmentlist : `SegmentList`
        `SegmentList` active and known segments read from file.

    Raises
    ------
    IndexError
        if ``source`` is an empty list

    Notes
    -----"""

    def merge(  # type: ignore[override]
        self,
        items: Sequence[SegmentList],
        *,
        coalesce: bool = False,
    ) -> SegmentList:
        """Combine the `SegmentList` from each file into a single object."""
        out = self._cls(seg for seglist in items for seg in seglist)
        if coalesce:
            return out.coalesce()
        return out


class SegmentListWrite(UnifiedWrite):
    """Write this `SegmentList` to a file.

    Arguments and keywords depend on the output format, see the
    online documentation for full details for each format.

    Parameters
    ----------
    target : `str`
        Output filename.

    Notes
    -----"""


class DataQualityFlagRead(UnifiedRead):
    """Read segments from file into a `DataQualityFlag`.

    Parameters
    ----------
    filename : `str`
        Path of file to read.

    name : `str`, optional
        Name of flag to read from file, otherwise read all segments.

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    coltype : `type`, optional, default: `float`
        Datatype to force for segment times, only valid for
        ``format='segwizard'``.

    strict : `bool`, optional, default: `True`
        Require segment start and stop times match printed duration,
        only valid for ``format='segwizard'``.

    coalesce : `bool`, optional
        If `True` coalesce the all segment lists before returning,
        otherwise return exactly as contained in file(s).

    parallel : `int`, optional, default: 1
        Number of threads to use for parallel reading of multiple files.

    verbose : `bool`, optional, default: `False`
        Print a progress bar showing read status.

    Returns
    -------
    dqflag : `DataQualityFlag`
        formatted `DataQualityFlag` containing the active and known
        segments read from file.

    Raises
    ------
    IndexError
        If ``source`` is an empty list.

    Notes
    -----"""

    def merge(  # type: ignore[override]
        self,
        items: Sequence[DataQualityFlag],
        *,
        coalesce: bool = False,
    ) -> DataQualityFlag:
        """Combine `DataQualityFlag` from each file into a single object."""
        itrtr = iter(items)
        out = next(itrtr)
        for flag in itrtr:
            out.known += flag.known
            out.active += flag.active
        if coalesce:
            return out.coalesce()
        return out

    def __call__(
        self,
        source: NamedReadable | list[NamedReadable],
        *args,
        coalesce: bool = False,
        **kwargs,
    ) -> DataQualityFlag:
        """Read data into a `DataQualityFlag`."""
        if "flag" in kwargs:  # pragma: no cover
            warnings.warn(
                "'flag' keyword was renamed 'name', "
                "this warning will result in an error in the future",
                DeprecationWarning,
                stacklevel=2,
            )
            kwargs.setdefault("name", kwargs.pop("flag"))

        return super().__call__(
            source,
            *args,
            coalesce=coalesce,
            **kwargs,
        )


class DataQualityFlagWrite(UnifiedWrite):
    """Write this `DataQualityFlag` to file.

    Notes
    -----"""


class DataQualityDictRead(UnifiedRead):
    """Read segments from file into a `DataQualityDict`.

    Parameters
    ----------
    source : `str`
        Path of file to read

    format : `str`, optional
        Source format identifier. If not given, the format will be
        detected if possible. See below for list of acceptable
        formats.

    names : `list`, optional, default: read all names found
        List of names to read, by default all names are read separately.

    coalesce : `bool`, optional
        If `True` coalesce the all segment lists before returning,
        otherwise return exactly as contained in file(s).

    parallel : `int`, optional, default: 1
        Number of threads to use for parallel reading of multiple files.

    verbose : `bool`, optional, default: `False`
        Print a progress bar showing read status.

    Returns
    -------
    flagdict : `DataQualityDict`
        A new `DataQualityDict` of `DataQualityFlag` entries with
        ``active`` and ``known`` segments seeded from the XML tables
        in the given file.

    Notes
    -----"""

    def merge(  # type: ignore[override]
        self,
        inputs: Sequence[DataQualityDict],
        *,
        names: list[str] | None = None,
        on_missing: str = "error",
        coalesce: bool = False,
    ) -> DataQualityDict:
        """Combine a list of `DataQualityDict` objects into one `DataQualityDict`.

        Must be given at least one `DataQualityDict`.
        """
        # check all names are contained
        required = set(names or [])
        found = {name for dqdict in inputs for name in dqdict}
        for name in required - found:  # validate all names are found once
            msg = f"'{name}' not found in any input file"
            if on_missing == "ignore":
                continue
            if on_missing == "warn":
                warnings.warn(msg, stacklevel=2)
                continue
            raise ValueError(msg)

        # combine flags
        out = self._cls()
        for dqdict in inputs:
            for flag in dqdict:
                if flag in out:
                    out[flag].known.extend(dqdict[flag].known)
                    out[flag].active.extend(dqdict[flag].active)
                else:
                    out[flag] = dqdict[flag]
        if coalesce:
            return out.coalesce()
        return out

    def __call__(
        self,
        *args,
        names: list[str] | None = None,
        on_missing: str = "error",
        coalesce: bool = False,
        **kwargs,
    ):
        return super().__call__(
            *args,
            names=names,
            on_missing=on_missing,
            coalesce=coalesce,
            **kwargs,
        )


class DataQualityDictWrite(UnifiedWrite):
    """Write this `DataQualityDict` to file.

    Notes
    -----"""
