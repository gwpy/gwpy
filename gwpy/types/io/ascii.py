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

"""Read a `Series` from an ASCII file.

These files should be in (at least) two-column x,y format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import (
    column_stack,
    loadtxt,
    savetxt,
)

from ...io.registry import identify_factory
from .. import (
    Array2D,
    Series,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path
    from typing import IO

__author__ = "Duncan Macleod <duncan.macleod@ligo.org>"


# -- read ----------------------------

def read_ascii_series(
    input_: str | Path | IO,
    array_type: type[Series] = Series,
    *,
    unpack: bool = True,
    **kwargs,
) -> Series:
    """Read a `Series` from an ASCII file.

    Parameters
    ----------
    input_ : `str`, `file`
        File to read.

    array_type : `type`
        Desired return type.

    unpack : `bool`
        If `True` (default), unpack the data into separate columns.
        This argument is passed to `numpy.loadtxt`.

    kwargs
        Additional keyword arguments to pass to `numpy.loadtxt`.

    Returns
    -------
    Series
        The data read from the file, as a `Series` object.

    See Also
    --------
    numpy.loadtxt
        For documentation of keyword arguments.
    """
    # load data
    data = loadtxt(input_, unpack=unpack, **kwargs)
    # separate first column as index
    xindex = data[0]
    # reshape other columns as data
    if array_type._ndim == 1:
        ydata = data[1:][0]  # one column
    else:
        ydata = data[1:].T  # ND array
    return array_type(ydata, xindex=xindex)


# -- write ---------------------------

def write_ascii_series(
    series: Series,
    output: str | Path | IO,
    **kwargs,
) -> None:
    """Write a `Series` to a file in ASCII format.

    Parameters
    ----------
    series : `Series`
        Data series to write.

    output : `str`, `file`
        File to write to.

    kwargs
        Additional keyword arguments to pass to `numpy.savetxt`.

    See Also
    --------
    numpy.savetxt
        For documentation of keyword arguments.
    """
    xarr = series.xindex.value
    yarr = series.value
    savetxt(output, column_stack((xarr, yarr)), **kwargs)


# -- register ------------------------


def register_ascii_io(
    array_type: type[Series],
    format: str = "txt",  # noqa: A002
    *,
    identify: bool = True,
    reader: Callable = read_ascii_series,
    writer: Callable = write_ascii_series,
    **defaults,
) -> None:
    """Register ASCII read/write/identify methods for the given type."""

    def _read(source: str | Path | IO, **kwargs) -> Series:
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        return reader(
            source,
            array_type=array_type,
            **kwgs,
        )

    def _write(
        series: Series,
        output: str | Path | IO,
        **kwargs,
    ) -> None:
        kwgs = defaults.copy()
        kwgs.update(kwargs)
        return writer(series, output, **kwgs)

    array_type.read.registry.register_reader(format, array_type, _read)
    array_type.write.registry.register_writer(format, array_type, _write)
    if identify:
        array_type.read.registry.register_identifier(
            format,
            array_type,
            identify_factory(format),
        )


register_ascii_io(Series, "csv", delimiter=",")
register_ascii_io(Array2D, "csv", delimiter=",")
